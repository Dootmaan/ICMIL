from model.network import Classifier_1fc, DimReduction
from model.Attention import Attention_Gated as Attention
# from model.Attention import Attention_with_Classifier
import argparse
import torch
from dataset.EmbededFeatsDataset import EmbededFeatsDataset
# torch.autograd.set_detect_anomaly(True)
from sklearn.metrics import roc_auc_score,f1_score,roc_curve
import numpy as np

parser = argparse.ArgumentParser(description='abc')

parser.add_argument('--name', default='abc', type=str)
parser.add_argument('--EPOCH', default=200, type=int)
parser.add_argument('--epoch_step', default='[100]', type=str)
parser.add_argument('--device', default='cuda', type=str)
# parser.add_argument('--isPar', default=False, type=bool)
# parser.add_argument('--log_dir', default='./debug_log', type=str)   ## log file path
# parser.add_argument('--train_show_freq', default=40, type=int)
parser.add_argument('--droprate', default='0', type=float)
parser.add_argument('--droprate_2', default='0', type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
# parser.add_argument('--batch_size', default=1, type=int)
# parser.add_argument('--batch_size_v', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_cls', default=2, type=int)
# parser.add_argument('--mDATA0_dir_train0', default='', type=str)  ## Train Set
# parser.add_argument('--mDATA0_dir_val0', default='', type=str)      ## Validation Set
# parser.add_argument('--mDATA_dir_test0', default='', type=str)         ## Test Set
# parser.add_argument('--numGroup', default=5, type=int)
# parser.add_argument('--total_instance', default=4, type=int)
# parser.add_argument('--numGroup_test', default=4, type=int)
# parser.add_argument('--total_instance_test', default=4, type=int)
parser.add_argument('--mDim', default=512, type=int)
parser.add_argument('--grad_clipping', default=5, type=float)
# parser.add_argument('--isSaveModel', action='store_false')
# parser.add_argument('--debug_DATA_dir', default='', type=str)
parser.add_argument('--numLayer_Res', default=0, type=int)
# parser.add_argument('--temperature', default=1, type=float)
# parser.add_argument('--num_MeanInference', default=1, type=int)
# parser.add_argument('--distill_type', default='AFS', type=str)   ## MaxMinS, MaxS, AFS
params = parser.parse_args()

classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
attention = torch.nn.AdaptiveMaxPool1d(1)
dimReduction = DimReduction(1024, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)

pretrained_weights=torch.load('MaxPool_model_best.pth')
classifier.load_state_dict(pretrained_weights['classifier'])
dimReduction.load_state_dict(pretrained_weights['dim_reduction'])
attention.load_state_dict(pretrained_weights['attention'])

trainset=EmbededFeatsDataset('/your/path/to/CAMELYON16/',mode='train',level=1)
valset=EmbededFeatsDataset('/your/path/to/CAMELYON16/',mode='val',level=1)
testset=EmbededFeatsDataset('/your/path/to/CAMELYON16/',mode='test',level=1)

trainloader=torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, drop_last=False)
valloader=torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True, drop_last=False)
testloader=torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, drop_last=False)

classifier.train()
dimReduction.train()
attention.train()

trainable_parameters = []
trainable_parameters += list(classifier.parameters())
trainable_parameters += list(attention.parameters())
trainable_parameters += list(dimReduction.parameters())

optimizer0 = torch.optim.Adam(trainable_parameters, lr=params.lr,  weight_decay=params.weight_decay)

best_auc = 0
best_epoch = -1
test_auc = 0

ce_cri = torch.nn.CrossEntropyLoss(reduction='none').to(params.device)

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def TestModel(test_loader):
    classifier.eval()
    dimReduction.eval()
    attention.eval()

    y_score=[]
    y_true=[]
    for i, data in enumerate(test_loader):
        inputs, labels=data

        labels=labels.data.numpy().tolist()
        inputs_tensor=inputs.to(params.device)

        with torch.no_grad():
            tmidFeat = dimReduction(inputs_tensor).squeeze(0)
            tAA = attention(tmidFeat.t()).squeeze(0).t()
        
        with torch.no_grad():
            tPredict = classifier(tAA)
        gSlidePred = torch.softmax(tPredict, dim=1)
        
        pred=(gSlidePred.cpu().data.numpy()).tolist()
        y_score.extend(pred)
        y_true.extend(labels)

    acc = np.sum(y_true==np.argmax(y_score,axis=1))/len(y_true)
    auc = roc_auc_score(y_true,[x[-1] for x in y_score])
    f1 = f1_score(y_true,np.argmax(y_score,axis=1))
    fpr, tpr, threshold = roc_curve(y_true, [x[1] for x in y_score], pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    opf1=f1_score(y_true,[x[1] for x in y_score]>=threshold_optimal)
    opacc=np.sum(y_true==([x[1] for x in y_score]>=threshold_optimal))/len(y_true)
    print('result: auc:{},acc:{},f1:{},opacc:{},opf1:{}, opthres:{}'.format(auc,acc,f1,opacc,opf1,threshold_optimal))
    return auc,acc,f1

best_auc=0.7
TestModel(testloader)
# raise Exception
for ii in range(params.EPOCH):

    for param_group in optimizer0.param_groups:
        curLR = param_group['lr']
        print('current learning rate {}'.format(curLR))

    classifier.train()
    dimReduction.train()
    attention.train()

    for i, data in enumerate(trainloader):
        inputs, labels=data
        labels=labels.to(params.device)

        inputs_tensor=inputs.to(params.device)

        tmidFeat = dimReduction(inputs_tensor).squeeze(0)
        tAA = attention(tmidFeat.t()).squeeze(0).t()
        
        tPredict = classifier(tAA)
        loss0 = ce_cri(tPredict, labels).mean()
        optimizer0.zero_grad()
        loss0.backward()
        torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), params.grad_clipping)
        torch.nn.utils.clip_grad_norm_(attention.parameters(), params.grad_clipping)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), params.grad_clipping)
        optimizer0.step()

        if i%10==0:
            print('[EPOCH{}:ITER{}] loss0:{};'.format(ii,i,loss0.item()))

    auc,acc,f1=TestModel(valloader)
    if auc>best_auc:
        best_auc=auc
        print('new best auc. Testing...')
        TestModel(testloader)
        tsave_dict = {
            'classifier': classifier.state_dict(),
            'dim_reduction': dimReduction.state_dict(),
            'attention': attention.state_dict(),
            # 'att_classifier': attCls.state_dict()
        }
        torch.save(tsave_dict, 'MaxPool_EM_model_best.pth')
