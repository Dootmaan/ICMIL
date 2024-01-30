from model.network import Classifier_1fc, DimReduction
from model.Attention import Attention_Gated as Attention
# from model.Attention import Attention_with_Classifier
import argparse
import torch
from dataset.EmbededFeatsDataset import EmbededFeatsDataset
# torch.autograd.set_detect_anomaly(True)
from sklearn.metrics import roc_auc_score, f1_score,roc_curve
import numpy as np
from dataset.RandMixup import randmixup

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
parser.add_argument('--lr', default=2e-4, type=float)
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
attention = Attention(params.mDim).to(params.device)
dimReduction = DimReduction(1024, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)

trainset=EmbededFeatsDataset('/path/to/CAMELYON16/',mode='train',level=1)
valset=EmbededFeatsDataset('/path/to/CAMELYON16/',mode='val',level=1)
testset=EmbededFeatsDataset('/path/to/CAMELYON16/',mode='test',level=1)

def collate_features(batch):
    img = [torch.from_numpy(item[0]).to(params.device) for item in batch]
    coords = [torch.tensor(item[1]).to(params.device) for item in batch]
    return [img, coords]

trainloader=torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, drop_last=False, collate_fn=collate_features)
valloader=torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True, drop_last=False)
testloader=torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, drop_last=False)

classifier.train()
dimReduction.train()
attention.train()

trainable_parameters = []
trainable_parameters += list(classifier.parameters())
trainable_parameters += list(attention.parameters())
trainable_parameters += list(dimReduction.parameters())

optimizer1 = torch.optim.Adam(trainable_parameters, lr=params.lr,  weight_decay=params.weight_decay)

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
            tAA = attention(tmidFeat).squeeze(0)
        tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
        
        with torch.no_grad():
            tPredict = classifier(tattFeat_tensor)
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

best_auc=0.1
best_avg=0
for ii in range(params.EPOCH):

    for param_group in optimizer1.param_groups:
        curLR = param_group['lr']
        print('current learning rate {}'.format(curLR))

    classifier.train()
    dimReduction.train()
    attention.train()
    
    for i, data in enumerate(trainloader):
        inputs, labels=data
        mix_inputs, labels_a, labels_b, lmbdas = randmixup(inputs,labels)
        for j in range(len(mix_inputs)):
            inputs_tensor=mix_inputs[j]
            label_a=labels_a[j].unsqueeze(0)
            label_b=labels_b[j].unsqueeze(0)
            lam=lmbdas[j]

            label_a=label_a.to(params.device)
            label_b=label_b.to(params.device)

            inputs_tensor=inputs_tensor.to(params.device)

            tmidFeat = dimReduction(inputs_tensor).squeeze(0)
            tAA = attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            
            tPredict = classifier(tattFeat_tensor)
        
            loss_1 = lam*ce_cri(tPredict, label_a)+(1-lam)*ce_cri(tPredict,label_b)
            optimizer1.zero_grad()
            loss_1.backward()
            torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(attention.parameters(), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), params.grad_clipping)
            optimizer1.step()

        if i%10==0:
            print('[EPOCH{}:ITER{}] loss_1:{};'.format(ii,i,loss_1.item()))
    
    print('testing...')
    auc,acc,f1=TestModel(valloader)
    if auc>best_auc:
        best_auc=auc
        tsave_dict = {
            'classifier': classifier.state_dict(),
            'dim_reduction': dimReduction.state_dict(),
            'attention': attention.state_dict(),
        }
        torch.save(tsave_dict, 'AB-MIL_model_best_resnet1024.pth')
        print('new best auc. saved.')
        TestModel(testloader)
