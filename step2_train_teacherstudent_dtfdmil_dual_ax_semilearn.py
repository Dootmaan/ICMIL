from model.network import Classifier_1fc, DimReduction
from model.Attention import Attention_Gated as Attention
from model.Attention import Attention_with_Classifier
from model.feature_extraction import resnet50_baseline
import argparse
import torch
import torchvision.models as models
# from dataset.EmbededFeatsDataset import EmbededFeatsDataset
from dataset.RandomPatchDistillationDataset import RandomPatchDistillationDataset
# torch.autograd.set_detect_anomaly(True)
from sklearn.metrics import roc_auc_score
import numpy as np
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='abc')

parser.add_argument('--name', default='abc', type=str)
parser.add_argument('--EPOCH', default=200, type=int)
parser.add_argument('--epoch_step', default='[100]', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--isPar', default=False, type=bool)
parser.add_argument('--log_dir', default='./debug_log', type=str)   ## log file path
parser.add_argument('--train_show_freq', default=40, type=int)
parser.add_argument('--droprate', default='0', type=float)
parser.add_argument('--droprate_2', default='0', type=float)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--batch_size', default=60, type=int)
parser.add_argument('--batch_size_v', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_cls', default=2, type=int)
parser.add_argument('--mDATA0_dir_train0', default='', type=str)  ## Train Set
parser.add_argument('--mDATA0_dir_val0', default='', type=str)      ## Validation Set
parser.add_argument('--mDATA_dir_test0', default='', type=str)         ## Test Set
parser.add_argument('--numGroup', default=5, type=int)
parser.add_argument('--total_instance', default=4, type=int)
parser.add_argument('--numGroup_test', default=4, type=int)
parser.add_argument('--total_instance_test', default=4, type=int)
parser.add_argument('--mDim', default=512, type=int)
parser.add_argument('--grad_clipping', default=5, type=float)
parser.add_argument('--isSaveModel', action='store_false')
parser.add_argument('--debug_DATA_dir', default='', type=str)
parser.add_argument('--numLayer_Res', default=0, type=int)
parser.add_argument('--temperature', default=1, type=float)
parser.add_argument('--num_MeanInference', default=1, type=int)
parser.add_argument('--distill_type', default='AFS', type=str)   ## MaxMinS, MaxS, AFS
params = parser.parse_args()

BETA=2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_teacher=resnet50_baseline(True).to(device)
# model_teacher=models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V2, norm_layer=torch.nn.BatchNorm2d).to(device)
for param in model_teacher.parameters():
    param.requires_grad = False
# model_teacher.fc = torch.nn.Identity()
# model_teacher.load_state_dict(torch.load('ResNet50_best.pth'))
model_teacher.eval()

model_student=resnet50_baseline(True).to(device)
# model_student=models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V2, norm_layer=torch.nn.BatchNorm2d).to(device) # THIS IS WHAT WE NEED!!!
# for param in model_student.parameters():
#     param.requires_grad = False
# model_student.fc = torch.nn.Identity()
# model.eval() # this will output the same pred as pseudo labels 
# model.load_state_dict(torch.load(''))

classifier_teacher = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
attention1_teacher = Attention(params.mDim).to(params.device)
classifier1_teacher = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
dimReduction_teacher = DimReduction(1024, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)
attCls = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2).to(params.device)
checkpoint=torch.load('/your/path/to/model/model_best.pth',map_location='cpu')
classifier_teacher.load_state_dict(
    {"fc.weight":checkpoint['att_classifier']['classifier.fc.weight'],"fc.bias":checkpoint['att_classifier']['classifier.fc.bias']})
attCls.load_state_dict(checkpoint['att_classifier'])
attention_teacher=attCls.attention
dimReduction_teacher.load_state_dict(checkpoint['dim_reduction'])
attention1_teacher.load_state_dict(checkpoint['attention'])
classifier1_teacher.load_state_dict(checkpoint['classifier'])
for param in classifier_teacher.parameters():
    param.requires_grad = False
for param in dimReduction_teacher.parameters():
    param.requires_grad = False

classifier_student = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
# attention = Attention(params.mDim).to(params.device)
dimReduction_student = DimReduction(1024, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)
# attCls = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2).to(params.device)
classifier_student.load_state_dict(
    {"fc.weight":checkpoint['att_classifier']['classifier.fc.weight'],"fc.bias":checkpoint['att_classifier']['classifier.fc.bias']})
# attention.load_state_dict(checkpoint['attentiontestloader'])
dimReduction_student.load_state_dict(checkpoint['dim_reduction'])

# classifier_teacher.eval()
# dimReduction_teacher.eval()

trainset=RandomPatchDistillationDataset('/your/path/to/CAMELYON16/',mode='train',level=1)
valset=RandomPatchDistillationDataset('/your/path/to/CAMELYON16/',mode='val',level=1)
# testset=RandomPatchDistillationDataset('/your/path/to/CAMELYON16/',mode='test',level=1)

trainloader=torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, drop_last=False)
valloader=torch.utils.data.DataLoader(valset, batch_size=100, shuffle=True, drop_last=False)
# testloader=torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, drop_last=False)

# attention.eval()
# attCls.train()

# trainable_parameters = []
# trainable_parameters += list(classifier.parameters())
# trainable_parameters += list(attention.parameters())
# trainable_parameters += list(dimReduction.parameters())

optimizer2 = torch.optim.Adam(list(model_student.parameters())+list(dimReduction_student.parameters())+list(classifier_student.parameters()), lr=params.lr,  weight_decay=params.weight_decay)
# optimizer1 = torch.optim.Adam(list(dimReduction_teacher.parameters())+list(classifier_teacher.parameters()), lr=params.lr,  weight_decay=params.weight_decay)

scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, [100], gamma=params.lr_decay_ratio)
# scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [100], gamma=params.lr_decay_ratio)

best_epoch = -1
test_auc = 0

ce_cri = torch.nn.KLDivLoss(reduction='none').to(params.device)

def min_max_norm(tAA):
    return (tAA-torch.min(tAA))/(torch.max(tAA)-torch.min(tAA))

def TestModel(test_loader):
    # classifier.eval()
    # dimReduction.eval()
    # attention.eval()
    # attCls.eval()

    # gPred_1 = torch.FloatTensor().to(params.device)
    # gt_1 = torch.LongTensor().to(params.device)
    model_student.eval()
    dimReduction_student.eval()
    classifier_student.eval()

    y_score=[]
    y_true=[]
    for i, data in enumerate(test_loader):
        inputs, inputs1, inputs2=data

        labels=classifier_teacher(dimReduction_teacher(model_teacher(inputs.type(torch.FloatTensor).to(params.device))))
        labels=torch.softmax(labels,-1).cpu().data.numpy().tolist()
        # labels=labels.to(params.device)
        
        with torch.no_grad():
            inputs_tensor=model_student(inputs1.type(torch.FloatTensor).to(params.device))
            tmidFeat = dimReduction_student(inputs_tensor).squeeze(0)
            tPredict = classifier_student(tmidFeat)
            gSlidePred = torch.softmax(tPredict, dim=1)
        
        pred=(gSlidePred.cpu().data.numpy()).tolist()
        y_score.extend(pred)
        y_true.extend(labels)

    acc = np.sum(np.argmax(y_true,axis=1)==np.argmax(y_score,axis=1))/len(y_true)
    # auc = roc_auc_score(y_true,[x[-1] for x in y_score])
    mae = np.mean(np.abs(np.array(y_score)-np.array(y_true)))
    print('test result: mae:{},acc:{}'.format(mae,acc))
    return mae,acc

best_mae=1
# TestModel(testloader)
for ii in range(params.EPOCH):
    model_student.train()
    for param_group in optimizer2.param_groups:
        curLR = param_group['lr']
        print('current learning rate {}'.format(curLR))

    # classifier.eval()
    # dimReduction.train()
    # attention.train()
    # attCls.train()

    # instance_per_group = total_instance // numGroup

    # numSlides = len(SlideNames_list)
    # numIter = numSlides // params.batch_size

    # tIDX = list(range(numSlides))
    # random.shuffle(tIDX)

    for i, data in enumerate(trainloader):
        inputs, inputs1, inputs2=data
        # labels=classifier_teacher(dimReduction_teacher(model_teacher(inputs.type(torch.FloatTensor).to(params.device))))

        inputs_tensor_teacher=model_teacher(inputs.type(torch.FloatTensor).to(params.device))

        tmidFeat_teacher = dimReduction_teacher(inputs_tensor_teacher)
        attention_score1=attention1_teacher(tmidFeat_teacher).squeeze(0)
        attention_score1=min_max_norm(attention_score1)
        attention_score=attention_teacher(tmidFeat_teacher).squeeze(0)
        attention_score=min_max_norm(attention_score)
        attention_score=attention_score*attention_score1
        attention_score=abs((2*attention_score-1)**BETA)
        # attention_score1=abs(attention_score1**BETA)

        tPredict_teacher = 0.7*classifier_teacher(tmidFeat_teacher)+0.3*classifier1_teacher(tmidFeat_teacher)

        inputs_tensor_student=model_student(inputs1.type(torch.FloatTensor).to(params.device))

        tmidFeat_student = dimReduction_student(inputs_tensor_student)
        tPredict_student = classifier_student(tmidFeat_student)

        consistency_tmidFeat=dimReduction_student(inputs_tensor_teacher)
        consistency_tPredict=classifier_student(consistency_tmidFeat)

        loss_c = ce_cri(F.log_softmax(tPredict_student,dim=-1), F.softmax(tPredict_teacher.detach(),dim=-1))
        loss_c = torch.einsum('ns,n->ns', loss_c, attention_score)
        loss_c = loss_c.mean()
        loss_w1 = ce_cri(F.log_softmax(consistency_tPredict,dim=-1), F.softmax(tPredict_teacher,dim=-1)).mean()
        loss_w2 = ce_cri(F.log_softmax(consistency_tmidFeat,dim=-1), F.softmax(tmidFeat_teacher,dim=-1)).mean()
        loss_2=loss_c+0.5*loss_w1+0.5*loss_w2

        optimizer2.zero_grad()
        # optimizer1.zero_grad()
        # loss1.backward()
        # optimizer1.step()

        loss_2.backward()
        torch.nn.utils.clip_grad_norm_(dimReduction_student.parameters(), params.grad_clipping)
        # torch.nn.utils.clip_grad_norm_(attention_student.parameters(), params.grad_clipping)
        torch.nn.utils.clip_grad_norm_(classifier_student.parameters(), params.grad_clipping)
        optimizer2.step()

        ## optimization for the second tier
        # gSlidePred = attCls(slide_pseudo_feat)
        # loss1 = ce_cri(gSlidePred, labels).mean()
        # optimizer1.zero_grad()
        # loss1.backward()
        # torch.nn.utils.clip_grad_norm_(attCls.parameters(), params.grad_clipping)
        # optimizer1.step()

        if i%400==0:
            print('[EPOCH{}:ITER{}] loss_c:{}; '.format(ii,i,loss_c.item()))
        # if i>100 and i%5000==0:
        #     # print('Testing:')
        #     mae,acc=TestModel(valloader)
        #     if mae<best_mae:
        #         best_mae=mae
        #     torch.save(model_student.state_dict(), 'ResNet50_DTFD_axgated_teacher_student_best1024_BETA2.pth')
        #     print('new best auc, weights saved. ')
    
    scheduler2.step()
    # scheduler1.step()
    print('End of epoch',ii)
    mae,acc=TestModel(valloader)
    if mae<best_mae:
        best_mae=mae
        torch.save(model_student.state_dict(), 'ResNet50_DTFD_axgated_teacher_student_best1024_BETA2.pth')
        print('new best auc, weights saved. ')
        # TestModel(testloader)
        # tsave_dict = {
        #     'classifier': classifier.state_dict(),
        #     'dim_reduction': dimReduction.state_dict(),
        #     # 'attention': attention.state_dict(),
        #     # 'att_classifier': attCls.state_dict()
        # }
        