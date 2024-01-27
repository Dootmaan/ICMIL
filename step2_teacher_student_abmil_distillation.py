from model.network import Classifier_1fc, DimReduction
from model.feature_extraction import resnet50_baseline
import argparse
import torch
from dataset.RandomPatchDistillationDataset import RandomPatchDistillationDataset
import numpy as np
import torch.nn.functional as F

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
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
# parser.add_argument('--batch_size', default=60, type=int)
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_teacher=resnet50_baseline(True).to(device)
model_teacher.load_state_dict(torch.load('ResNet50_ABMIL_teacher_student_best1024.pth','cpu'))
for param in model_teacher.parameters():
    param.requires_grad = False
model_teacher.eval()

model_student=resnet50_baseline(True).to(device)
model_student.load_state_dict(torch.load('ResNet50_ABMIL_teacher_student_best1024.pth','cpu'))

classifier_teacher = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
dimReduction_teacher = DimReduction(1024, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)
checkpoint=torch.load('/path/to/AB-MIL_model_best_em0_resnet1024.pth',map_location='cpu')
classifier_teacher.load_state_dict(checkpoint['classifier'])
dimReduction_teacher.load_state_dict(checkpoint['dim_reduction'])
for param in classifier_teacher.parameters():
    param.requires_grad = False
for param in dimReduction_teacher.parameters():
    param.requires_grad = False

classifier_student = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
dimReduction_student = DimReduction(1024, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)
classifier_student.load_state_dict(checkpoint['classifier'])
dimReduction_student.load_state_dict(checkpoint['dim_reduction'])

trainset=RandomPatchDistillationDataset('/your/path/to/CAMELYON16/',mode='train',level=1)
valset=RandomPatchDistillationDataset('/your/path/to/CAMELYON16/',mode='val',level=1)
# testset=RandomPatchDistillationDataset('/your/path/to/CAMELYON16/',mode='test',level=1) # test set should be invisible to this step. Our results are reported without using test set patches for finetuning.

trainloader=torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, drop_last=False)
valloader=torch.utils.data.DataLoader(valset, batch_size=100, shuffle=True, drop_last=False)
# testloader=torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, drop_last=False)

optimizer2 = torch.optim.Adam(list(model_student.parameters())+list(dimReduction_student.parameters())+list(classifier_student.parameters()), lr=params.lr,  weight_decay=params.weight_decay)

best_epoch = -1
test_auc = 0

ce_cri = torch.nn.KLDivLoss(reduction='none').to(params.device)

def TestModel(test_loader):
    
    model_student.eval()
    dimReduction_student.eval()
    classifier_student.eval()

    y_score=[]
    y_true=[]
    for i, data in enumerate(test_loader):
        inputs, inputs1, inputs2=data

        labels=classifier_teacher(dimReduction_teacher(model_teacher(inputs.type(torch.FloatTensor).to(params.device))))
        labels=torch.softmax(labels,-1).cpu().data.numpy().tolist()
        
        with torch.no_grad():
            inputs_tensor=model_student(inputs1.type(torch.FloatTensor).to(params.device))
            tmidFeat = dimReduction_student(inputs_tensor).squeeze(0)
            tPredict = classifier_student(tmidFeat)
            gSlidePred = torch.softmax(tPredict, dim=1)
        
        pred=(gSlidePred.cpu().data.numpy()).tolist()
        y_score.extend(pred)
        y_true.extend(labels)

    acc = np.sum(np.argmax(y_true,axis=1)==np.argmax(y_score,axis=1))/len(y_true)
    mae = np.mean(np.abs(np.array(y_score)-np.array(y_true)))
    print('test result: mae:{},acc:{}'.format(mae,acc))
    return mae,acc

best_mae=1
for ii in range(params.EPOCH):
    model_student.train()
    for param_group in optimizer2.param_groups:
        curLR = param_group['lr']
        print('current learning rate {}'.format(curLR))

    for i, data in enumerate(trainloader):
        inputs, inputs1, inputs2=data
        
        inputs_tensor_teacher=model_teacher(inputs.type(torch.FloatTensor).to(params.device))

        tmidFeat_teacher = dimReduction_teacher(inputs_tensor_teacher)
        tPredict_teacher = classifier_teacher(tmidFeat_teacher)

        inputs_tensor_student=model_student(inputs1.type(torch.FloatTensor).to(params.device))

        tmidFeat_student = dimReduction_student(inputs_tensor_student)
        tPredict_student = classifier_student(tmidFeat_student)

        consistency_tmidFeat=dimReduction_student(inputs_tensor_teacher)
        consistency_tPredict=classifier_student(consistency_tmidFeat)

        loss_c = ce_cri(F.log_softmax(tPredict_student,dim=-1), F.softmax(tPredict_teacher.detach(),dim=-1)).mean()
        loss_w1 = ce_cri(F.log_softmax(consistency_tPredict,dim=-1), F.softmax(tPredict_teacher,dim=-1)).mean()
        loss_w2 = ce_cri(F.log_softmax(consistency_tmidFeat,dim=-1), F.softmax(tmidFeat_teacher,dim=-1)).mean()
        loss=loss_c+0.5*loss_w1+0.5*loss_w2

        optimizer2.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(dimReduction_student.parameters(), params.grad_clipping)
        torch.nn.utils.clip_grad_norm_(classifier_student.parameters(), params.grad_clipping)
        optimizer2.step()

        if i%400==0:
            print('[EPOCH{}:ITER{}] loss_c:{}; '.format(ii,i,loss_c.item()))
        if i>100 and i%5000==0:
            # print('Testing:')
            # mae,acc=TestModel(valloader)
            # if mae<best_mae:
                # best_mae=mae
            torch.save(model_student.state_dict(), str(i)+'iter_ResNet50_teacher_student_em1_best1024.pth')
            print('new best auc, weights saved. ')
    
    print('End of epoch',ii)
    mae,acc=TestModel(valloader)
    if mae<best_mae:
        best_mae=mae
        torch.save(model_student.state_dict(), 'ResNet50_teacher_student_em1_best1024.pth')
        print('new best auc, weights saved. ')
        