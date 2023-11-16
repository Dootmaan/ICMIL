import torch
import glob
from model.network import Classifier_1fc, DimReduction
from model.Attention import Attention_Gated as Attention
# from model.Attention import Attention_with_Classifier
import argparse
# from dataset.EmbededFeatsDataset import EmbededFeatsDataset
# torch.autograd.set_detect_anomaly(True)
from sklearn.metrics import roc_auc_score,f1_score,roc_curve
import numpy as np
from utils import eval_metric
from dataset.psemix_core import augment_bag
from PIL import Image
import cv2
import openslide
import cmap
import matplotlib.pyplot as plt
from matplotlib import cm

colormap = cm.get_cmap('jet')
# print(colormap(0.23))
# plt.scatter([x/256 for x in range(256)],[x/256 for x in range(256)],c=[x/256 for x in range(256)],cmap=colormap)
# plt.colorbar()
# plt.savefig('colorbar.png',dpi=600)

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
parser.add_argument('--batch_size', default=1, type=int)
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

wsi_path='/path/to/CAMELYON16/extracted_patches_0.8/testing/images/256.1/test_040'
mask='/path/to/CAMELYON16/mask/test_040.tif'
mask_img=openslide.open_slide(mask)
mask_img=np.array(mask_img.read_region((0,0),4,mask_img.level_dimensions[4]).convert('RGB'))

patches_embedded=np.load(wsi_path+'/axgated_em0_resnet1024_feats.npy')
# patches_embedded=np.load(wsi_path+'/resnet1024_feats.npy')
# patches_embedded=np.load(wsi_path+'/em0_resnet1024_feats.npy')
patches_path=sorted(glob.glob(wsi_path+'/*.png'))
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
attention = Attention(params.mDim).to(params.device)
dimReduction = DimReduction(1024, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)
# attCls = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2).to(params.device)

pretrained_weights=torch.load('AB-MIL_psemix_model_best.pth')
# pretrained_weights=torch.load('AB-MIL_model_best.pth')
# pretrained_weights=torch.load('AB-MIL_EM_model_best.pth')
# pretrained_weights=torch.load('/home/why/Workspace-Python/EM-MIL/AB-MIL_model_best_em0_resnet1024.pth')

classifier.load_state_dict(pretrained_weights['classifier'])
dimReduction.load_state_dict(pretrained_weights['dim_reduction'])
attention.load_state_dict(pretrained_weights['attention'])

classifier.eval()
dimReduction.eval()
attention.eval()

def min_max_norm(tAA):
    return (tAA-torch.min(tAA))/(torch.max(tAA)-torch.min(tAA))

patches_embedded=torch.from_numpy(patches_embedded).to(params.device).unsqueeze(0)
with torch.no_grad():
    tmidFeat = dimReduction(patches_embedded).squeeze(0)
    tAA = attention(tmidFeat,isNorm=False).squeeze(0)
tAA=min_max_norm(tAA)

last_patch=patches_path[-1]
max_col=int(last_patch.split(r'_')[-6])
max_row=int(last_patch.split(r'_')[-5])
shape_col=int(last_patch.split(r'_')[-3])
shape_row=int(last_patch.split(r'_')[-1].split(r'.')[0])

final_output=np.ones((shape_row*20,shape_col*20,3))*255
mask_img=cv2.resize(mask_img,(shape_col*20-60,shape_row*20-120))
mask_img=cv2.copyMakeBorder(mask_img, 60, 60, 30, 30, cv2.BORDER_CONSTANT, None, 0)
edges=cv2.Canny(mask_img, 127, 200)
kernel = np.ones((3,3), np.uint8)
edges =cv2.dilate(edges, kernel)
edges_red=np.zeros_like(final_output)
edges_red[edges!=0]=[0,0,255]
edges_white=np.zeros_like(final_output)
edges_white[edges!=0]=[255,255,255]

idx=0
for patch_path in patches_path:

    col=int(patch_path.split(r'_')[-6])+1
    row=int(patch_path.split(r'_')[-5])+7

    attention_score=tAA[idx].cpu().item()
    print(attention_score)
    # attention_score=0 if attention_score<5e-17 else 1000*attention_score
    idx+=1

    patch_img=cv2.imread(patch_path)
    patch_img=cv2.resize(patch_img,(20,20))
    mask_img= np.array(Image.new("RGB",(20,20),(int(255*colormap(attention_score)[2]),int(255*colormap(attention_score)[1]),int(255*colormap(attention_score)[0]))))
    # print(colormap(attention_score))
    output_img=0.6*mask_img+0.4*patch_img

    final_output[20*row-6:20*row+20-6, 20*col-7:20*col+20-7, :]=output_img

final_output[edges_white==255]=0
cv2.imwrite('040_gated_cmap_fullred.png',final_output+edges_red)

    # patch=PIL.Image.open(patch_path)
    # patch=torch.from_numpy(np.array(patch)).permute(2, 0, 1).float()/255.0
