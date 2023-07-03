from model.feature_extraction import resnet50_baseline
import glob
import torch
import numpy as np
import torchvision.transforms as T
import PIL

BATCH_SIZE=448
transform=T.Compose([
    T.ToTensor(),
])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=resnet50_baseline(False).to(device)
model.load_state_dict(torch.load('/your/path/to/ResNet50/weights.pth')) # you can also use the default ImageNet pretrained weights by commenting this line

# model=torch.nn.DataParallel(model)
model.eval()
filenames=sorted(glob.glob('/your/path/to/the/extracted/patches/folder/*'))
for cases in filenames:
    print('processing:',cases)
    feats=[]
    imgs=sorted(glob.glob(cases+'/*.png'))
    for i in range(0,len(imgs),BATCH_SIZE):
        imgnames=imgs[i:i+BATCH_SIZE]
        input_tensor=torch.FloatTensor().to(device)
        for imgname in imgnames:
            img=PIL.Image.open(imgname)
            input_tensor=torch.cat([input_tensor,transform(img).to(device).unsqueeze(0)],dim=0)
        with torch.no_grad():
            feat=model(input_tensor)
        feats.extend(feat.cpu().data.numpy())
    feats=np.array(feats)
    np.save(cases+'/resnet1024_feats.npy',feats)
