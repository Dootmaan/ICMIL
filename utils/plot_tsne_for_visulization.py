import numpy as np
import matplotlib.pyplot as plt
# from sklearn import datasets
import torch
from model.network import Classifier_1fc, DimReduction
from model.Attention import Attention_Gated as Attention
from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
from dataset.EmbededFeatsDataset import EmbededFeatsDataset

def plot_tsne(features, features_maxpool, labels):

    import pandas as pd
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    # tsne=PCA(2)

    class_num = len(np.unique(labels))
    latent = features
    tsne_features = tsne.fit_transform(features) 
    tsne_features1 = tsne.fit_transform(features_maxpool) 
    
    plt.figure()
    plt.title("PCA on CAMELYON16 Instances")

    plt.scatter(tsne_features1[:,0]/12,tsne_features1[:,1]/18,25,1.0*labels,cmap='rainbow',alpha=0.9)
    plt.scatter(tsne_features[:,0],tsne_features[:,1],50,1.0*labels,marker='^',cmap='jet',alpha=0.5)
    plt.savefig('pca_original.pdf',dpi=250)

if __name__ == '__main__':
    trainset=EmbededFeatsDataset('/your/path/to/CAMELYON16/',mode='train',level=1)
    testset=EmbededFeatsDataset('/your/path/to/CAMELYON16/',mode='test',level=1)
    trainloader=torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, drop_last=False)
    testloader=torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, drop_last=False)

    attention = Attention(512).to('cuda')
    dimReduction = DimReduction(1024, 512, numLayer_Res=0).to('cuda')
    attention1=torch.nn.AdaptiveMaxPool1d(1)
    pretrained_weights=torch.load('/your/path/to/AB-MIL_model_best_em0_resnet1024.pth')
    # classifier.load_state_dict(pretrained_weights['classifier'])
    dimReduction.load_state_dict(pretrained_weights['dim_reduction'])
    attention.load_state_dict(pretrained_weights['attention'])
    # attCls.load_state_dict(pretrained_weights['att_classifier'])
    dimReduction.eval()
    attention.eval()
    final_feats=[]
    final_feats_maxpool=[]
    final_labels=[]
    for i, data in enumerate(trainloader):
        inputs, labels=data

        labels=labels.data.numpy().tolist()
        # labels=labels.to('cuda')
        inputs_tensor=inputs.to('cuda')

        with torch.no_grad():
            tmidFeat = dimReduction(inputs_tensor).squeeze(0)
            tAA1 = attention1(tmidFeat.t()).squeeze(0).t()
            tAA = attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)

        final_feats_maxpool.extend(tAA1.cpu().data.numpy().tolist())
        final_feats.extend(tattFeat_tensor.cpu().data.numpy().tolist())
        final_labels.extend(labels)
    
    for i, data in enumerate(testloader):
        inputs, labels=data

        labels=labels.data.numpy().tolist()
        # labels=labels.to('cuda')
        inputs_tensor=inputs.to('cuda')

        with torch.no_grad():
            tmidFeat = dimReduction(inputs_tensor).squeeze(0)
            tAA1 = attention1(tmidFeat.t()).squeeze(0).t()
            tAA = attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)

        final_feats_maxpool.extend(tAA1.cpu().data.numpy().tolist())
        final_feats.extend(tattFeat_tensor.cpu().data.numpy().tolist())
        final_labels.extend(labels)
    
    final_feats_maxpool=np.array(final_feats_maxpool)
    final_feats=np.array(final_feats)
    final_labels=np.array(final_labels)

    plot_tsne(final_feats,final_feats_maxpool, final_labels)
