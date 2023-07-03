import torch
import glob
import random
import numpy as np
import pandas
import pickle
import os
testMask_dir = '/path/to/CAMELYON16/mask/' ## Point to the Camelyon test set mask location

class PickleDataset(torch.utils.data.Dataset):
    def __init__(self,mode='train'):
        super().__init__()
        self.mode=mode
        self.data=[]
        self.label=[]
        if mode=='train' or self.mode=='val':
            with open('/path/to/CAMELYON16/mDATA_train.pkl', 'rb') as f:
                mDATA_train_full = pickle.load(f)
                mDATA_train_full = self.random_dic(mDATA_train_full)
                mDATA_train=self.dict_slice(mDATA_train_full,0, int(0.9*len(mDATA_train_full)))
            # with open(params.mDATA0_dir_val0, 'rb') as f:
                mDATA_val = self.dict_slice(mDATA_train_full,int(0.9*len(mDATA_train_full)),len(mDATA_train_full)+1)
            
            if mode=='train':

                for slide_name in mDATA_train.keys():
                    # SlideNames.append(slide_name)

                    if slide_name.startswith('tumor'):
                        label = 1
                    elif slide_name.startswith('normal'):
                        label = 0
                    else:
                        raise RuntimeError('Undefined slide type')
                    self.label.append(label)

                    patch_data_list = mDATA_train[slide_name]
                    featGroup = []
                    for tpatch in patch_data_list:
                        tfeat = torch.from_numpy(tpatch['feature'])
                        featGroup.append(tfeat.unsqueeze(0))
                    featGroup = torch.cat(featGroup, dim=0) ## numPatch x fs
                    self.data.append(featGroup)

            if mode=='val':

                for slide_name in mDATA_val.keys():

                    if slide_name.startswith('tumor'):
                        label = 1
                    elif slide_name.startswith('normal'):
                        label = 0
                    else:
                        raise RuntimeError('Undefined slide type')
                    self.label.append(label)
                
                    patch_data_list = mDATA_val[slide_name]
                    featGroup = []
                    for tpatch in patch_data_list:
                        tfeat = torch.from_numpy(tpatch['feature'])
                        featGroup.append(tfeat.unsqueeze(0))
                    featGroup = torch.cat(featGroup, dim=0) ## numPatch x fs
                    self.data.append(featGroup)

        if mode=='test':
            with open('/path/to/CAMELYON16/mDATA_test.pkl', 'rb') as f:
                mDATA_test = pickle.load(f)

            tumorSlides = os.listdir(testMask_dir)
            tumorSlides = [sst.split('.')[0] for sst in tumorSlides]

            for slide_name in mDATA_test.keys():

                if slide_name in tumorSlides:
                    label = 1
                else:
                    label = 0
                self.label.append(label)
            
                patch_data_list = mDATA_test[slide_name]
                featGroup = []
                for tpatch in patch_data_list:
                    tfeat = torch.from_numpy(tpatch['feature'])
                    featGroup.append(tfeat.unsqueeze(0))
                featGroup = torch.cat(featGroup, dim=0) ## numPatch x fs
                self.data.append(featGroup)

    def random_dic(self,dicts,seed=552):
        random.seed(seed)
        dict_key_ls = list(dicts.keys())
        random.shuffle(dict_key_ls)
        new_dic = {}
        for key in dict_key_ls:
            new_dic[key] = dicts.get(key)
        random.seed()
        return new_dic

    def dict_slice(self,adict, start, end):
        keys = adict.keys()
        dict_slice = {}
        for k in list(keys)[start:end]:
            dict_slice[k] = adict[k]
        return dict_slice
    
    def __len__(self):
        return len(self.label)

    def augment(self,feats):
        np.random.shuffle(feats)
        return feats

    def __getitem__(self,index):
        # npy=np.load(self.data[index])
        return self.data[index],self.label[index]

if __name__=="__main__":
    dataset=PickleDataset(mode='val')
    dataset.__getitem__(40)
    dataset=PickleDataset(mode='test')