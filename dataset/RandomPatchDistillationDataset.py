import torch
import glob
import random
import PIL
import torchvision.transforms as T

class RandomPatchDistillationDataset(torch.utils.data.Dataset):
    def __init__(self, path, mode='train', level=1,
                 transform=T.Compose([
                     T.ToTensor(),
                 ]),
                 transform1=T.Compose([
                     T.RandomHorizontalFlip(),
                     T.RandomVerticalFlip(),
                     T.RandomResizedCrop((256,256),(0.5,1)),
                    #  T.RandomGrayscale(),
                     T.ToTensor(),
                 ]), transform2=T.Compose([
                     T.RandomHorizontalFlip(),
                     T.RandomVerticalFlip(),
                     T.RandomResizedCrop((256,256),(0.5,1)),
                    #  T.RandomGrayscale(),
                     T.ToTensor(),
                 ])):
        super().__init__()
        self.mode = mode
        self.data = []
        self.transform = transform
        self.transform1 = transform1
        self.transform2 = transform2
        if mode == 'train' or self.mode == 'val':
            filenames = sorted(
                glob.glob(path+'/extracted_patches/training/*/256.'+str(level)+'/*/*.png'))
            # make sure each time we have the same filenames order.
            random.seed(552)
            random.shuffle(filenames)
            random.seed()
            train_frac, val_frac = 0.9, 0.1
            n_train = int(train_frac*len(filenames))
            n_val = int(len(filenames)-n_train)

            if mode == 'train':
                print('processing training dataset. ')
                filenames = filenames[:n_train]
                for fname in filenames:
                    self.data.append(fname)

            if mode == 'val':
                print('processing val dataset. ')
                filenames = filenames[n_train:]
                for fname in filenames:
                    self.data.append(fname)

        if mode == 'test':
            print('processing testing dataset. ')
            filenames_all = sorted(
                glob.glob(path+'/extracted_patches/testing/*/256.'+str(level)+'/*/*.png'))
            filenames = []
            for potential_fname in filenames_all:
                if 'test_114' in potential_fname or 'test_124' in potential_fname:
                    continue
                else:
                    filenames.append(potential_fname)

            for fname in filenames:
                self.data.append(fname)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fname = self.data[index]
        img = PIL.Image.open(fname)
        npy = self.transform(img)
        npy1 = self.transform1(img)
        npy2 = self.transform2(img)
        # with torch.no_grad():
        #     lbl=classifier(dimReduction(npy.to(device).unsqueeze(0))).squeeze(0)

        return npy, npy1, npy2
