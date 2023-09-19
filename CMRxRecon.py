"""
To generate img data from the raw mat file
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision import utils
SCALE = 100000
import random

class CMRxReconDataset(Dataset):
    
    def __init__(self, file_path, transform=None, length=-1, limit_val=False):
        """
        root_dir: absolute path of "ChallengData"
        file_path: the train_pair_file.txt
        """
        self.name_dict = {"MultiCoil":{"AccFactor04":"kspace_sub04",
                                       "AccFactor08":"kspace_sub08",
                                       "AccFactor10":"kspace_sub10",
                                       "FullSample":"kspace_full"}, 
                          "SingleCoil":{"AccFactor04":"kspace_single_sub04",
                                       "AccFactor08":"kspace_single_sub08",
                                       "AccFactor10":"kspace_single_sub10",
                                       "FullSample":"kspace_single_full"}}
        self.file_path = file_path
        file_obj = open(self.file_path, "r")
        self.train_pairs = file_obj.readlines()
        if length>0 and not limit_val:
            self.train_pairs = self.train_pairs[:length]
        if limit_val:
            random.Random(666).shuffle(self.train_pairs)
            self.train_pairs = self.train_pairs[:240]
        self.transform = transform
        file_obj.close()
        

    def __len__(self):
        return len(self.train_pairs)
    
    def __getitem__(self, index):
        path, GT_path = self.train_pairs[index].replace("\n","").split(" ")
        item = np.float32(np.load(path))
        GT_item = np.float32(np.load(GT_path))
        output = {"input": item, "GT": GT_item}
        if self.transform:
            data = np.stack((item, GT_item), axis=-1)
            transformed_data = self.transform(data)
            output = {"input": transformed_data[0,:,:].unsqueeze(0), "GT": transformed_data[1,:,:].unsqueeze(0), "ipath":path, "gtpath":GT_path}
        return output


if __name__=="__main__":

    tsfm = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((256, 512), antialias=True),
        # transforms.RandomCrop((224,448), pad_if_needed=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    training_set = CMRxReconDataset("/home/txiang/CMRxRecon/CMRxRecon_Repo/dataset/train_pair_file/Task2_acc_10_train_pair_file_npy_clean.txt",
                                    transform=tsfm)
    print(len(training_set))
    # for i in range(len(training_set)):
    #     print(i)
    #     item = training_set[i]
    print(len(training_set))
    pair0 = training_set[0]
    utils.save_image(pair0["input"]/255, "input.png")
    utils.save_image(pair0["GT"]/255, "GT.png")