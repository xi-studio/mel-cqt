import glob
import torch
import numpy as np
import torch.utils.data as data
from os import listdir
from os.path import join
import os
import random
import librosa
import re
import h5py


def default_loader(path):
    f = h5py.File(path,'r')
    mel = f['mel'][:]
    cqt = f['cqt'][:]
    f.close()
    return mel, cqt


class Audio(data.Dataset):
    def __init__(self, name='train'):
        super(Audio, self).__init__()
        self.image_list = glob.glob('./data/lrm_data/*.h5')

    def __getitem__(self, index):
        path = self.image_list[index]
        mel, cqt = default_loader(path) 
        mel = mel.astype(np.float32)
        cqt = cqt.astype(np.float32)
        
        return mel, cqt

    def __len__(self):

        return len(self.image_list)


