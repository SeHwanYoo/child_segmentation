import torch
from torch.utils.data import Dataset
import os 
from glob import glob
from PIL import Image
import numpy as np

class SegDataset(Dataset):
    def __init__(self, path, ints, grds, transform=None, is_test=False):
        super().__init__()
        
        self.transform = transform
        
        self.img_list = []
        self.mask_list = []

        if is_test:
            self.img_list = glob(os.path.join(path, ints, 'Atopy_Segment_Test', f'{grds}/*.jpg'))
            self.mask_list = glob(os.path.join(path, ints, 'Atopy_Segment_Test', f'{grds}/*.png'))
            
        else:
            self.img_list = glob(os.path.join(path, ints, 'Atopy_Segment_Train', f'{grds}/*.jpg'))
            self.mask_list = glob(os.path.join(path, ints, 'Atopy_Segment_Train', f'{grds}/*.png'))
            
            # x_extra_list = glob(os.path.join(path, inters, 'Atopy_Segment_Extra', f'{grade}/*.jpg'))
            # y_extra_list = glob(os.path.join(path, inters, 'Atopy_Segment_Extra', f'{grade}/*.png'))
        

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):      
        img = Image.open(self.img_list[index]).convert('RGB')
        mask = Image.open(self.mask_list[index]).convert('L')
        
        img = np.array(img)
        mask = np.array(mask)
        # mask[mask==255.0] = 1.0
        mask[mask > 0] = 1.0 
        
        # if self.transform is not None:
        #     img, mask = self.transform(image=img, mask=mask)
            
        # print('-' * 20)
        # print(img)
        # print(mask)

        img = torch.tensor(img).permute(2, 0, 1).float()
        mask = torch.tensor(mask).permute(2, 0, 1).float()
        # mask = torch.tensor(mask).unsqueeze(0).float()
        
        return img, mask
       