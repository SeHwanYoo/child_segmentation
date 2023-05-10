import torch
from torch.utils.data import Dataset
import os 
from glob import glob
from PIL import Image

class SegDataset(Dataset):
    def __init__(self, path, inters, grade, transform=None, is_test=False):
        super().__init__()
        
        self.transform = transform
        
        if is_test:
            x_test_list = glob(os.path.join(path, inters, 'Atopy_Segment_Test', f'{grade}/*.jpg'))
            y_test_list = glob(os.path.join(path, inters, 'Atopy_Segment_Test', f'{grade}/*.png'))
            
        else:
            x_train_list = glob(os.path.join(path, inters, 'Atopy_Segment_Train', f'{grade}/*.jpg'))
            y_train_list = glob(os.path.join(path, inters, 'Atopy_Segment_Train', f'{grade}/*.png'))
            
            # x_extra_list = glob(os.path.join(path, inters, 'Atopy_Segment_Extra', f'{grade}/*.jpg'))
            # y_extra_list = glob(os.path.join(path, inters, 'Atopy_Segment_Extra', f'{grade}/*.png'))
        

    def __len__(self):
        return len(self.ppg_signals)

    def __getitem__(self, index):      
        img = Image.open(self.x_train_list[index]).convert('RGB')
        mask = Image.open(self.y_train_list[index]).convert('L')
        
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        img = torch.tensor(img).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()
        
        return img, mask
       