import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import argparse
import datasets
import models
import os 
from sklearn.metrics import f1_score
import random

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path = '../../datasets/vitalDB'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', required=False, type=int, default=256)
    parser.add_argument('--epochs', required=False, type=int, default=100)
    args = parser.parse_args()
    
    return args

def train(t_model, t_train_loader, t_optimizer, loss_function, scheduler):    
    t_model.train()
    t_loss = 0
    t_count = 0
    pred_array = np.array([])
    label_array = np.array([])
    
    for batch_ppg, batch_target in tqdm(t_train_loader):
        
        batch_ppg = batch_ppg.transpose(1, 2).to(device)
        batch_target = batch_target.to(device)
        
        t_optimizer.zero_grad()
        output = t_model(batch_ppg)
                
        loss = loss_function(output, batch_target)
        loss.backward()
        t_optimizer.step()
        
        t_loss += loss.item()
        t_count += 1
        
        pred = (output > 0.5).int().to('cpu')
        batch_target = batch_target.int().to('cpu')
        
        pred = pred.numpy()
        batch_target = batch_target.numpy()
        
        pred_array = np.concatenate((pred_array, pred), axis=0)
        label_array = np.concatenate((label_array, batch_target), axis=0)
        
    print('Train Loss: {:.4f}'.format(t_loss/float(t_count)))
    
    scheduler.step()
    
    return pred_array, label_array

def main():
    args = parse_args()
    
    Train_Dataset = datasets.SignalDataset(path, is_test=False)
    
    train_dataset = DataLoader(Train_Dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, )
  
    # loss_function = nn.BCELoss()
    # loss_function = models.FocalLoss()
    # loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.pos]).to(device))
    loss_function = models.total_loss()
    
    
    model = models.BayesianSegmentationModel().to(device)
    learning_rate = 0.0001
    
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)

    for epoch in range(1, args.epochs+1):
        r_pred, r_label = train(model, train_dataset, optimizer, loss_function, scheduler)
        
        t_label_0_count = 0
        t_label_1_count = 0
        t_acc_0_count = 0
        t_acc_1_count = 0   
    
        for i in range(len(r_label)):
            if r_label[i] == 0:
                t_label_0_count += 1
                
                if r_label[i] == r_pred[i]:
                    t_acc_0_count += 1
                
            else:
                t_label_1_count += 1
                if r_label[i] == r_pred[i]:
                    t_acc_1_count += 1    

        print('-' * 20)
        print(epoch, ' / ',  args.epochs+1)
        print('F1 Score : ', f1_score(r_label, r_pred, average='macro'))
        print('t_label_0_count : ', t_acc_0_count, ' / ', t_label_0_count)
        print('t_label_1_count : ', t_acc_1_count, ' / ', t_label_1_count)
        
        
        if (epoch % 10 == 0) and (epoch > 0):
            torch.save(model.state_dict(), f'../../models/vitalDB/model_{str(epoch)}.pt')
            
    torch.save(model.state_dict(), f'../../models/vitalDB/model_final.pt')
    

if __name__ == '__main__':
    main()
