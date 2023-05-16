import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

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

path = '../../datasets/Atopy Segmentation'

ints = ['Intersect_0.75', 'Intersect_0.8', 'Intersect_0.85']
grds = ['Grade0', 'Grade1', 'Grade2', 'Grade3']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', required=False, type=int, default=256)
    parser.add_argument('--epochs', required=False, type=int, default=100)
    parser.add_argument('--eval_epocs', required=False, type=int, default=10)
    
    parser.add_argument('--ints', required=False, type=int, default=2)
    parser.add_argument('--grds', required=False, type=int, default=3)
    args = parser.parse_args()
    
    return args

def train(t_model, t_train_loader, t_optimizer, t_loss_func):    
# def train(t_model, t_train_loader, t_optimizer):    
    t_model.train()
    # t_loss = 0
    # t_count = 0
    # pred_array = np.array([])
    # label_array = np.array([])
    
    epoch = 0 
    total_epochs = len(t_train_loader) 
    for images, masks in tqdm(t_train_loader):
        
        images = images.to(device)
        masks = masks.to(device)
        
        # Generate predictions from the model and compute total uncertainty
        logits = t_model(images)
        # preds = torch.argmax(logits, dim=1)
        
        t_optimizer.zero_grad()
        
        # Compute loss, weighting by uncertainty
        # loss = t_model.compute_loss(logits, masks)
        loss = t_loss_func(logits, masks)
        # loss = models.bayesian_loss(logits, masks)
        # weighted_loss = (loss * uncertainty).mean()
        
        # Backpropagate and update parameters
        loss.backward()
        
        t_optimizer.step()
        
        # Compute accuracy metrics for this batch, weighting by uncertainty
        # correct_pixels = (preds == masks).sum(dim=(1, 2)).float()
        # weighted_correct_pixels = (correct_pixels * uncertainty).sum()
        # total_correct_pixels += weighted_correct_pixels.item()
        # total_pixels += np.prod(masks.shape)
        
        # Update running loss
        # running_loss += weighted_loss.item() * images.size(0)
        
    # Compute loss and accuracy metrics for the epoch
    # epoch_loss = running_loss / len(t_train_loader.dataset)
    # pixel_accuracy = total_correct_pixels / total_pixels
    
    # print(f"Epoch {epoch+1}/{total_epochs} | Train Loss: {epoch_loss:.4f} | Train Pixel Accuracy: {pixel_accuracy:.4f}")

    # return pred_array, label_array

def evaluate(model, dataloader):
    # Set model to evaluation mode
    model.eval()
    
    # Initialize variables to keep track of accuracy metrics
    total_correct_pixels = 0
    total_pixels = 0
    
    # Iterate through dataloader to evaluate model on entire dataset
    with torch.no_grad():
        for images, masks in dataloader:
            # Move inputs to the specified device
            images = images.to(device)
            masks = masks.to(device)
            
            # Generate predictions from the model and compute total uncertainty
            logits, _ = model(images)
            preds = torch.argmax(logits, dim=1)
            uncertainty = 1 - torch.softmax(logits, dim=1).max(dim=1)[0]
            
            # Compute accuracy metrics for this batch, weighting by uncertainty
            correct_pixels = (preds == masks).sum(dim=(1, 2)).float()
            weighted_correct_pixels = (correct_pixels * uncertainty).sum()
            total_correct_pixels += weighted_correct_pixels.item()
            total_pixels += np.prod(masks.shape)
    
    # Compute overall accuracy metrics
    pixel_accuracy = total_correct_pixels / total_pixels
    return pixel_accuracy


def main():
    args = parse_args()
    
    # Define any data augmentation transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])
    
    Train_Dataset = datasets.SegDataset(path, ints=ints[args.ints], grds=grds[args.grds], transform=transform, is_test=False)
    train_dataset = DataLoader(Train_Dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, )
    
    Test_Dataset = datasets.SegDataset(path, ints=ints[args.ints], grds=grds[args.grds],is_test=True)
    test_dataset = DataLoader(Test_Dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, )
  
    loss_func = nn.CrossEntropyLoss()
    
    model = models.SegmentationModel().to(device)
    learning_rate = 0.0001
    
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)

    # train
    for _ in range(1, args.epochs+1):
        # r_pred, r_label = train(model, train_dataset, optimizer, loss_function, scheduler)
       train(model, train_dataset, optimizer, loss_func)
    #    train(model, train_dataset, optimizer)
    
    
    # eval 
    random_seed = 1234
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    pixel_accuracy_list = []
    for i in range(args.eval_epocs):
        
        torch.manual_seed(random_seed + i)
        np.random.seed(random_seed + i)
        random.seed(random_seed + i)
        
        pixel_accuracy = evaluate(model, test_dataset) 
        pixel_accuracy_list.append(pixel_accuracy)
        
    mean_pixel_accuracy = np.mean(pixel_accuracy_list)
    stddev_pixel_accuracy = np.std(pixel_accuracy_list)
    
    print(f"Pixel accuracy: {mean_pixel_accuracy:.4f} +/- {stddev_pixel_accuracy:.4f}")

if __name__ == '__main__':
    main()
