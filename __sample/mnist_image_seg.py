import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np 


# Set random seed for reproducibility
torch.manual_seed(42)

# Define transforms to apply to the images
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor()
])


# Import the necessary libraries
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

# Define the EfficientNet model for image segmentation
class EfficientNetSeg(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetSeg, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.conv = nn.Conv2d(1280, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1, x2, x3, x4, x5 = self.backbone.extract_features(x)
        
        # Decoder
        d5 = self.conv(x5)
        d4 = self.conv(x4)
        d3 = self.conv(x3)
        d2 = self.conv(x2)
        d1 = self.conv(x1)
        
        # Upsample
        d4 = F.interpolate(d5, scale_factor=2, mode='bilinear', align_corners=True) + d4
        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True) + d3
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True) + d2
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True) + d1
        
        # Output
        out = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)
        return out


class BayesianSegNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, dropout_rate=0.5):
        # super(BayesianSegNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(p=self.dropout_rate)
        self.dropout2 = nn.Dropout2d(p=self.dropout_rate)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, self.out_channels)
        
        # super(BayesianSegNet, self).__init__()
        # self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout = nn.Dropout(p=self.dropout_rate)
        # self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        # self.fc2 = nn.Linear(4096, 4096)
        # self.fc3 = nn.Linear(4096, self.out_channels)
        # self.log_var = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # First convolutional block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Flatten and pass through fully-connected layers
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # log_var = self.log_var.expand
        log_var = torch.log(torch.exp(x) + 1)
        
        # Apply softmax to get the class probabilities
        output = F.softmax(x, dim=1)
        
        # # Compute the aleatoric variance
        
        
        # Forward pass through convolutional layers
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = self.pool1(x)
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = self.pool2(x)
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))
        # x = F.relu(self.conv7(x))
        # x = self.pool3(x)
        # x = F.relu(self.conv8(x))
        # x = F.relu(self.conv9(x))
        # x = F.relu(self.conv10(x))
        # x = self.pool4(x)
        # x = F.relu(self.conv11(x))
        # x = F.relu(self.conv12(x))
        # x = F.relu(self.conv13(x))
        # x = self.pool5(x)

        # # Flatten and pass through fully connected layers for aleatoric uncertainty estimation
        # x = x.view(-1, 512 * 7 * 7)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        # # log_var = self.log_var.expand
        # log_var = torch.log(torch.exp(x) + 1)
        
        # # Apply softmax to get the class probabilities
        # output = F.softmax(x, dim=1)
        
        return output, log_var
    
    
def total_loss(output, target, log_var, num_samples=10):
    # Compute the mean squared error loss between the output and target
    mse_loss = F.mse_loss(output, target, reduction='mean')
    
    # Compute the epistemic uncertainty loss
    epi_loss = torch.mean(torch.sum(log_var, dim=1))
    
    # Compute the aleatoric uncertainty loss
    ale_loss = 0.0
    for i in range(num_samples):
        sample = torch.randn_like(output) * torch.exp(log_var / 2.0)
        ale_loss += F.mse_loss(sample, output, reduction='mean')
    ale_loss /= num_samples
    
    # Total loss is a combination of the mean squared error loss, the epistemic uncertainty loss,
    # and the aleatoric uncertainty loss
    total_loss = mse_loss + epi_loss + ale_loss
    
    return total_loss

def predict(model, dataloader, num_samples=10):
    # Set model to evaluation mode
    model.eval()
    
    # Lists to store the mean predictions and uncertainties for each input
    mean_preds = []
    epi_uncertainties = []
    ale_uncertainties = []
    
    # Iterate over the inputs in the dataloader
    for input, target in dataloader:
        # Move the input and target to the device
        input = input.to(device)
        target = target.to(device)
        
        # Repeat the input multiple times to compute Monte Carlo samples
        input = input.repeat(num_samples, 1, 1, 1)
        
        # Compute the output and log_var for the input
        output, log_var = model(input)
        
        # Compute the mean prediction and uncertainty estimates for each input
        mean_pred = torch.mean(output, dim=0)
        epi_uncertainty = torch.sum(torch.exp(log_var), dim=0) / num_samples
        ale_uncertainty = 0.0
        for i in range(num_samples):
            sample = torch.randn_like(output) * torch.exp(log_var / 2.0)
            ale_uncertainty += torch.sum((sample - mean_pred)**2, dim=0)
        ale_uncertainty /= num_samples
        
        # Append the mean prediction and uncertainties to the lists
        mean_preds.append(mean_pred.detach().cpu().numpy())
        epi_uncertainties.append(epi_uncertainty.detach().cpu().numpy())
        ale_uncertainties.append(ale_uncertainty.detach().cpu().numpy())
    
    # Convert the lists to numpy arrays
    mean_preds = np.array(mean_preds)
    epi_uncertainties = np.array(epi_uncertainties)
    ale_uncertainties = np.array(ale_uncertainties)
    
    # Return the mean predictions and uncertainties
    return mean_preds, epi_uncertainties, ale_uncertainties


import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BayesianSegNet(num_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Load the MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Define the data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 100

for epoch in range(num_epochs):
    # Set model to training mode
    model.train()

    # Initialize running loss and accuracy
    running_loss = 0.0
    correct = 0
    total = 0

    # Loop over the training data in batches
    for i, data in enumerate(train_loader):
        # Get inputs and labels from batch
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs, log_var = model(inputs)
        loss = total_loss(outputs, labels, log_var)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Update running loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print statistics
    print('Epoch: %d | Loss: %.3f | Accuracy: %.3f' %
          (epoch + 1, running_loss / len(train_loader),
           100 * correct / total))

print('Finished Training')