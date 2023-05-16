import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class SegmentationModel(nn.Module):
    def __init__(self, num_classes=2):
        super(SegmentationModel, self).__init__()
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1(x)
        x = self.conv2(x)
        logits = self.conv3(x)
        
        return logits
    
    def compute_loss(self, x, y_true):
        logits = self(x)
        softmax_logits = F.softmax(logits, dim=1)
        
        # Aleatoric uncertainty
        aleatoric_uncertainty = torch.mean(torch.log(torch.sum(softmax_logits ** 2, dim=1)))
        
        # Epistemic uncertainty
        sample_logits = torch.stack([self(x) for _ in range(10)], dim=0)
        mean_logits = torch.mean(sample_logits, dim=0)
        epistemic_uncertainty = torch.mean(torch.sum((softmax_logits - mean_logits) ** 2, dim=1))
        
        # Total loss
        loss = F.cross_entropy(logits, y_true)
        total_loss = loss + aleatoric_uncertainty + epistemic_uncertainty
        
        return total_loss
