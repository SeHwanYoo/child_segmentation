import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Pre-trained ResNet-50 encoder for feature extraction.
    """
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu = self.relu(bn1)
        maxpool = self.maxpool(relu)
        layer1 = self.layer1(maxpool)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4

class Decoder(nn.Module):
    """
    Decoder with a Density Network architecture.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channels*2, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        mean = x[:, :self.out_channels, :, :]
        log_var = x[:, self.out_channels:, :, :]
        return mean, log_var

class SegmentationModel(nn.Module):
    def __init__(self, num_samples=10):
        super(SegmentationModel, self).__init__()
        self.num_samples = num_samples

    def forward(self, logits, masks):
        """
        Calculates the total uncertainty loss as the sum of epistemic and aleatoric uncertainty losses.

        Args:
            logits: The logits output by the segmentation model.
            masks: The ground truth segmentation masks.

        Returns:
            The total uncertainty loss.
        """
        batch_size, num_classes, height, width = logits.shape

        # Resample logits for Monte Carlo sampling
        logits_reshaped = logits.reshape(batch_size, num_classes, -1)
        weights = torch.softmax(logits_reshaped, dim=1)
        weights = weights.reshape(batch_size, num_classes, height, width)

        logits_resized = F.interpolate(logits, size=masks.shape[2:], mode='bilinear', align_corners=False)

        # Calculate epistemic uncertainty loss
        epistemic_loss = torch.mean(torch.var(weights, dim=0), dim=(1, 2))

        # Calculate aleatoric uncertainty loss
        aleatoric_loss = 0
        for i in range(self.num_samples):
            logits_sample = logits_resized + torch.randn_like(logits_resized)
            probs_sample = torch.softmax(logits_sample, dim=1)
            aleatoric_loss += torch.mean(torch.sum(-weights * torch.log(probs_sample + 1e-10), dim=1), dim=(1, 2))
        aleatoric_loss /= self.num_samples

        # Calculate total uncertainty loss
        loss = epistemic_loss + aleatoric_loss

        return loss