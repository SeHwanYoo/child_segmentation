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
    """
    Segmentation model that combines the encoder and decoder.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(2048, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        mean, log_var = self.decoder(features)
        return mean, log_var
    
    
class BayesianLoss(nn.Module):
    def __init__(self, num_samples=10):
        super().__init__()
        self.num_samples = num_samples

    def forward(self, logits, masks):
        aleatoric_loss = 0
        epistemic_loss = 0

        # Calculate aleatoric uncertainty loss
        for i in range(self.num_samples):
            probs_sample = F.softmax(logits, dim=1)
            weights = 1.0 - probs_sample

            # expand weights tensor to match the number of output channels in probs_sample
            weights = weights.expand(-1, probs_sample.size(1), -1, -1, -1)

            aleatoric_loss += torch.mean(torch.sum(-weights * torch.log(probs_sample + 1e-10), dim=1), dim=(1, 2))

        aleatoric_loss /= self.num_samples

        # Calculate epistemic uncertainty loss
        probs = F.softmax(logits, dim=1)
        epistemic_loss += torch.sum(torch.square(probs - torch.mean(probs, dim=0)), dim=1)
        epistemic_loss /= (2 * self.num_samples)

        # Total loss is the sum of aleatoric and epistemic losses
        loss = aleatoric_loss + epistemic_loss

        # Apply the segmentation mask
        loss = -torch.mean(torch.sum(masks * loss, dim=1))

        return loss