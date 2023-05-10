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
    
    
def bayesian_loss(mean=None, log_var=None, target=None):
    # Compute the negative log likelihood
    log_likelihood = -0.5 * torch.sum(((target - mean) ** 2) / torch.exp(log_var), dim=(1, 2, 3)) \
                     - 0.5 * torch.sum(log_var, dim=(1, 2, 3)) \
                     - 0.5 * (target.shape[1] * target.shape[2] * target.shape[3]) \
                     * (torch.log(torch.tensor(2.0 * 3.141592653589793, device=target.device))
                        + torch.log(torch.exp(log_var)))

    neg_log_likelihood = torch.mean(log_likelihood)

    # Compute the epistemic uncertainty
    epistemic_uncertainty = -0.5 * torch.sum(log_var, dim=(1, 2, 3))

    # Compute the aleatoric uncertainty
    aleatoric_uncertainty = 0.5 * torch.sum(torch.exp(log_var), dim=(1, 2, 3))

    # Compute the total uncertainty
    total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

    # Compute the loss as a weighted sum of the negative log likelihood and the total uncertainty
    beta = 1.0  # hyperparameter that controls the weighting between the two terms
    loss = neg_log_likelihood + beta * torch.mean(total_uncertainty)

    return loss


# Define the loss function
# def bayesian_loss(mean, log_var, target):
#     aleatoric_uncertainty = torch.exp(log_var)
#     # Compute the epistemic uncertainty using Monte Carlo dropout
#     num_samples = 10
#     outputs = []
#     for i in range(num_samples):
#         output = mean + aleatoric_uncertainty * torch.randn_like(mean)
#         outputs.append(output)
#     outputs = torch.stack(outputs)
#     mean_output = torch.mean(outputs, dim=0)
#     epistemic_uncertainty = torch.var(outputs, dim=0)
#     # Compute the negative log likelihood
#     log_likelihood = -0.5 * torch.sum((target - mean_output)**2 / aleatoric_uncertainty + torch.log(aleatoric_uncertainty) + 2 * torch.log(torch.tensor(2.0 * 3.141592653589793, device=target.device)))  
#     # constant term
#     neg_log_likelihood = torch.mean(log_likelihood)
#     # Compute the total uncertainty
#     total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
#     # Compute the loss as a weighted sum of the negative log likelihood and the total uncertainty
#     beta = 1.0  # hyperparameter that controls the weighting between the two terms
#     loss = neg_log_likelihood + beta * torch.mean(total_uncertainty)
#     return loss
