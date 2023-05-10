import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Define the Bayesian neural network with a density network architecture
class BayesianNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BayesianNet, self).__init__()

        # Define the layers with Bayesian parameters
        self.fc1_w_mu = nn.Parameter(torch.randn(input_size, hidden_size))
        self.fc1_w_rho = nn.Parameter(torch.randn(input_size, hidden_size))
        self.fc1_b_mu = nn.Parameter(torch.randn(hidden_size))
        self.fc1_b_rho = nn.Parameter(torch.randn(hidden_size))

        self.fc2_w_mu = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.fc2_w_rho = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.fc2_b_mu = nn.Parameter(torch.randn(hidden_size))
        self.fc2_b_rho = nn.Parameter(torch.randn(hidden_size))

        self.fc3_w_mu = nn.Parameter(torch.randn(hidden_size, output_size))
        self.fc3_w_rho = nn.Parameter(torch.randn(hidden_size, output_size))
        self.fc3_b_mu = nn.Parameter(torch.randn(output_size))
        self.fc3_b_rho = nn.Parameter(torch.randn(output_size))

        self.activation = nn.ReLU()

    def forward(self, x):
        # Sample the weights and biases from the posterior distributions
        fc1_w = dist.Normal(loc=self.fc1_w_mu, scale=torch.abs(self.fc1_w_rho)).rsample()
        fc1_b = dist.Normal(loc=self.fc1_b_mu, scale=torch.abs(self.fc1_b_rho)).rsample()
        fc2_w = dist.Normal(loc=self.fc2_w_mu, scale=torch.abs(self.fc2_w_rho)).rsample()
        fc2_b = dist.Normal(loc=self.fc2_b_mu, scale=torch.abs(self.fc2_b_rho)).rsample()
        fc3_w = dist.Normal(loc=self.fc3_w_mu, scale=torch.abs(self.fc3_w_rho)).rsample()
        fc3_b = dist.Normal(loc=self.fc3_b_mu, scale=torch.abs(self.fc3_b_rho)).rsample()

        # Forward pass through the layers
        x = x.view(-1, self.fc1_w_mu.size(0))
        x = self.activation(torch.matmul(x, fc1_w) + fc1_b)
        x = self.activation(torch.matmul(x, fc2_w) + fc2_b)
        x = torch.matmul(x, fc3_w) + fc3_b

        return x
    
    def bayesian_loss_fn(pred_class, target, aleatoric_uncertainty, epistemic_uncertainty):
            # Compute the classification loss
        classification_loss = F.nll_loss(pred_class, target, reduction='none')
        
        # Compute the aleatoric uncertainty loss
        aleatoric_loss = torch.mean(torch.sum(aleatoric_uncertainty ** 2, dim=-1))

        # Compute the epistemic uncertainty loss
        epistemic_loss = torch.mean(torch.sum(epistemic_uncertainty ** 2, dim=-1))

        # Combine the losses and return the total loss
        total_loss = classification_loss + aleatoric_loss + epistemic_loss

        return torch.mean(total_loss)

# Define the Bayesian classifier with an aleatoric loss function
class BayesianClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BayesianClassifier, self).__init__()

        self.net = BayesianNet(input_size, hidden_size, output_size)
        self.log_var = nn.Parameter(torch.randn(output_size))

    def forward(self, x, target=None):
        # Compute the logits and sample the predictions from the softmax distribution
        logits = self.net(x)
        probs = nn.functional.softmax(logits, dim=-1)

        if self.training:
            # Compute the negative log likelihood loss
            log_lik = dist.Categorical(probs=probs).log_prob(target).sum(-1)
            # aleatoric_loss = 0.5 * torch
            aleatoric_loss = 0.5 * torch.exp(-self.log_var) * (logits - target)**2 + 0.5 * self.log_var
            aleatoric_loss = aleatoric_loss.mean()

            # Compute the KL divergence between the prior and the posterior distributions
            kl_loss = 0.5 * (self.net.fc1_w_rho**2 + self.net.fc1_w_mu**2 - 2 * torch.log(torch.abs(self.net.fc1_w_rho)) -
                            self.net.fc2_w_rho**2 - self.net.fc2_w_mu**2 + 2 * torch.log(torch.abs(self.net.fc2_w_rho)) -
                            self.net.fc3_w_rho**2 - self.net.fc3_w_mu**2 + 2 * torch.log(torch.abs(self.net.fc3_w_rho)) -
                            self.net.fc1_b_rho**2 + self.net.fc1_b_mu**2 - 2 * torch.log(torch.abs(self.net.fc1_b_rho)) -
                            self.net.fc2_b_rho**2 + self.net.fc2_b_mu**2 - 2 * torch.log(torch.abs(self.net.fc2_b_rho)) -
                            self.net.fc3_b_rho**2 + self.net.fc3_b_mu**2 - 2 * torch.log(torch.abs(self.net.fc3_b_rho)))
            kl_loss = kl_loss.mean()

            # Compute the total loss
            loss = aleatoric_loss + kl_loss

            # Return the total loss, the aleatoric loss, and the predicted probabilities
            return loss, aleatoric_loss, probs

        else:
            # Compute the predicted class and the epistemic uncertainty
            pred_class = torch.argmax(probs, dim=-1)
            epistemic_uncertainty = torch.zeros_like(pred_class)

            # Sample from the posterior to compute the epistemic uncertainty
            for i in range(100):
                logits = self.net(x)
                probs = nn.functional.softmax(logits, dim=-1)
                pred_class = torch.argmax(probs, dim=-1)
                epistemic_uncertainty += (pred_class != pred_class)

            epistemic_uncertainty = epistemic_uncertainty.float() / 100.0

            # Return the predicted class, the aleatoric uncertainty, and the epistemic uncertainty
            return pred_class, torch.exp(self.log_var), epistemic_uncertainty
        
        
def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    train_loss = 0.0
    
    for x, target in train_loader:
        x, target = x.to(device), target.to(device)
        optimizer.zero_grad()
        loss, _, _ = model(x, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    return train_loss / len(train_loader.dataset)


def validate(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, target in val_loader:
            x, target = x.to(device), target.to(device)
            loss, _, _ = model(x, target)
            val_loss += loss.item()
            
    return val_loss / len(val_loader.dataset)


train_data = MNIST(root='data', train=True, transform=ToTensor(), download=True)
val_data = MNIST(root='data', train=False, transform=ToTensor(), download=True)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128, shuffle=False)

model = BayesianNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# class BayesianNet(nn.Module):
#     def init(self):
#         super(BayesianNet, self).init()
#         # Define the architecture of the neural network
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.fc1 = nn.Linear(128 * 7 * 7, 256)
#         self.fc2 = nn.Linear(256, 10)
        
#         # Define the parameters of the prior distribution
#         self.mu_0 = nn.Parameter(torch.zeros_like(self.fc1.weight))
#         self.rho_0 = nn.Parameter(torch.zeros_like(self.fc1.weight))
#         self.mu_1 = nn.Parameter(torch.zeros_like(self.fc2.weight))
#         self.rho_1 = nn.Parameter(torch.zeros_like(self.fc2.weight))
        
#         # Define the parameters of the noise distribution
#         self.sigma_1 = nn.Parameter(torch.ones_like(self.fc2.weight))
        
#     def forward(self, x):
#         # Forward pass through the convolutional layers
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.max_pool2d(x, 2)
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.max_pool2d(x, 2)
#         x = x.view(-1, 128 * 7 * 7)
        
#         # Forward pass through the fully connected layers
#         fc1_mu = F.linear(x, self.mu_0)
#         fc1_rho = F.linear(x, F.softplus(self.rho_0))
#         fc1_z = sample_normal(fc1_mu, fc1_rho)
#         fc1 = F.relu(fc)
#         fc2_mu = F.linear(fc1, self.mu_1)
#         fc2_rho = F.linear(fc1, F.softplus(self.rho_1))
#         fc2_z = sample_normal(fc2_mu, fc2_rho)
#         pred_class = F.log_softmax(self.fc2(fc2_z), dim=-1)
        
#         # Compute the aleatoric uncertainty
#         aleatoric_uncertainty = torch.log1p(torch.exp(self.sigma_1))
        
#         # Compute the epistemic uncertainty
#         epistemic_uncertainty = -torch.log(torch.abs(torch.sigmoid(fc1_rho)))
        
#         return pred_class, aleatoric_uncertainty, epistemic_uncertainty
    
#     def bayesian_loss_fn(pred_class, target, aleatoric_uncertainty, epistemic_uncertainty):
#         # Compute the classification loss
#         classification_loss = F.nll_loss(pred_class, target, reduction='none')
        
#         # Compute the aleatoric uncertainty loss
#         aleatoric_loss = torch.mean(torch.sum(aleatoric_uncertainty ** 2, dim=-1))

#         # Compute the epistemic uncertainty loss
#         epistemic_loss = torch.mean(torch.sum(epistemic_uncertainty ** 2, dim=-1))

#         # Combine the losses and return the total loss
#         total_loss = classification_loss + aleatoric_loss + epistemic_loss

#         return torch.mean(total_loss)
    
#     def train_loop(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs):
#         best_val_loss = float('inf')
        
#         for epoch in range(num_epochs):
#             train_loss = train(model, train_loader

