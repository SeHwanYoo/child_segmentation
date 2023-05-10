import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BayesianSegmentationModel(nn.Module):
    def __init__(self, input_shape, num_classes, dropout_rate=0.5):
        super(BayesianSegmentationModel, self).__init__()
        
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(128*32*32, 256)
        self.dense2 = nn.Linear(256, num_classes)
    
    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        outputs = F.softmax(self.dense2(x), dim=1)
        return outputs
    
def total_loss(outputs, log_vars, targets):
    ce_loss = 0
    epistemic_loss = 0
    aleatoric_loss = 0
    for i in range(outputs.shape[1]):
        ce_loss += F.cross_entropy(outputs[:, i, :, :], targets[:, :, :], reduction='mean')
        epistemic_loss += 0.5 * torch.mean(log_vars[:, i, :, :])
        aleatoric_loss += 0.5 * torch.mean(torch.exp(-log_vars[:, i, :, :]) * (outputs[:, i, :, :] - targets[:, :, :])**2)
    return ce_loss + epistemic_loss + aleatoric_loss


def predict(model, inputs, num_samples=10):
    model.eval()
    outputs = torch.zeros((num_samples, inputs.shape[0], model.num_classes, inputs.shape[2], inputs.shape[3])).to(device)
    log_vars = torch.zeros((num_samples, inputs.shape[0], model.num_classes, inputs.shape[2], inputs.shape[3])).to(device)
    with torch.no_grad():
        for i in range(num_samples):
            output, log_var = model(inputs)
            outputs[i] = output
            log_vars[i] = log_var
    mean_output = torch.mean(outputs, dim=0)
    epistemic_var = torch.var(outputs, dim=0)
    aleatoric_var = torch.exp(torch.mean(log_vars, dim=0))
    total_var = epistemic_var + aleatoric_var
    return mean_output.cpu().numpy(), total_var.cpu().numpy()


def train(model, train_loader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, log_vars = model(inputs)
            loss = total_loss(outputs, log_vars, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if i % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                      .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, epoch_loss/len(train_loader)))


# def predict(model, inputs, num_samples=10):
#     model.eval()
#     outputs = torch.zeros((num_samples, inputs.shape[0], n, inputs.shape[2], inputs.shape[3])).to(device)
#     log_vars = torch.zeros((num_samples, inputs.shape[0], n, inputs.shape[2], inputs.shape[3])).to(device)
#     with torch.no_grad():
#         for i in range(num_samples):
#             output, log_var = model(inputs)
#             outputs[i] = output
#             log_vars[i] = log_var
#     mean_output = torch.mean(outputs, dim=0)
#     epistemic_var = torch.var(outputs, dim=0)
#     aleatoric_var = torch.exp(torch.mean(log_vars, dim=0))
#     total_var = epistemic_var + aleatoric_var
#     return mean_output.cpu().numpy(), total_var.cpu().numpy()
    
    
### classification   
# def epistemic_loss(outputs, targets):
#     mean_output = torch.mean(outputs, dim=0)
#     epistemic_var = torch.var(outputs, dim=0)
#     log_epistemic_var = torch.log(epistemic_var + 1e-8)
#     epistemic_loss = torch.mean(log_epistemic_var + (mean_output - targets)**2 / (2*epistemic_var))    
#     return epistemic_loss

# def aleatoric_loss(log_var, targets):
#     aleatoric_var = torch.exp(log_var) + 1e-8
#     log_aleatoric_var = torch.log(aleatoric_var)
#     aleatoric_loss = torch.mean(log_aleatoric_var + (targets - targets)**2 / (2*aleatoric_var))    
#     return aleatoric_loss

# def total_loss(outputs, log_var, targets):
#     cross_entropy = nn.CrossEntropyLoss()(outputs, targets)
#     epistemic = epistemic_loss(outputs, targets)
#     aleatoric = aleatoric_loss(log_var, targets)
#     total_loss = cross_entropy + epistemic + aleatoric
#     return total_loss  

    
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# def train_model(model, train_loader, optimizer, num_epochs):
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for i, (inputs, targets) in enumerate(train_loader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             model.train()
#             outputs = torch.zeros((inputs.shape[0], num_classes)).to(device)
#             log_vars = torch.zeros((inputs.shape[0], num_classes)).to(device)
#             for j in range(num_samples):
#                 output, log_var = model(inputs)
#                 outputs += output
#                 log_vars += log_var
#             outputs /= num_samples
#             log_vars /= num_samples
#             loss = total_loss(outputs, log_vars, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         print('Epoch %d loss: %.3f' % (epoch+1, running_loss / len(train_loader)))
        
        
# def predict(model, inputs, num_samples=10):
#     model.eval()
#     outputs = torch.zeros((num_samples, inputs.shape[0], num_classes)).to(device)
#     log_vars = torch.zeros((num_samples, inputs.shape[0], num_classes)).to(device)
#     with torch.no_grad():
#         for i in range(num_samples):
#             output, log_var = model(inputs)
#             outputs[i] = output
#             log_vars[i] = log_var
#     mean_output = torch.mean(outputs, dim=0)
#     epistemic_var = torch.var(outputs, dim=0)
#     aleatoric_var = torch.exp(torch.mean(log_vars, dim=0))
#     total_var = epistemic_var + aleatoric_var
#     return mean_output.cpu().numpy(), total_var.cpu().numpy()
        
        

# for epoch in range(10):
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print(f"Epoch {epoch+1} loss: {running_loss/len(train_loader)}")
    
    
# model.eval()
# predictions = []
# with torch.no_grad():
#     for inputs, _ in test_loader:
#         outputs = []
#         for i in range(10):
#             outputs.append(model(inputs))
#         outputs = torch.stack(outputs)
#         predictions.append(torch.mean(outputs, dim=0))
# predictions = torch.cat(predictions)