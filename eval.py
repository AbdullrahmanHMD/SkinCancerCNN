import torch
from train import get_device


def evaluate(model, data_loader):
    device = get_device()
    
    model.eval()
    model = model.to(device=device)
    accuracy = 0
    for x, y, _ in data_loader:
        
        x = x.to(device=device)
        y = y.to(device=device)
        
        y_pred = model(x)
        
        _, label = torch.max(y_pred, axis=1)
        num_correct += (y == label).sum().item()
        
    accuracy = num_correct / len(data_loader) * data_loader.batch_size
    
    return accuracy