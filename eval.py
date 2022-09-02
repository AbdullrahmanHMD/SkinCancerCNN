import torch

def get_device():
    """ Returns the device according to the following:
        If the system supports "cuda" cuda is returned
        Otherwise "cpu" is returned

    Returns:
        str: The device in which the training will occur.
    """
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    return device

def evaluate(model, data_loader, device=None):
    if device is None:
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