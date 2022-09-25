import torch
from tqdm import tqdm

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
    num_correct = 0
    count = 0
    for datum in data_loader:
        
        x, y = datum
        
        x = x.to(device=device)
        y = y.to(device=device)
        
        y_pred = model(x)
        
        _, label = torch.max(y_pred, axis=1)
        num_correct += (y == label).sum().item()
        # Using count variable becuase len(data_loader) * data_loader.batch_size does not
        # necessarly return the correct number of data points. Consider the case where
        # the number of data points is not divisible by the batch size.
        count += 1 * label.shape[0]
        
    accuracy = 100 * num_correct / count
    
    return accuracy