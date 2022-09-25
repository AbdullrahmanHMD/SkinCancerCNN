import torch
import time
from tqdm import tqdm
from evaluate import evaluate
from torch import optim

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


def train(model, train_loader, validation_loader, criterion, optimizer,
          epochs, scheduler=None, verbose=False, device=None):
    """ The training routine that will be used for the SkinCancerCNN.

    Args:
        model (nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The DataLoader object
            that contains the training data
        criterion (_type_): The loss function to be used for training.
        optimizer (_type_): The optimizer to be used for training.
        epochs (int): The number of training epochs
        verbose (bool, optional): If true, the function prints details about
            the training procedure. Defaults to False.

    Returns:
        _type_: _description_
    """
    if device is None:
        device = get_device()
    
    epoch_durations = []
    total_loss = []    
    
    accuracies_validation = []
    accuracies_train = []
    
    model.train()
    model = model.to(device=device)
    
    for epoch in tqdm(range(epochs)):
        
        epoch_loss = 0
        epoch_tic = time.time()
        count = 0
        for datum in train_loader:
            x, y = datum
            
            x = x.to(device=device)
            y = y.to(device=device)
            
            optimizer.zero_grad()
            
            y_pred = model(x)
            
            loss = criterion(y_pred, y)
            
            epoch_loss += loss.item()
            
            # This variable is used to calculate the mean loss for
            # each epoch so that it can be used with ReduceLROnPlateau.
            count += 1
            
            loss.backward()
            optimizer.step()
            
        print('Evaluating epoch...', flush=True)
        
        # Calculating test and train accuracies:
        test_accuracy = evaluate(model, validation_loader, device)
        train_accuracy = evaluate(model, train_loader, device)
        
        accuracies_validation.append(test_accuracy)
        accuracies_train.append(train_accuracy)
        
        # Appending the epoch loss to a list:
        total_loss.append(epoch_loss)
        
        # Calculating the time it takes for an epoch to complete:
        epoch_toc = time.time()
        epoch_time = epoch_toc - epoch_tic
        epoch_durations.append(epoch_time)

        # Advancing the scheduler:
        if scheduler is not None:
            # If the scheduler is ReduceLROnPlateau then provide the mean
            # loss to the step() method:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                lr = optimizer.param_groups[0]['lr']
                print(f'Learning rate: {lr}')
                scheduler.step(epoch_loss / count)
            else:
                lr = optimizer.param_groups[0]['lr']
                print(f'Learning rate: {lr}')
                scheduler.step()

        # Printing information about the epoch:
        if verbose:
            print(f'Epoch: {epoch} | Train_acc: {train_accuracy:.2f}% | Val_acc: {test_accuracy:.2f}% \
| Loss: {epoch_loss:.2f} | Runtime: {epoch_time:.2f} seconds')
        
    return total_loss, epoch_durations, accuracies_train, accuracies_validation

