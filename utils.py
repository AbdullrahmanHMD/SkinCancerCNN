import torch
import os

def save_model(model, path, model_name):
    final_path = os.path.join(path, model_name)
    with open(final_path, 'wb') as file:
        torch.save({'model_state_dict': model.state_dict()}, file)


def load_model_state_dict(path):
    path = os.path.join(path)
    with open(path, 'rb') as file:
        model_state_dict = torch.load(file)['model_state_dict']
    
    return model_state_dict