import torch
import torch.nn as nn

def real_loss(D_out, smooth=False, device='cuda'):
    batch_size = D_out.shape[0]
    labels = torch.ones(batch_size)
    
    labels = labels.to(device)
        
    if smooth:
        labels *= 0.9
        
    criterion = nn.BCEWithLogitsLoss().to(device)
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.shape[0]
    labels = torch.zeros(batch_size)

    labels = labels.to(device)
        
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss
