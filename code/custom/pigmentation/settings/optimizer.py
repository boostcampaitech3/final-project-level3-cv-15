from torch.optim import Adam, AdamW

def getOptimizer(model, optimizer, lr):
    if optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0)
    
    optimizer.zero_grad()
    
    return optimizer