from torch.optim import Adam, AdamW

def getOptimizer(model, args):
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    
    optimizer.zero_grad()
    
    return optimizer