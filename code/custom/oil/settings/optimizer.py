from torch import optim

def get_optimizer(model, args):
    origin = ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'SparseAdam', 'Adamax', 
              'ASGD', 'LBFGS', 'NAdam', 'RAdam', 'RMSprop', 'Rprop', 'SGD']
    lowers = ['adadelta', 'adagrad', 'adam', 'adamw', 'sparseadam', 'adamax', 
              'asgd', 'lbfgs', 'nadam', 'radam', 'rmsprop', 'rprop', 'sgd']
    if args.optimizer.lower() in lowers:
        idx = lowers.index(args.optimizer.lower())
        exec(f'optimizer = optim.{origin[idx]}(model.parameters(), lr=args.lr)')
    else : raise Exception(f"{args.optimizer} is not in torch.optim")
    optimizer.zero_grad()
    
    return optimizer

# from torch.optim import Adam, AdamW

# def getOptimizer(model, optimizer, lr):
#     if optimizer == 'adam':
#         optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0)
    
#     optimizer.zero_grad()
    
#     return optimizer
