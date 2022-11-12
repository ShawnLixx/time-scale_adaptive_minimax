import torch
import torch.nn as nn
import torch.nn.functional as F

from config import args


def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

class MLP(nn.Module) :
    def __init__(self, activation='relu'):
        super(MLP, self).__init__()
        
        self.linear1 = nn.Linear(2,4) #input dimension:2
        self.linear2 = nn.Linear(4,2)
        self.linear3 = nn.Linear(2,2)
        if activation == 'relu':
            self.active = nn.ReLU() 
        else :
            self.active = nn.ELU()
    
    def forward(self,input):
        x = self.active(self.linear1(input))
        x = self.active(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def init_weights_glorot(self):
        for m in self._modules :
            if type(m) == nn.Linear:
                nn.init.xavier_uniform(m.weight)

class CNN(nn.Module):
    # initializers, d=num_filters
    def __init__(self, d=32, activation='elu'):
        super(CNN, self).__init__()
        
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=d, kernel_size=(8, 8)), #(28-8 )+1 = 21
            nn.BatchNorm2d(d),
            nn.ELU(),
    
            # Layer 2
            nn.Conv2d(in_channels=d, out_channels=2*d, kernel_size=(6, 6)), # (21-6)+1 = 16 
            nn.BatchNorm2d(2*d)  ,          
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # 8 
            
            # Layer 3
            nn.Conv2d(in_channels=2*d, out_channels=4*d, kernel_size=(5, 5)), # (8-5)+1 = 4
            nn.BatchNorm2d(4*d),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # chanel 128 feature map 2*2
            
        )
        # Logistic Regression
        self.clf = nn.Linear(512, 10)

    def init_weights(self, mean, std):
        for m in self._modules:
            if type(m) == nn.Linear:
                nn.init.xavier_uniform(m.weight)
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()


    def forward(self, input): 
        
        x = self.conv(input)
        return self.clf(x.view(len(x), -1 ))

def get_model():
    if args.model == 'MLP':
        model = MLP()
    elif args.model == 'CNN':
        model = CNN()
        model.init_weights(mean=0.0, std=0.02)
    else:
        raise NotImplementedType

    model = model.to(args.device)
    return model
