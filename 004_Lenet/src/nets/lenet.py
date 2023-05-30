from torch import nn

'''
Notice: How the LeNet architecture uses the tanh activation 
function rather than the more popular ReLU.
Back in 1998 the ReLU had not been used in the context of 
deep learning — it was more common to use tanh or sigmoid 
as an activation function. When implementing LeNet today, it’s 
common to swap out TANH for RELU — we’ll follow this same 
guideline and use ReLU as our activation function later in this section.
'''
class Letnet(nn.Module):
    def __init__(self, num_classes=10):
        super(Letnet, self).__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), 
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), 
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), 
            nn.ReLU(),
            nn.LazyLinear(84), 
            nn.ReLU(),
            nn.LazyLinear(num_classes),
            nn.Softmax(dim=1))
        
    def forward(self, x):
        return self.net(x)