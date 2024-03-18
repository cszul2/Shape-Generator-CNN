import torch.nn
import torch.optim
import torch.nn.functional

class BasicCNN(torch.nn.Module):
    def __init__(self, learningRate=0.001, momentum=0.9, imageHeight=None, imageWidth=None):
        super().__init__()
        self.convLayer1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.convLayer2 = torch.nn.Conv2d(6, 16, 5)
        print(self.convLayer2.padding)
        print(self.convLayer2.kernel_size)
        print(self.convLayer2.stride)
        #[(InputVolume−KernelSize+2*Padding)/Stride]+1
        self.fullyConnectedLayer1 = torch.nn.Linear(16*5*5, 120) 
        self.fullyConnectedLayer2 = torch.nn.Linear(120, 84)
        self.fullyConnectedLayer3 = torch.nn.Linear(84, 2)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learningRate,
                                         momentum=momentum)

    def forward(self, sample):
        x = self.pool(torch.nn.functional.relu(self.convLayer1(sample)))
        x = self.pool(torch.nn.functional.relu(self.convLayer2(x)))
        x = torch.flatten(x, 1)
        x = torch.nn.functional.relu(self.fullyConnectedLayer1(x))
        x = torch.nn.functional.relu(self.fullyConnectedLayer2(x))
        output = self.fullyConnectedLayer3(x)
        return output