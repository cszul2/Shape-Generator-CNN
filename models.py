import torch.nn
import torch.optim
import torch.nn.functional

class BasicCNN(torch.nn.Module):
    def __init__(self, learningRate=0.001, momentum=0.9, inputSize=None):
        super().__init__()
        #((InputVolume−KernelSize+2*Padding)/Stride)+1
        self.convLayer1 = torch.nn.Conv2d(3, 6, 5) #Output Size = 150-5+1=146
        self.pool = torch.nn.MaxPool2d(2, 2) #Output Size = (146-2)/2+1=73
        self.convLayer2 = torch.nn.Conv2d(6, 16, 5) #Output Size = 73-5+1=69
        self.fullyConnectedLayer1 = torch.nn.Linear(18496, 120) 
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