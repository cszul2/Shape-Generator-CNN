# %%
import torch
import numpy
import Models
import SampleGeneration
import DataPreprocessing
import matplotlib.pyplot
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

samples = list()
trainingLabelsList = list()
trainingFeaturesList = list()
testingLabelsList = list()
testingFeaturesList = list()
classes = ["Ellipse", "Square"]

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

for i in range(100):
    for type in classes:
        sample = SampleGeneration.Sample(type)
        samples.append(sample)
trainingSet, testingSet = train_test_split(samples, test_size=0.25)

for sample in trainingSet:
    trainingFeaturesList.append(sample.sample)
    for index, item in enumerate(classes):
        if item == sample.sampleType:
            trainingLabelsList.append(index)

for sample in testingSet:
    testingFeaturesList.append(sample.sample)
    for index, item in enumerate(classes):
        if item == sample.sampleType:
            testingLabelsList.append(index)

trainingData = DataPreprocessing.ShapeDataset(trainingLabelsList, trainingFeaturesList, 
                                              transform=ToTensor())
testingData = DataPreprocessing.ShapeDataset(testingLabelsList, testingFeaturesList, 
                                             transform=ToTensor())

trainingDataLoader = DataLoader(trainingData, batch_size=4, shuffle=True)
testingDataLoader = DataLoader(testingData, batch_size=4, shuffle=True)

# %%
model = Models.BasicCNN()
model.to(device)

for epoch in range(10):
    runningLoss = 0.0
    for index, data in enumerate(trainingDataLoader, 0):
        trainingFeatures, trainingLabels = data
        trainingFeatures = trainingFeatures.to(device)
        trainingLabels = trainingLabels.to(device)
        model.optimizer.zero_grad()
        outputs = model(trainingFeatures)
        loss = model.criterion(outputs, trainingLabels)
        loss.backward()
        model.optimizer.step()
        runningLoss += loss.item()
        if index % 2 == 1:
            print(f"[{epoch+1}, {index+1}] Loss: {runningLoss/20:.3f}")
print("Finished")

# %%
correct = 0
total = 0
with torch.no_grad():
    for data in testingDataLoader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')
# %%
