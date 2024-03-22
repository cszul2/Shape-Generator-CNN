import torch
import Models
import SampleGeneration
import DataPreprocessing
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

numberOfEachSampleType = 100
testSizePercentage = 0.25
numberOfEpochs = 10
batchSize = 4


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

samples = list()
trainingLabelsList = list()
trainingFeaturesList = list()
testingLabelsList = list()
testingFeaturesList = list()
classes = ["Ellipse", "Square"]

for i in range(numberOfEachSampleType):
    for type in classes:
        sample = SampleGeneration.Sample(type)
        samples.append(sample)
trainingSet, testingSet = train_test_split(samples, test_size=testSizePercentage)

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

trainingDataLoader = DataLoader(trainingData, batch_size=batchSize, shuffle=True)
testingDataLoader = DataLoader(testingData, batch_size=batchSize, shuffle=True)

model = Models.BasicCNN()
model.to(device)
for epoch in range(numberOfEpochs):
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
print("Training Finished")

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
print(f'Accuracy of CNN on test images: {100 * correct // total} %')