import torch
import Models
import SampleGeneration
import DataPreprocessing
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

numberOfEachSampleType = 100
testSizePercentage = 0.25
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