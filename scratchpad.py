# %%
import numpy
import SampleGeneration
import DataPreprocessing
import matplotlib.pyplot
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

samples = list()
trainingLabelsList = list()
trainingFeaturesList = list()
testingLabelsList = list()
testingFeaturesList = list()
classes = ["Ellipse", "Square"]

for i in range(100):
    for type in classes:
        sample = SampleGeneration.Sample(type)
        samples.append(sample)
trainingSet, testingSet = train_test_split(samples, test_size=0.25)

for sample in trainingSet:
    trainingFeaturesList.append(sample.sample)
    trainingLabelsList.append(sample.sampleType)

for sample in testingSet:
    testingFeaturesList.append(sample.sample)
    testingLabelsList.append(sample.sampleType)

trainingData = DataPreprocessing.ShapeDataset(trainingLabelsList, trainingFeaturesList, 
                                              transform=ToTensor())
testingData = DataPreprocessing.ShapeDataset(testingLabelsList, testingFeaturesList, 
                                             transform=ToTensor())

trainingDataLoader = DataLoader(trainingData, batch_size=4, shuffle=True)
testingDataLoader = DataLoader(testingData, batch_size=4, shuffle=True)

trainingFeatures, trainingLabels = next(iter(trainingDataLoader))
print(f"Feature batch shape: {trainingFeatures.size()}")
print(f"Labels batch shape: {len(trainingLabels)}")
img = trainingFeatures[0].squeeze()
label = trainingLabels[0]
matplotlib.pyplot.imshow(numpy.transpose(img, (1,2,0)), cmap="gray")
matplotlib.pyplot.show()
print(f"Label: {label}")
# %%
