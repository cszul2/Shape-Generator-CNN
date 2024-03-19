# %%
import torch
import numpy
import Models
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

for i in range(10):
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

#trainingFeatures, trainingLabels = next(iter(trainingDataLoader))
#print(f"Feature batch shape: {trainingFeatures.size()}")
#print(f"Labels batch shape: {len(trainingLabels)}")
#img = trainingFeatures[0].squeeze()
#label = trainingLabels[0]
#matplotlib.pyplot.imshow(numpy.transpose(img, (1,2,0)), cmap="gray")
#matplotlib.pyplot.show()
#print(f"Label: {label}")

# %%
model = Models.BasicCNN()
model.cuda()
for epoch in range(10):
    runningLoss = 0.0
    for index, data in enumerate(trainingDataLoader, 0):
        trainingFeatures, trainingLabels = data
        model.optimizer.zero_grad()
        outputs = model(trainingFeatures.cuda())
        loss = model.criterion(outputs, trainingLabels.cuda())
        loss.backward()
        model.optimizer.step()
        runningLoss += loss.item()
        if index % 2 == 1:
            print(f"[{epoch+1}, {index+1}] Loss: {runningLoss/20:.3f}")
print("Finished")
# %%
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testingDataLoader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images.cuda())
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')
# %%
