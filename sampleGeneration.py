import numpy
from PIL import Image, ImageDraw

class Sample:
    def __init__(self, sampleType, **kwargs):
        self.makeBackground(**kwargs)
        self.sampleType = sampleType
        if self.sampleType == "Square":
            self.makeSquare(**kwargs)
        elif self.sampleType == "Ellipse":
            self.makeEllipse(**kwargs)

    def showSample(self):
        self.background.show()

    def saveSample(self, filename=None):
        if filename is None:
            self.background.save(self.sampleType+".png")
        else:
            self.background.save(filename)

    def makeNumpyVersion(self):
        self.sample = numpy.array(self.background)

    def makeBackground(self, backgroundSize=(150,150), backgroundColor=(255,255,255)):
        self.background = Image.new("RGB", size=backgroundSize, color=backgroundColor)
    
    def makeSquare(self, x0=50, y0=50, width=50, height=50, color=(0,0,0)):
        drawing = ImageDraw.Draw(self.background)
        drawing.rectangle([(x0, y0),(x0+width, y0+height)], fill=color)
    
    def makeEllipse(self, x0=25, y0=50, width=100, height=50, color=(0,0,0)):
        drawing = ImageDraw.Draw(self.background)
        drawing.ellipse([(x0, y0),(x0+width, y0+height)], fill=color)