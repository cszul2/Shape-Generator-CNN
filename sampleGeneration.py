from PIL import Image, ImageDraw

class Sample:
    def __init__(self, sampleType, backgroundSize=None):
        self.makeBackground()
        if sampleType == "Square":
            self.makeSquare()
        elif sampleType == "Circle":
            self.makeCircle()

    def showSample(self):
        self.background.show()

    def makeBackground(self, size=(150,150), color=(255,255,255)):
        self.background = Image.new("RGB", size=size, color=color)
    
    def makeSquare(self, x0=50, y0=50, width=50, height=50, color=(0,0,0)):
        drawing = ImageDraw.Draw(self.background)
        drawing.rectangle([(x0, y0),(x0+width, y0+height)], fill=color)
    
    def makeCircle(self, x0=50, y0=50, width=50, height=50, color=(0,0,0)):
        drawing = ImageDraw.Draw(self.background)
        drawing.ellipse([(x0, y0),(x0+width, y0+height)], fill=color)