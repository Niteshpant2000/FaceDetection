
class Rectangle:
    def __init__(self,x,y,width,height):
        self.x=x
        self.y=y
        self.width=width
        self.height=height
    def comp_feature(self,iiv):
        return iiv[self.y+self.height][self.x+self.width]+iiv[self.y][self.x]-(iiv[self.y+self.height][self.x]+iiv[self.y][self.x+self.width])
