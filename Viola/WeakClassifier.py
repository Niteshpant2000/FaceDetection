class WeakClassifier:
    def __init__(self,positive,negative,threshold,polarity):
        self.positive=positive
        self.negative=negative
        self.threshold=threshold
        self.polarity=polarity
    def classifier(self,x):
        feature=lambda ii: sum([pos.comp_feature for pos in self.positive])-sum([neg.comp_feature(ii) for neg in self.negative])
        return 1 if self.polarity*feature(x) <self.polarity*self.threshold else 0
    







            