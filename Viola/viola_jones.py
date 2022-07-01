
from Integral_image import integral
from Rectangle import Rectangle
import numpy as np
import WeakClassifier
import math
from sklearn.feature_selection import SelectPercentile,f_classif
import pickle
class ViolaJones:
    def __init__(self,T=10):
        self.T=T # number of weak classifiers
        self.alphas=[]
        self.clfs=[]
    def train(self,training,positive,negative):
        weights=np.zeros(len(training))
        training_data=[]
        print("Integral images")
        for x in range(len(training)):
            training_data.append((integral(training[x][0]),training[x][1]))

            if(training[x][1]==1):
                weights[x]=1.0/(2*positive)
            else:
                weights[x]=1.0/(2*negative)
        print("feature building")
        features=self.features(training_data[0][0].shape)
        print("Applying features")
        X,Y=self.feature_apply(features,training_data)
        print("selecting best features")
        indices=SelectPercentile(f_classif,percentile=10).fit(X.T,Y).get_support(indices=True)
        X=X[indices]
        features=features[indices]
        print("Selected %d potential features" %len(x))
        for t in range(self.T):
            weights=weights/np.linalg.norm(weights)
            weak_classifiers=self.train_weak(X,Y,features,weights)
            clf,error,accuracy=self.select_best(weak_classifiers,weights,training_data)
            beta=error/(1.0-error)
            for i in range(len(accuracy)):
                weights[i]=weights[i]*(beta**(1-accuracy[i]))
            alpha=math.log(1.0/beta)
            self.alphas.append(alpha)
            self.clfs.append(clf)
            print("chose classifier: %s with accuracy: %f and alpha: %f" % (str(clf),len(accuracy)-sum(accuracy),alpha))

      


    def train_weak(self,X,y,features,weights):
        total_pos,total_neg=0,0
        for w, label in zip(weights,y):
            if label==1:
                total_pos+=w
            else:
                total_neg+=w
        classify=[]
        total_features=X.shape[0]
        for index, feature in enumerate(X):
            if(len(classify)%1000==0 and len(classify)!=0):
                print("Trained %d classifiers out of %d" %(len(classify),total_features))
            applied_feature=sorted(zip(weights,feature,y),key=lambda x:x[1])
            pos_seen,neg_seen=0,0
            pos_weights,neg_weights=0,0
            min_error,best_feature,best_threshold,best_polarity=float("inf"),None,None,None
            for w,f, label in applied_feature:
                error=min(neg_weights+total_pos-pos_weights,pos_weights+total_neg-neg_weights)
                if(error<min_error):
                    min_error=error
                    best_feature=features[index]
                    best_threshold=f
                    best_polarity=1 if pos_seen>neg_seen else -1
                if(label==1):
                    pos_seen+=1
                    pos_weights+=w
                else:
                    neg_seen+=1
                    neg_weights+=w
            clf=WeakClassifier(best_feature[0],best_feature[1],best_threshold,best_polarity)
            classify.append(clf)
        return classify
    def select_best(self,classify,weights,tdata):
        best_clf,best_error,best_accuracy=None,float('inf'),None
        for clf in classify:
            error,accuracy=0,[]
            for data, w in zip(tdata,weights):
                correctness=abs(clf.classifier(data[0])-data[1])
                accuracy.append(correctness)
                error+=w*correctness
            error=error/len(tdata)
            if error<best_error:
                best_clf,best_error,best_accuracy=clf,error,accuracy
        return best_clf,best_error,best_accuracy



    def features(self,image_shape):
        height,width=image_shape
        features=[]
        for wid in range(1,width+1):
            for hei in range(1,height+1):
                i=0
                while(i+wid<width):
                    j=0
                    while(j+hei<height):
                        immediate=Rectangle(i,j,wid,hei)
                        right=Rectangle(i+wid,j,wid,hei)
                        if(i+2*wid<width):
                            features.append(([right],[immediate]))
                        bottom=Rectangle(i,j+hei,wid,hei)
                        if(j+2*hei<height):
                            features.append(([immediate],[bottom]))
                        right_2=Rectangle(i+2*wid,j,wid,hei)
                        if(i+3*wid<width):
                            features.append(([right],[right_2,immediate]))
                        bottom_2=Rectangle(i,j+2*hei,wid,hei)
                        if(j+3*hei<height):
                            features.append(([bottom],[bottom_2,immediate]))
                        bottom_right=Rectangle(i+wid,j+hei,wid,hei)
                        if(i+2*wid<width and j+2*hei<height):
                            features.append(([right,bottom],[immediate,bottom_right]))
                        j+=1
                    i+=1
        return features
    def feature_apply(self,features,data):
        X=np.zeros((len(features),len(data)))
        Y=np.array(map(lambda data: data[1],data))
        i=0
        for positive,negative in features:
            feature=lambda ii: sum([pos.comp_feature(ii) for pos in positive]-sum([neg.comp_feature(ii) for neg in negative]))
            X[i]=list(map(lambda data: feature(data[0]),data))
            i+=1
        return X,Y
    def classify(self,image):
        total=0
        ii=integral(image)
        for alpha,clf in zip(self.alphas,self.clfs):
            total+=alpha*clf.classify(ii)
        return 1 if total>=0.5* sum(self.alphas) else 0
    def save(self,filename):
        with open(filename+".pkl",'wb') as f:
            pickle.dump(self,f)
    
    @staticmethod
    def load(filename):
        with open(filename+".pkl",'rb') as f:
            return pickle.load(f)
                        

def train(t):
    with open("training.pkl",'rb') as f:
        training=pickle.load(f)
    clf=ViolaJones(T=t)
    clf.train(training,2429,4548)
    evaluate(clf,training)
    clf.save(str(t))
def test(filename):
    with open("test.pkl",'rb') as f:
        test=pickle.load(f)
    clf=ViolaJones.load(filename)
    evaluate(clf,test)

def evaluate(clf,data):
    correct=0
    for x,y in data:
        correct+=1 if clf.classify(x) ==y else 0
    print("classified %d out of %d test examples" %(correct,len(data)))

train(10)
test("10")