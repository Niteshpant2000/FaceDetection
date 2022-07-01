import pickle
name={0:'Name',1:'Nitesh',2:'ISHA'}
with open('name.pkl','wb') as f:
    pickle.dump(name,f)
pic=open('name.pkl',"rb")
print(pickle.load(pic))