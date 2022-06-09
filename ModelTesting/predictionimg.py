import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import rospy

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score



flat_data_arr=[] #input image array
path='test_tennisball'    #file path
expectedoutput = 1 #state the expected output (i.e. index of the class)
for img in os.listdir(path):        
    img_array=imread(os.path.join(path,img))        
    img_resized=resize(img_array,(150,150,3))        
    flat_data_arr.append(img_resized.flatten()) 

#input image array after processing
x=np.array(flat_data_arr)

#load ML model
import joblib
# model = joblib.load("sim_model.pkl")  
# model = joblib.load("realdata3objModel.pkl")  
# model = joblib.load("realextdata3objModel.pkl")
# model = joblib.load("mixModel3obj.pkl")    
# model = joblib.load("mixedext50simdata3objModel.pkl")  
# model = joblib.load("mixedext100simdata3objModel.pkl")  
# model = joblib.load("mixedextdata3objModel.pkl")  
# model = joblib.load("mixedextsimdata3objModel.pkl")  
# model = joblib.load("realgrayscale3objModel.pkl")  
# model = joblib.load("greyscale_Model.pkl")  
model = joblib.load("mixedobj5Model.pkl")  


#predict outputs for input array
outputarr = model.predict(x)

#gives probability predicted for each class
outputprob = [model.predict_proba(x)]

#count occurences for each class
unique, counts = np.unique(outputarr, return_counts=True)
outputcount = dict(zip(unique, counts)) 
print(outputcount)

accuracy = (outputcount.get(expectedoutput)/len(outputarr))*100   #accuracy as percentage
print('Accuracy: '+ str(accuracy) + ' %')


#get probability of outputs
n = len(outputprob[0])
prob_arr = []

#get indices of outputarr when it detected the correct output
indices = [i for i, x in enumerate(outputarr) if x == expectedoutput]


for i in range(n):
    if i in indices:
        prob_arr += [outputprob[0][i][expectedoutput]]

#calculate mean and std of probability 
print('Mean: ' + str(100* np.mean(prob_arr)) + '%    Std: ' + str(100*np.std(prob_arr)))




