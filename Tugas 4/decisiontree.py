#decisiontree
from pyexpat import model
import numpy as np
import pandas as pd
from sklearn import tree

irisDataset = pd.read_csv('E:\Dataset_Iris.csv',delimiter=',',header=0)

irisDataset["species"] = pd.factorize(irisDataset.Species)[0]

irisDataset = irisDataset.drop(labels="id",axis=1)

#irisDataset = irisDataset.as_matrix()
irisDataset = irisDataset.to_numpy()

dataTraining = np.concatenate((irisDataset[0:40,:],irisDataset[50:90,:]),axis=0)
dataTesting = np.concatenate((irisDataset[40:50,:],irisDataset[90:100,:]), axis=0)

inputTraining = dataTraining[:,0:4]
inputTesting = dataTesting[:,0:4]
labelTraining = dataTraining[:,4]
labelTesting = dataTesting[:,4]

model = tree.DecisionTreeClassifier()
model = model.fit(inputTraining, labelTraining)

hasilPrediksi = model.predict(inputTesting)
print("label sebenarnya ", labelTesting)
print("Hasil Prediksi: ", hasilPrediksi)

prediksiBenar = (hasilPrediksi == labelTesting).sum()
prediksiSalah = (hasilPrediksi != labelTesting).sum()
print("prediksi benar : ", prediksiBenar, " data")
print("prediksi salah : ", prediksiSalah, " data")
print("akurasi : ",prediksiBenar/(prediksiBenar+prediksiSalah)* 100, "%")