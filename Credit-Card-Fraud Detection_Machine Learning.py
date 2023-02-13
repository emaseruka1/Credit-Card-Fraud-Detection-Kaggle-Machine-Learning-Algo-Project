# -*- coding: utf-8 -*-

"""
Created on Sun Jul 24 11:58:27 2022
@author: Emmanuel Maseruka
"""
#downloading relevant packages
from sklearn.tree import DecisionTreeClassifier ##used to clasify data
from sklearn.model_selection import train_test_split ##used to split data into train & test set
import joblib # used to store tested model
from sklearn import tree # visualise model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import smtplib 
pw = "PW"
data=pd.read_csv('C:/Users/Emmanuel/OneDrive/Desktop/machine learning project/original.csv') #pulling my dataset
data=data.drop(columns=["index"]) #dropping the index column



#NEAREST NEIGHBOUR

from sklearn.neighbors import KNeighborsClassifier #importing KNN algo


#subsetting relevant columns (Only Numeric Variables)

data1=data.drop(columns=["repeat_retailer","used_chip","used_pin_number","online_order"]) 
x1 = data1.drop(columns=['fraud']) 
y1 = data1['fraud']


#splitting the data into test and train sets
x1_train,x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.4, random_state=42) 
knn = KNeighborsClassifier(n_neighbors=115, metric='euclidean') #running 115 KNN
knn.fit(x1_train, y1_train) #training the model
y_pred = knn.predict(x1_test) #testing the algorithmn
fn1=["distance_from_home","distance_from_last_transaction","ratio_to_median_purchase_price"]
cn=["No","Yes"]
print(classification_report(y1_test,y_pred, target_names=cn)) #print classification report
matrix = confusion_matrix(y1_test,y_pred)
print('Confusion matrix : \n',matrix) #print confusion matrix
cmd_obj=ConfusionMatrixDisplay(matrix, display_labels=cn)
cmd_obj.plot()
joblib.dump(knn, 'credit-card-knn.joblib') #saving my alogorithm



#DECISION TREE

xm = data.drop(columns=['fraud']) #input dataset
ym = data['fraud'] #output dataset
x_train, x_test, y_train, y_test = train_test_split(xm, ym, test_size=0.4, shuffle=True) # Split into train and test set
#running Decision tree model with specificed parameters
model = DecisionTreeClassifier(max_leaf_nodes=10, max_features = 2, random_state=42,class_weight=
model.fit(x_train,y_train) #training the model
predictions = model.predict(x_test) #testing the algorithmn
fn=["distance_from_home","distance_from_last_transaction","ratio_to_median_purchase_price","repeat_retailer"
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=400)
fig=tree.plot_tree(model,feature_names = fn, class_names=cn, filled = True)
print(classification_report(y_test,predictions, target_names=cn)) #print classification report
matrix = confusion_matrix(y_test,predictions, labels = None) #print confusion matrix
cmd_obj=ConfusionMatrixDisplay(matrix, display_labels=cn)
cmd_obj.plot()
print('Confusion matrix : \n',matrix)



joblib.dump(model, 'CCF.joblib') ##storing our model


# Implementation of a Stacked Heterogeneous Ensemble


print("*Powered by Emma-Tech System* : Validating transaction............")
my_email = input("What is your email: ")
print("*Powered by Emma-Tech System* : Validating transaction............")
dist_from_home = input("How far is home? ")
dist_from_last_trans = input("When was the last transaction? ")
ratio_to_median_purchase_price = input("How much are you spending? ")
dist_from_home = float(dist_from_home)
dist_from_last_trans = float(dist_from_last_trans)
ratio_to_median_purchase_price = float(ratio_to_median_purchase_price)
2
knn = joblib.load('credit-card-knn.joblib') 
predictions_from_knn = knn.predict([[dist_from_home,dist_from_last_trans,ratio_to_median_purchase_price]])
print(predictions_from_knn)
if predictions_from_knn=="Yes":
 
 print("Please enter either 0 or 1\n0 = No, 1 = Yes")
 
 repeat_retailer = input ("Is this a repeat retailer? ")
 
 used_chip = input ("Is a chip being used? ")
 
 used_pin_number = input ("Is a pin being used? ")
 
 online_order = input ("Is it an online transaction? ")
 
 repeat_retailer = int(repeat_retailer)
 used_chip = int(used_chip)
 used_pin_number = int(used_pin_number)
 
 online_order = int(online_order)
 
 
 model = joblib.load('CCF.joblib') 
 predictions_from_J48 = model.predict([[dist_from_home,dist_from_last_trans,ratio_to_median_purchase_price,repeat_retailer,used_chip,used_pin_number,online_order]])
 
 if predictions_from_J48 =="Yes":
 server = smtplib.SMTP('smtp.gmail.com',587)
 server.starttls()
 server.login('b1810079@gl.aiu.ac.jp',pw)
 server.sendmail("b1810079@gl.aiu.ac.jp",my_email,'Dear Customer, There is suspicious activity on your credit card. We will halt the transaction pending further verification of this case. We sincerely apologize for the inconvenience'
 
 
 else: print("Thank you for using our Card Service \n\n\n*Powered by Emma-Tech*")
 
else: print("Thank you for using our Card Service \n*Powered by Emma-Tech*")
exit()