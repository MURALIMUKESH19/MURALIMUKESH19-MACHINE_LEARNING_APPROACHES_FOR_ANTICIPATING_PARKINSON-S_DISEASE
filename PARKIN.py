#!/usr/bin/env python 
"""Django's command-line utility for administrative tasks.""" 
import os 
import sys 
def main(): 
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'self.settings') 
try: 
from django.core.management import execute_from_command_line 
except ImportError as exc: 
raise ImportError( 
"Couldn't import Django. Are you sure it's installed and " 
"available on your PYTHONPATH environment variable? Did you " 
"forget to activate a virtual environment?" 
) from exc 
execute_from_command_line(sys.argv) 
if    name == ' main ': 
main() 
{% load static %} 
<!DOCTYPE html> 
<html lang="en"> 
32 
<head> 
<title>PARKINSONS DISEASE detection</title> 
<meta charset="UTF-8"> 
<meta name="viewport" content="width=device-width, initial-scale=1"> 
<!-- 
============================================================= 
==================================--> 
<link rel="icon" type="image/png" href="{% static 
'mages/icons/favicon.ico' %}"/> 
<!-- 
============================================================= 
==================================--> 
<link rel="stylesheet" type="text/css" href="{% static 
'vendor/bootstrap/css/bootstrap.min.css' %}"> 
<!-- 
============================================================= 
==================================--> 
<link rel="stylesheet" type="text/css" href="{% static 'fonts/font-awesome- 
4.7.0/css/font-awesome.min.css' %}"> 
<!-- 
============================================================= 
==================================--> 
<link rel="stylesheet" type="text/css" href="{% static 'fonts/Linearicons-Free- 
v1.0.0/icon-font.min.css' %}"> 
<!-- 
============================================================= 
==================================--> 
<link rel="stylesheet" type="text/css" href="{% static 
'vendor/animate/animate.css' %}"> 
<!-- 
============================================================= 
==================================--> 
<link rel="stylesheet" type="text/css" href="{% static 'vendor/css- 
hamburgers/hamburgers.min.css' %}"> 
33 
<!-- 
============================================================= 
==================================--> 
<link rel="stylesheet" type="text/css" href="{% static 
'vendor/animsition/css/animsition.min.css' %}"> 
<!-- 
============================================================= 
==================================--> 
<link rel="stylesheet" type="text/css" href="{% static 
'vendor/select2/select2.min.css' %}"> 
<!-- 
============================================================= 
==================================--> 
<link rel="stylesheet" type="text/css" href="{% static 
'vendor/daterangepicker/daterangepicker.css' %}"> 
<!-- 
============================================================= 
==================================--> 
<link rel="stylesheet" type="text/css" href="{% static 'css/util.css' %}"> 
<link rel="stylesheet" type="text/css" href="{% static 'css/main.css' %}"> 
<!-- 
============================================================= 
==================================--> 
</head> 
<body style="background-color: #0af350;"> 
<div class="limiter"> 
<div class="container-login100"> 
<div class="wrap-login100"> 
<form action='input' class="login100-form validate-form"> 
<span class="login100-form-title p-b-43"> 
Login to continue 
34 
</span> 
<div class="wrap-input100 validate-input" data-validate = "Valid email is 
required: ex@abc.xyz"> 
<input class="input100" type="text" name="AGE"> 
<span class="focus-input100"></span> 
<span class="label-input100">AGE</span> 
</div> 
<div class="wrap-input100 validate-input" data-validate="Password is 
required"> 
<input class="input100" type="text" name="pass"> 
<span class="focus-input100"></span> 
<span class="label-input100">GENDER</span> 
</div> 
<div class="flex-sb-m w-full p-t-3 p-b-32"> 
<div class="contact100-form-checkbox"> 
<input class="input-checkbox100" id="ckb1" type="checkbox" 
name="remember-me"> 
<label class="label-checkbox100" for="ckb1"> 
Jitter:DDP 
</label> 
</div> 
<div> 
<a href="#" class="txt1"> 
PPE 
</a> 
</div> 
</div> 
35 
<div class="container-login100-form-btn"> 
<button class="login100-form-btn"> 
Analyze 
</button> 
</div> 
<div class="text-center p-t-46 p-b-20"> 
<span class="txt2"> or sign up using 
</span> 
</div> 
<div class="login100-form-social flex-c-m"> 
<a href="#" class="login100-form-social-item flex-c-m bg1 m-r-5"> 
<i class="fa fa-facebook-f" aria-hidden="true"></i> 
</a> 
<a href="#" class="login100-form-social-item flex-c-m bg2 m-r-5"> 
<i class="fa fa-twitter" aria-hidden="true"></i> 
</a> 
</div> 
</form> 
<!-- <div class="login100-more" style="background-image: url({% static 
'images/123.mp4' %});"> 
</div> --> 
<video style="background-color: black; height: 670px; width: 1020px;" autoplay 
muted loop > 
<source src="/static/images/123.webm" type="video/mp4" > 
</video> 
</div> 
</div> 
36 
</div> 
<!-- 
============================================================= 
==================================--> 
<script src="{% static 'vendor/jquery/jquery-3.2.1.min.js' %}"></script> 
<!-- 
============================================================= 
==================================--> 
<script src="{% static 'vendor/animsition/js/animsition.min.js' %}"></script> 
<!-- 
============================================================= 
==================================--> 
<script src="{% static 'vendor/bootstrap/js/popper.js' %}"></script> 
<script src="{% static 'vendor/bootstrap/js/bootstrap.min.js' %}"></script> 
<!-- 
============================================================= 
==================================--> 
<script src="{% static 'vendor/select2/select2.min.js' %}"></script> 
<!-- 
============================================================= 
==================================--> 
<script src="{% static 'vendor/daterangepicker/moment.min.js' %}"></script> 
<scriptsrc="{%static'vendor/daterangepicker/daterangepicker.js' %}"></script> 
<!========================================================== 
=====================================--> 
<script src="{% static 'vendor/countdowntime/countdowntime.js' %}"></script> 
<!-- 
============================================================= 
==================================--> 
<script src="{% static 'js/main.js' %}"></script> 
</body> 
</html> 
37 
5.1.2 BACKEND CODE 
import pandas as pd 
#list of useful imports that I will use 
%matplotlib inline 
import os 
import matplotlib.pyplot as plt 
import pandas as pd 
import cv2 
import numpy as np 
from glob import glob 
import seaborn as sns 
import random 
import pickle 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_curve 
data = pd.read_csv() 
data 
data.columns 
data.info() 
data.describe() 
data.isnull().sum() 
data.isnull().any() 
data.drop('name',axis=1,inplace=True) 
data.corr() 
data['status'].value_counts() 
38 
sns.set(style="whitegrid") 
plt.figure(figsize=(10, 5)) 
ax = sns.countplot(x="status", data=data, palette=sns.color_palette("cubehelix", 4)) 
plt.xticks(rotation=90) 
plt.title("Class 
Label 
"fontsize":"medium"}) 
Counts", 
{"fontname":"fantasy", 
plt.ylabel("count", {"fontname": "serif", "fontweight":"bold"}) 
plt.xlabel("Class", {"fontname": "serif", "fontweight":"bold"}) 
from sklearn.utils import resample 
# Separate majority and minority classes 
df_majority = data[data['status']== 1] 
df_minority = data[data['status']== 0] 
# Downsample majority class and upsample the minority class 
df_minority_upsampled=resample(df_minority, 
replace=True,n_samples=1000,random_state=100) 
df_majority_downsampled=resample(df_majority, 
replace=True,n_samples=1000,random_state=100) 
# Combine minority class with downsampled majority class 
"fontweight":"bold", 
df_balanced = pd.concat([df_minority_upsampled, df_majority_downsampled]) 
# Display new class counts 
df_balanced['status'].value_counts() 
sns.countplot(df_balanced[['status']]) 
plt.grid() 
plt.legend() 
plt.title(' 0 :not affected & 1 : affected ') 
39 
plt.show() 
print(' ') 
plt.pie([1000,1000],labels=['not affected','affect'],autopct='%.2f%%') 
plt.legend(loc=(1,0.5)) 
plt.title(' 0 :not affected & 1 : affect ') 
plt.show() 
data= df_balanced.sample(frac = 1) 
data 
data.isnull().sum() 
data.dropna(inplace=True) 
data 
x = data.loc[:, data.columns != 'status'] 
x 
y = data.iloc[:,-7] 
y 
x.head() 
y.tail() 
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,stratify=y,random_sta 
te=40) 
x_train 
y_test 
x_test.to_csv('Parkinsons_test.csv',index = False) 
40 
from sklearn.svm import SVC 
from sklearn.calibration import CalibratedClassifierCV 
import math 
from sklearn.metrics import accuracy_score 
C = [10000,1000,100,10,1,0.1,0.01,0.001,0.0001] 
train_auc = [] 
cv_auc = [] 
for i in C: 
model = SVC(C=i,gamma=50) 
clf = CalibratedClassifierCV(model, cv=3) 
clf.fit(x_train,y_train) 
prob_cv = clf.predict(x_test) 
cv_auc.append(accuracy_score(y_test,prob_cv)) 
prob_train = clf.predict(x_train) 
train_auc.append(accuracy_score(y_train,prob_train)) 
optimal_C= C[cv_auc.index(max(cv_auc))] 
C=[math.log(x) for x in C] 
#plot auc vs alpha 
x = plt.subplot( ) 
x.plot(C, train_auc, label='AUC train') 
x.plot(C, cv_auc, label='AUC CV') 
plt.title('AUC vs hyperparameter') 
plt.xlabel('C') 
41 
plt.ylabel('AUC') 
x.legend() 
plt.show() 
print('optimal C for which auc is maximum : ',optimal_C) 
gamma = [10,20,30,30,40] 
train_auc = [] 
cv_auc = [] 
for i in gamma: 
model = SVC(C=1000,gamma=i) 
clf = CalibratedClassifierCV(model, cv=3) 
clf.fit(x_train,y_train) 
prob_cv = clf.predict(x_test) 
cv_auc.append(accuracy_score(y_test,prob_cv)) 
prob_train = clf.predict(x_train) 
train_auc.append(accuracy_score(y_train,prob_train)) 
optimal_gamma= gamma[cv_auc.index(max(cv_auc))] 
# C=[math.log(x) for x in C] 
#plot auc vs alpha 
x = plt.subplot( ) 
x.plot(gamma, train_auc, label='AUC train') 
x.plot(gamma, cv_auc, label='AUC CV') 
plt.title('AUC vs hyperparameter') 
plt.xlabel('gamma') 
plt.ylabel('AUC') 
42 
x.legend() 
plt.show() 
print('optimal gamma for which auc is maximum : ',optimal_gamma) 
#Testing AUC on Test data 
svc = SVC(C=optimal_C,gamma=optimal_gamma) 
svc.fit(x_train,y_train) 
filename=r'C:\Users\RAJAKANNAN\Music\PARKINSON\CODING\frontend\svc_p 
ark.pkl' 
pickle.dump(svc, open(filename, 'wb')) 
#predict on test data and train data 
y_predtests = svc.predict(x_test) 
y_predtrains = svc.predict(x_train) 
print('*'*35) 
#accuracy on training and testing data 
print('the accuracy on testing data',accuracy_score(y_test,y_predtests)) 
print('the accuracy on training data',accuracy_score(y_train,y_predtrains)) 
train = accuracy_score(y_train,y_predtrains) 
test = accuracy_score(y_test,y_predtests) 
print('*'*35) 
# Code for drawing seaborn heatmaps 
class_names = ['not affcted','affect'] 
cm=pd.DataFrame(confusion_matrix(y_test,y_predtests.round()), index=class_names, 
columns=class_names ) 
fig = plt.figure( ) 
heatmap = sns.heatmap(cm, annot=True, fmt="d") 
43 
original = ['affected' if x==1 else 'not affected' for x in y_test[:20]] 
predicted = svc.predict(x_test[:20]) 
pred = [] 
for i in predicted: 
if i == 1: 
k = 'affected' 
pred.append(k) 
else: 
k = 'not affected' 
pred.append(k) 
# Creating a data frame 
dfr = pd.DataFrame(list(zip(original, pred,)), 
columns =['original_Classlabel', 'predicted_classlebel']) 
dfr 
all_model_result=pd.DataFrame(columns=['Classifier', 'Train-Accuracy', 'Test- 
Accuracy' ]) 
new = ['SUPPORT VECTOR-Classifier',train, test] 
all_model_result.loc[0] = new 
import xgboost as xgb 
from xgboost import XGBClassifier 
xgb = XGBClassifier() 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score,confusion_matrix 
from sklearn.model_selection import GridSearchCV 
dept = [1, 5, 10, 50, 100, 500, 1000] 
n_estimators = [20, 40, 60, 80, 100, 120] 
44 
xgb = XGBClassifier() 
param_grid={'n_estimators':n_estimators , 'max_depth':dept} 
# clf = RandomForestClassifier() 
model = GridSearchCV(xgb,param_grid,scoring='accuracy',n_jobs=-1,cv=3) 
model.fit(x_train,y_train) 
print("optimal n_estimators",model.best_estimator_.n_estimators) 
print("optimal max_depth",model.best_estimator_.max_depth) 
optimal_n_estimators = model.best_estimator_.n_estimators 
optimal_max_depth = model.best_estimator_.max_depth 
#Testing AUC on Test data 
xgb.fit(x_train,y_train) 
filename=r'C:\Users\RAJAKANNAN\Music\PARKINSON\CODING\frontend\xgb_p 
ark.pkl' 
pickle.dump(xgb, open(filename, 'wb')) 
#predict on test data and train data 
y_predtest = xgb.predict(x_test) 
y_predtrain = xgb.predict(x_train) 
print('*'*35) 
#accuracy on training and testing data 
print('the accuracy on testing data',accuracy_score(y_test,y_predtest)) 
print('the accuracy on training data',accuracy_score(y_train,y_predtrain)) 
train2 = accuracy_score(y_train,y_predtrain) 
test2 = accuracy_score(y_test,y_predtest) 
print('*'*35) 
45 
# Code for drawing seaborn heatmaps 
class_names = ['not affected','affected'] 
cm = pd.DataFrame(confusion_matrix(y_test, y_predtest.round()), 
index=class_names, columns=class_names ) 
fig = plt.figure( ) 
heatmap = sns.heatmap(cm, annot=True, fmt="d") 
original = ['affected' if x==1 else 'not affected' for x in y_test[:20]] 
predicted = xgb.predict(x_test[:20]) 
pred = [] 
for i in predicted: 
if i == 1: 
k = 'affected' 
pred.append(k) 
else: 
k = 'not affected' 
pred.append(k) 
# Creating a data frame 
dfr = pd.DataFrame(list(zip(original, pred,)), 
columns =['original_Classlabel', 'predicted_classlebel']) 
dfr 
new = ['XGB-Classifier',train2, test2] 
all_model_result.loc[1] = new 
all_model_result