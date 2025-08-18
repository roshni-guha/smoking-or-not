
import warnings
warnings.filterwarnings(action="ignore")

import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_column', None)

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
import time


data = pd.read_csv("smoking.csv", index_col = False)
data['gender'] = data['gender'].apply(lambda x: 1 if x == 'M' else 0)
data['oral'] = data['oral'].apply(lambda x: 1 if x == 'Y' else 0)
data['tartar'] = data['tartar'].apply(lambda x: 1 if x == 'Y' else 0)

print("\nSample Data:\n", data.head())
print("\nShape of dataset:", data.shape)
print("\nData types:\n", data.dtypes)
print("\nMissing values:\n", data.isnull().sum())

print("\nData description:\n", data.describe())

print("\nTarget distribution:\n", data['smoking'].value_counts())


plt.hist(data['smoking'])
plt.title("Diagnosis: Smoking (0=No, 1=Yes)")
#plt.show()

data = data.set_index('ID')
#print("After id", data)

data.plot(kind = 'density', subplots=True, layout=(5,6), sharex= False, legend = False, fontsize= 1)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
cax = ax1.imshow(data.corr(), interpolation=None)
ax1.grid(True)
plt.title('Smoking correlation')
fig.colorbar(cax, ticks = [.75,.80,.85,.90,.95,1])
plt.show()

Y = data['smoking'].values
X = data.drop(['smoking'], axis=1).values

X_train = pd.read_csv("X_train.csv", index_col = False)
X_test = pd.read_csv("X_test.csv", index_col = False)
Y_train = pd.read_csv("Y_train.csv", index_col = False)
Y_test = pd.read_csv("Y_test.csv", index_col = False)

X_train['gender'] = X_train['gender'].apply(lambda x: 1 if x == 'M' else 0)
X_train['oral'] = X_train['oral'].apply(lambda x: 1 if x == 'Y' else 0)
X_train['tartar'] = X_train['tartar'].apply(lambda x: 1 if x == 'Y' else 0)

X_test['gender'] = X_test['gender'].apply(lambda x: 1 if x == 'M' else 0)
X_test['oral'] = X_test['oral'].apply(lambda x: 1 if x == 'Y' else 0)
X_test['tartar'] = X_test['tartar'].apply(lambda x: 1 if x == 'Y' else 0)

X_train = X_train.set_index('ID')
X_test = X_test.set_index('ID')
Y_train = Y_train.set_index('ID')
Y_test = Y_test.set_index('ID')

models_list = []
models_list.append(('CART', DecisionTreeClassifier()))
models_list.append(('KNN', KNeighborsClassifier()))
models_list.append(('NB', GaussianNB()))
models_list.append(('SVM', SVC()))

num_folds = 10

names = []
results = []

for name, model in models_list:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=123)
    start_Time = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    end_Time = time.time()
    results.append(cv_results)
    names.append(name)
    print( "%-10s: %10f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end_Time-start_Time))

fig = plt.figure()
fig.suptitle('Performance Comparison')

ax = fig.add_subplot(111)
plt.boxplot(results, labels=names)
ax.set_xticklabels(names)
plt.show()

# CART      :   0.686688 (0.008834) (run time: 3.238230)
# KNN       :   0.688349 (0.007592) (run time: 1.702321)
# NB        :   0.703050 (0.005133) (run time: 0.101225)
# SVM       :   0.729513 (0.008456) (run time: 203.551013)

# SVM has the best performance but takes too long. 

pipelines = []
pipelines.append(('ScaledCART', Pipeline([ ('Scaler', StandardScaler()),('CART', DecisionTreeClassifier()) ] )))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC( ))])))

results = []
names = []

print("\n\n\nAccuracies of algorithm after scaled dataset\n")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=123)
for name, model in pipelines:
    start = time.time()
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    end = time.time()
    results.append(cv_results)
    names.append(name)
    print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))

# ScaledCART: 0.685453 (0.006000) (run time: 5.782933)
# ScaledNB: 0.703050 (0.005133) (run time: 0.228313)
# ScaledKNN: 0.711332 (0.005164) (run time: 3.399760)
# ScaledSVM: 0.758063 (0.004555) (run time: 376.210853)

fig = plt.figure()
fig.suptitle('Performance Comparison after Scaled Data')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

model = SVC()
start = time.time()
model.fit(X_train_scaled, Y_train) #Training of algorithm using 67% of dataend 
end = time.time()
print( "\n\nSVM Training Completed. It's Run Time: %f" % (end-start))

#Run Time: 40.355459
# estimate accuracy on test dataset
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)
print("All predictions done successfully by SVM Machine Learning Algorithms")
print("\n\nAccuracy score %f" % accuracy_score(Y_test, predictions))

print("\n\n")
print("confusion_matrix = \n")
print( confusion_matrix(Y_test, predictions))