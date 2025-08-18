
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

# print("\nSample Data:\n", data.head())
# print("\nShape of dataset:", data.shape)
# print("\nData types:\n", data.dtypes)
# print("\nMissing values:\n", data.isnull().sum())

# print("\nData description:\n", data.describe())

# print("\nTarget distribution:\n", data['smoking'].value_counts())


# plt.hist(data['smoking'])
# plt.title("Diagnosis: Smoking (0=No, 1=Yes)")
# #plt.show()

# data = data.set_index('ID')
# #print("After id", data)

# data.plot(kind = 'density', subplots=True, layout=(5,6), sharex= False, legend = False, fontsize= 1)
# plt.show()

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# cax = ax1.imshow(data.corr(), interpolation=None)
# ax1.grid(True)
# plt.title('Smoking correlation')
# fig.colorbar(cax, ticks = [.75,.80,.85,.90,.95,1])
# plt.show()

Y = data['smoking'].values
X = data.drop(['smoking'], axis=1).values

