
import warnings

from sklearn.calibration import LabelEncoder
warnings.filterwarnings(action="ignore")

import pandas as pd

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

# ======================================
# 1. Load Dataset
# ======================================

data = pd.read_csv("smoking.csv", index_col=False)

print("\nSample Data:\n", data.head())
print("\nShape of dataset:", data.shape)
print("\nData types:\n", data.dtypes)
print("\nMissing values:\n", data.isnull().sum())

# ======================================
# 2. Quick Data Understanding
# ======================================
print("\nTarget distribution:\n", data['smoking'].value_counts())

plt.figure(figsize=(5,3))
plt.bar(data['smoking'].value_counts().index, data['smoking'].value_counts().values, color = ['red', 'green'])
plt.xlabel('smoking')
plt.ylabel('count')
plt.title("Target Distribution: Smoking (0=No, 1=Yes)")
plt.show()

# ======================================
# 3. Preprocessing
# ======================================

data.set_index('id', inplace=True)

# Encode categorical features if needed
cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

print("\nCategorical features encoded:\n", cat_cols)

