import blender as bl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import pickle
import fscore_plot as fscore
import learning_curve_plot as learning_curve
import make_features as mk_ft
from sklearn import preprocessing
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif, RFE
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn import cross_validation



# This is a hit at the Forest Cover problem on kaggle.com, which achieved 163 place(estimated, after deadline) out of 1700
# I've tryed many classifiers like Logistic Regression, Support Vector Machines, Random Decision Forests, Adaptive Boosting
# and Extra trees classifier which gave me the best cross-validated accuracy

# df = data frame for storing the initial table
df = pd.read_csv('train.csv')

# y => represents an output vector which contains classes we need to predict
# y = y - 1, because initially classes are represented as 1-7
y = df['Cover_Type'].as_matrix()
y = y - 1


# column names
names = df.columns.values.tolist()

# now we create new features through combination of existing ones, which are pre-selected through exploratory analysis
# and wild guesses(which worked! :P)
df = mk_ft.make_features(df)

# drop unecessary, redundant columns, and create a matrix from a data frame
df.drop('Cover_Type',axis=1,inplace=True)
df.drop('Id',axis=1,inplace=True)
X = df.as_matrix()
X = X.astype(float)

# train classifier
clf = ExtraTreesClassifier(n_estimators=750)
clf.fit(X, y)

# read test samples
df_test = pd.read_csv("test.csv")


# no need for them anymore
del df
del X
del y

#final represents the data frame which we'll use for outputing results
df_final = pd.DataFrame(columns=['Id', 'Cover_Type'], dtype=int)
df_final['Id'] = df_test['Id']

# make new features for the test set also
df_test.drop('Id',axis=1,inplace=True)
df_test = mk_ft.make_features(df_test)

X_submission = df_test.as_matrix().astype(float)

del df_test


# predicting the test set values for Forest Cover Type
df_final["Cover_Type"] = pd.Series(clf.predict(X_submission) + 1)


# now we write our resulsts to a file for submission
df_final.to_csv('submission.csv')

del X_submission
del df_final
del clf
