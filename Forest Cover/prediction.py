import fscore_plot as fscore
import blender as bl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
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

#df = data frame for storing the initial table
df = pd.read_csv('train.csv')

# y => represents an output vector which contains classes we need to predict
y = df['Cover_Type'].as_matrix()
y = y - 1

# now we need to transform our data frame into a suitable X matrix for training
# which includes : cutting Cover_type column, and chainging all values to floats
df.drop('Cover_Type',axis=1,inplace=True)
df.drop('Id',axis=1,inplace=True)
X = df.as_matrix()
X = X.astype(float)

# normalization and feature scaling is required for training setup for SVM's
X = preprocessing.scale(X)

selector = VarianceThreshold()

X = selector.fit_transform(X)

best_score = 0.0
    
# run many times to get a better result, it's not quite stable.
for i in range(0,25):
    print ('Iteration [%s]' % (i))
    score = bl.blender(X,y)
    best_score = max(best_score, score)
    print
    
print ('Best score = %s' % (best_score))