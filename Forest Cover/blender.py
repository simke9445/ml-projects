import fscore_plot as fscore
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

def blender(X,y):

    # splitting our data set into training and test sets for further validation

    random_state = np.random.RandomState(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)



    # our level 0 classifiers

    clfs = [
            RandomForestClassifier(n_estimators = 100, criterion = 'gini',n_jobs=-1),
            ExtraTreesClassifier(n_estimators = 100 * 2, criterion = 'gini',n_jobs=-1),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=200),
            RandomForestClassifier(n_estimators=100, criterion='entropy',n_jobs=-1),
            ExtraTreesClassifier(n_estimators=100*2, criterion='entropy',n_jobs=-1)
        ]


    # Ready for cross validation

    n_folds = 5
    skf = list(StratifiedKFold(y_train, n_folds))


    blend_train = np.zeros((X_train.shape[0], len(clfs))) # Number of training data x Number of classifiers
    blend_test = np.zeros((X_test.shape[0], len(clfs))) # Number of testing data x Number of classifiers

    print ('X_test.shape = %s' % (str(X_test.shape)))
    print ('blend_train.shape = %s' % (str(blend_train.shape)))
    print ('blend_test.shape = %s' % (str(blend_test.shape)))


    # for each classfier, we train the number of fold times(= len(skf)) 
    for j, clf in enumerate(clfs):

        print ('Training classifier [%s]' % (j))
        blend_test_j = np.zeros((X_test.shape[0], len(skf))) # Number of testing data x Number of folds ,
                                                             # we will take the mean of the predictions later
        for i, (train_index, cv_index) in enumerate(skf):

            print ('Fold [%s]' % (i))
            # This is the training and validation set
            X_blend_train = X_train[train_index]
            y_blend_train = y_train[train_index]
            X_blend_cv = X_train[cv_index]
            y_blend_cv = y_train[cv_index]
            
            clf.fit(X_blend_train, y_blend_train)
            # This output will be the basis for our blended classifier to train against,
            # which is also the output of our classifiers
            blend_train[cv_index, j] = clf.predict(X_blend_cv)
            blend_test_j[:, i] = clf.predict(X_test)

        # Take the mean of the predictions of the cross validation set
        blend_test[:, j] = blend_test_j.mean(1)


    # blending the results using logistic regression
    bclf = LogisticRegression()
    bclf.fit(blend_train, y_train)

    # prediction
    y_test_predict = bclf.predict(blend_test)
    score = metrics.accuracy_score(y_test, y_test_predict)
    print ('Accuracy = %s' % (score))

    return score