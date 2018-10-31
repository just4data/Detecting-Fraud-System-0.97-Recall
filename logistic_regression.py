import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, cohen_kappa_score

data = pd.read_csv('creditcard.csv')

X = data.ix[:, 1:29]
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

lr = LogisticRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
print conf_mat

TN = conf_mat[0][0]
FP = conf_mat[0][1]
FN = conf_mat[1][0]
TP = conf_mat[1][1]

accuracy = 100*float(TP+TN)/float(np.sum(conf_mat))
print 'Accuracy: ' + str(np.round(accuracy, 2)) + '%'

recall = 100*float(TP)/float(TP+FN)
print "Recall: " + str(np.round(recall, 2)) + '%'

cohen = cohen_kappa_score(y_test, y_pred)
print 'Cohen Kappa: ' + str(np.round(cohen, 3))