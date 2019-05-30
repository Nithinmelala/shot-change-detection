import numpy as np
import cv2
import os

vidcap = cv2.VideoCapture('news.mpg')
success,image = vidcap.read()

no_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Tottal_no_soccer_frames :',no_frames )
dimns = image.shape
print('dimension:',dimns)
      
print("Please wait program is executing")
count = 0
success = True
while success:
  success,image = vidcap.read()
  Folder = '\\news frames'
  path1 = os.getcwd()
  path = str(path1 + Folder )
  cv2.imwrite(os.path.join(path, "frame%d.jpg" % count,),image)
  count += 1
  # save frame as JPEG file
  cv2.waitKey(10)

print("All the images all storded in the folder news frames")

###########################################################################
cv2.destroyAllWindows()


data1 = cv2.imread(str(path + '//Frame73.jpg'))
cv2.imshow('frame73',data1)

data2 = cv2.imread(str(path + '//Frame235.jpg'))
cv2.imshow('Frame235',data2)

data3 = cv2.imread(str(path + '//Frame301.jpg'))
cv2.imshow('Frame301',data3)

data4 = cv2.imread(str(path + '//Frame370.jpg'))
cv2.imshow('Frame370',data4)

data5 = cv2.imread(str(path + '//Frame452.jpg'))
cv2.imshow('Frame452',data5)

data6 = cv2.imread(str(path + '//Frame861.jpg'))
cv2.imshow('Frame861',data6)

data7 = cv2.imread(str(path + '//Frame1281.jpg'))
cv2.imshow('Frame1281',data7)
############################################################################
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

dimention = [dimns[0],dimns[1]]
#Load the soccer data
# generate 2 class dataset
X, y = make_classification(n_samples=no_frames, n_classes=2,  weights= dimention)

#slipt the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=44)

#Model
clf = LogisticRegression(penalty='l2', C=0.1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#Accurecy
print("Accuracy", metrics.accuracy_score(y_test, y_pred))

#AUC Curve
y_pred_proba = clf.predict_proba(X_test)[::,0]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, roc="+str(auc))
plt.legend(loc=4)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('PERFORMANCE INDICATOR using PRECISION-RECALL CURVE- ')
plt.show()

