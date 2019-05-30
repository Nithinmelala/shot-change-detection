import cv2
import os
import numpy as np

cv2.destroyAllWindows()

################ Frame extraction ############################

vidcap = cv2.VideoCapture('soccer.mpg')
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
  Folder = '\\soccer frame'
  path1 = os.getcwd()
  path = str(path1 + Folder )
  cv2.imwrite(os.path.join(path, "frame%d.jpg" % count,),image)
  count += 1
  # save frame as JPEG file
  cv2.waitKey(10)

print("All the images all storded in the folder soccer frames")



#################### SHOT FRAME DETECTION  ####################

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def label(input, neighbors=None, background=None, return_num=False,
          connectivity=None):
    print("Hello World-Nithin")

print("Hello World-Nithin_NCCU")

path = os.getcwd()

Folder = '\\soccer frame'

##Imag_89 = cv2.imread('frame89.JPG')
Imag_89 = cv2.imread(str(path + Folder + '\\frame89.JPG'))

cv2.imshow('Imag_89',Imag_89)

A2 = Imag_89[:,:,2]
Orginal_gray_89 = A2

##cv2.imshow('Orginal_gray_89', A2)
################################# for image 89##################################

dimns = Imag_89.shape
nr = dimns[0]
nc = dimns[1]
nm = dimns[2]
print("NUmber of Rows:",nr )
print("NUmber of colums:", nc )
print("NUmber of Matrix:", nm )
threshold = 150 #55 70

for k1 in range(nr):
	for k2 in range(nc):

		if(A2[k1,k2]>=threshold):
			A2[k1,k2] = 255
		else:
			A2[k1,k2] = 0



#cv2.imshow('Image2', A2)
A2[1:99,:] = 0;
#cv2.imshow('Image3', A2)
A2[:,501:] = 0;
#cv2.imshow('Image4', A2)
kernel_1 = np.ones((2,2),np.uint8)
A2 = cv2.erode(A2,kernel_1,iterations = 1)
#cv2.imshow('Image5', A2)


kernel = np.ones((10,10),np.uint8)
A2 = cv2.morphologyEx(A2, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('Image6', A2)

A2[A2>=1] = 1
#cv2.imshow('Image7', A2)



for k1 in range(1,nr):
	for k2 in range(1,nc):

		A2[k1,k2] = A2[k1,k2]*Orginal_gray_89[k1,k2]


A5 = A2;

for j11 in range(1,nr):
	for j22 in range(1,nc):

		if(A5[j11,j22] >=1):
			A5[j11,j22] = 1
		else:
			A5[j11,j22] = 0


A6 = np.ones((nr,nc,nm),np.uint8)
A7 = np.ones((nr,nc,nm),np.uint8)

A6[:,:,0] = A5;
A6[:,:,1] = A5;
A6[:,:,2] = A5;

for k11 in range(1,nr):
	for k22 in range(1,nc):
		for k33 in range(1,nm):

			A7[k11,k22,k33] = Imag_89[k11,k22,k33]*A6[k11,k22,k33]


cv2.imshow('Template_89_Blob', A7)

## for number of persons identification 
A8 = A7>=250  # thresholds = 150,200
A9 = np.array(A8,dtype=np.double)
A10 = A9[:,:,1]
##cv2.imshow('A10_valid',A10)
A11 = np.array(A10,dtype = np.uint8)
ret_Frame_89, labels = cv2.connectedComponents(A11)
print('number of Persomns in Frame_89= ', ret_Frame_89)


##################################################   for image 92##########################3


##Imag_92 = cv2.imread('frame92.JPG')
Imag_92 = cv2.imread(str(path + Folder + '\\frame92.JPG'))
cv2.imshow('Imag_92',Imag_92)

A2 = Imag_92[:,:,2]
Orginal_gray_92 = A2

##cv2.imshow('Orginal_gray_92', A2)


dimns = Imag_92.shape
nr = dimns[0]
nc = dimns[1]
nm = dimns[2]
print("NUmber of Rows:",nr )
print("NUmber of colums:", nc )
print("NUmber of Matrix:", nm )

threshold = 150 # 150 #55 70

for k1 in range(nr):
	for k2 in range(nc):

		if(A2[k1,k2]>=threshold):
			A2[k1,k2] = 255
		else:
			A2[k1,k2] = 0



#cv2.imshow('Image2', A2)
A2[1:99,:] = 0;
#cv2.imshow('Image3', A2)
A2[:,501:] = 0;
#cv2.imshow('Image4', A2)
kernel_1 = np.ones((2,2),np.uint8)
A2 = cv2.erode(A2,kernel_1,iterations = 1)
#cv2.imshow('Image5', A2)

kernel = np.ones((10,10),np.uint8)
A2 = cv2.morphologyEx(A2, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('Image6', A2)

A2[A2>=1] = 1
#cv2.imshow('Image7', A2)

for k1 in range(1,nr):
	for k2 in range(1,nc):

		A2[k1,k2] = A2[k1,k2]*Orginal_gray_92[k1,k2]

##cv2.imshow('Template_92', A2)

A5 = A2;

for j11 in range(1,nr):
	for j22 in range(1,nc):

		if(A5[j11,j22] >=1):
			A5[j11,j22] = 1
		else:
			A5[j11,j22] = 0

A6 = np.ones((nr,nc,nm),np.uint8)
A7 = np.ones((nr,nc,nm),np.uint8)

A6[:,:,0] = A5;
A6[:,:,1] = A5;
A6[:,:,2] = A5;

for k11 in range(1,nr):
	for k22 in range(1,nc):
		for k33 in range(1,nm):

			A7[k11,k22,k33] = Imag_92[k11,k22,k33]*A6[k11,k22,k33]

cv2.imshow('Template_92_Blob', A7)


## for number of persons identification 
A8 = A7>= 150  # 150 205
A9 = np.array(A8,dtype=np.double)
A10 = A9[:,:,1]
##cv2.imshow('A10_valid',A10)
A11 = np.array(A10,dtype = np.uint8)
ret_Frame_92, labels = cv2.connectedComponents(A11)
print('number of Persomns in Frame_92 = ', ret_Frame_92)



################################  for image 96#######################################3

##Imag_96 = cv2.imread('frame96.JPG')

Imag_96 = cv2.imread(str(path + Folder + '\\frame96.JPG'))
cv2.imshow('Imag_96',Imag_96)

A2 = Imag_96[:,:,2]
Orginal_gray_96 = A2

##cv2.imshow('Orginal_gray_96', A2)


dimns = Imag_96.shape
nr = dimns[0]
nc = dimns[1]
nm = dimns[2]
print("NUmber of Rows:",nr )
print("NUmber of colums:", nc )
print("NUmber of Matrix:", nm )

threshold = 150 #55 70

for k1 in range(nr):
	for k2 in range(nc):

		if(A2[k1,k2]>=threshold):
			A2[k1,k2] = 255
		else:
			A2[k1,k2] = 0

#cv2.imshow('Image2', A2)
A2[1:99,:] = 0;
#cv2.imshow('Image3', A2)
A2[:,501:] = 0;
#cv2.imshow('Image4', A2)
kernel_1 = np.ones((2,2),np.uint8)
A2 = cv2.erode(A2,kernel_1,iterations = 1)
#cv2.imshow('Image5', A2)


kernel = np.ones((10,10),np.uint8)
A2 = cv2.morphologyEx(A2, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('Image6', A2)

A2[A2>=1] = 1
#cv2.imshow('Image7', A2)



for k1 in range(1,nr):
	for k2 in range(1,nc):

		A2[k1,k2] = A2[k1,k2]*Orginal_gray_96[k1,k2]


A5 = A2;

for j11 in range(1,nr):
	for j22 in range(1,nc):

		if(A5[j11,j22] >=1):
			A5[j11,j22] = 1
		else:
			A5[j11,j22] = 0

A6 = np.ones((nr,nc,nm),np.uint8)
A7 = np.ones((nr,nc,nm),np.uint8)

A6[:,:,0] = A5;
A6[:,:,1] = A5;
A6[:,:,2] = A5;

for k11 in range(1,nr):
	for k22 in range(1,nc):
		for k33 in range(1,nm):

			A7[k11,k22,k33] = Imag_96[k11,k22,k33]*A6[k11,k22,k33]

cv2.imshow('Template_96_Blob', A7)

## for number of persons identification 
A8 = A7>= 150 # 150
A9 = np.array(A8,dtype=np.double)
A10 = A9[:,:,1]
##cv2.imshow('A10_valid',A10)
A11 = np.array(A10,dtype = np.uint8)
ret_Frame_96, labels = cv2.connectedComponents(A11)
print('number of Persomns in Frame_96 = ', ret_Frame_96)


####################################################################

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

