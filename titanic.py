import cv2
import os

vidcap = cv2.VideoCapture('titanic.mpg')
success,image = vidcap.read()

no_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Tottal_no_soccer_frames :',no_frames )
dimns = image.shape
print('Dimension:',dimns)
      
print("Please wait program is executing")
count = 0
success = True
while success:
  success,image = vidcap.read()
  Folder = '\\titanic frames'
  path1 = os.getcwd()
  path = str(path1 + Folder )
  cv2.imwrite(os.path.join(path, "frame%d.jpg" % count,),image)
  count += 1
  # save frame as JPEG file
  cv2.waitKey(10)

print("All the images all storded in the folder titanic frames")

######################################

cv2.destroyAllWindows()

data1 = cv2.imread(str(path + '//Frame91.jpg'))
cv2.imshow('frame91',data1)

data2 = cv2.imread(str(path + '//Frame197.jpg'))
cv2.imshow('frame197',data2)

data3 = cv2.imread(str(path + '//Frame299.jpg'))
cv2.imshow('frame299',data3)




