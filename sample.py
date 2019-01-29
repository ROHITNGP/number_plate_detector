import cv2
import numpy as np
import pytesseract
from PIL import Image

cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
cv2.namedWindow("dilate", cv2.WINDOW_NORMAL)
# cv2.namedWindow("2", cv2.WINDOW_NORMAL)
# cv2.namedWindow("3", cv2.WINDOW_NORMAL)


img = cv2.imread("/home/rohit/Pictures/vlcsnap-2019-01-28-16h43m26s243.png") #read image
img = img[200:500,800:1500] #crop
# img = cv2.GaussianBlur(img,(3,3),0)
img = cv2.bilateralFilter(img,9,50,50)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret2,frame_threshold = cv2.threshold(gray,0,250,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
frame_threshold = cv2.inRange(img, (0,0,0), (70,70,70)) # Try to use DRGB colorspace, if not do it only for black

cv2.imshow("threshold", frame_threshold)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,3)) 	# Make a vertical ellipse using appropriate size
frame_threshold = cv2.dilate(frame_threshold,kernel,iterations =2)
# frame_threshold = cv2.erode(frame_threshold,kernel,iterations = 1)
cv2.imshow("dilate", frame_threshold)

cnts, hierarchy = cv2.findContours(frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bboxs = []

for cnt in cnts:
	x,y,w,h = cv2.boundingRect(cnt)
	if h <= 100 and w <= 100 and h * w > 100 :	# Use distance and size information
		cv2.rectangle(img,(x,y),(x+w+4,y+h+4),(0,255,0),2)
		bboxs.append([x,y,w,h])

x,y,w,h  = bboxs[3]
print(bboxs[1])
text = img[y:y+w,x:x+w,:]
text = cv2.resize(text, None, fx=5,fy=5, interpolation = cv2.INTER_CUBIC) 
gray = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)
ret2,frame_threshold = cv2.threshold(gray,0,250,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow("found", text)
config = ('-l eng --oem 1 --psm 3')
p = pytesseract.image_to_string(frame_threshold,config=config)
print('no. of  countours: '+str(len(cnts)))
cv2.imshow("3", img)
cv2.waitKey(0) 
cv2.destroyAllWindows()