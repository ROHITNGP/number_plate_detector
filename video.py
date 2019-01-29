import numpy as np
import cv2
import pytesseract
from PIL import Image

cap = cv2.VideoCapture('/home/rohit/Desktop/Video/VID_20190125_131806.mp4')
cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
# cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

while(cap.isOpened()):
	ret, img = cap.read()
	img = img[200:600,900:1400,:]
	img = cv2.GaussianBlur(img,(3,3),0)
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# ret2,frame_threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	frame_threshold = cv2.inRange(img, (0,0,0), (50,50,50))
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,3)) 	# Make a vertical ellipse using appropriate size
	frame_threshold = cv2.dilate(frame_threshold,kernel,iterations = 3)
	cv2.imshow("threshold", frame_threshold)
	cnts, hierarchy = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	for cnt in cnts:
		x,y,w,h = cv2.boundingRect(cnt)
		if h <= 50 and w <= 50 and 600 >h * w > 100 :	# Use distance and size information
			cv2.rectangle(img,(x-4,y-4),(x+w+4,y+h+4),(0,255,0),-1)
	
	# print(pytesseract.image_to_string(img))
	cv2.namedWindow("contour", cv2.WINDOW_NORMAL)
	cv2.imshow("contour",img)	
	

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
