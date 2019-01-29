import pytesseract
import cv2
from PIL import Image
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
	ret, img = cap.read()
	copy = img
	img = cv2.bilateralFilter(img,10,100,100)
	img = cv2.GaussianBlur(img,(5,5),0)

	# cv2.imshow('frame',img)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# th = cv2.inRange(img,(140,140,140), (255,255,255) ,cv2.THRESH_BINARY)
	# th = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
	ret2,frame_threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	cv2.imshow('threshold', frame_threshold)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,8)) 
	frame_morphed = cv2.dilate(frame_threshold,kernel,iterations = 2)
	frame_morphed = cv2.dilate(frame_threshold,kernel,iterations = 1)

	cnts, hierarchy = cv2.findContours(frame_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	i = 0
	for cnt in cnts:
		x,y,w,h = cv2.boundingRect(cnt)
		if h <= 100 and w <= 400 and 5000>h*w>500 :	# Use distance and size information
			i+=1
			cv2.rectangle(img,(x,y),(x+w+4,y+h+4),(0,255,0),2)
	print(i)
	cv2.imshow('morphed', frame_morphed)
	cv2.imshow("countour",img)
	# cv2.imshow("textbox",new)
	
	# print(pytesseract.image_to_string(frame_morphed))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()