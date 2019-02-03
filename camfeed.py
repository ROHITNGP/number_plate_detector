import pytesseract
import cv2
from PIL import Image
import numpy as np
import copy

cap = cv2.VideoCapture(0)
plate = np.zeros((10,10))
plate_number_list = []
while(True):
	ret, img = cap.read()
	copy = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
	copy2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
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
		if h <= 100 and w <= 400 and 7000>h*w>800 :	# Use distance and size information
			
			cv2.rectangle(img,(x-4,y-4),(x+w+4,y+h+4),(0,255,0),-1)
			th = cv2.inRange(img,(0,250,0), (1,255,1) ,cv2.THRESH_BINARY)
			c, hier = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			for con in c:
				x1,y1,w1,h1 = cv2.boundingRect(con)
				if w1 > 300:
					cv2.rectangle(copy,(x1,y1),(x1+w1,y1+h1),(255,255,0),2)
					plate = copy2[y1:y1+h1,x1:x1+w1]
	
	x = str(pytesseract.image_to_string(plate))
	# print(x)			
	if x not in plate_number_list:
		plate_number_list.append(x)
		print(plate_number_list)

	cv2.imshow('morphed', frame_morphed)
	cv2.imshow("countour",img)
	cv2.imshow("th",th)
	cv2.imshow("boundind box",copy)
	cv2.imshow("detected plate",plate)
	# 
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows ()