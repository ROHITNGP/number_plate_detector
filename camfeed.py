import pytesseract
import cv2
from PIL import Image
import numpy as np
from time import gmtime, strftime

cap = cv2.VideoCapture("/home/rohit/Desktop/Video/VID_20190125_133435.mp4")
plate = np.zeros((100,100))
plate_number_list = []
counter =0
cv2.namedWindow(" ", cv2.WINDOW_NORMAL)
cv2.namedWindow("draw", cv2.WINDOW_NORMAL)
while(True):
	ret, img = cap.read()
	cv2.imshow('frame',img)
	blur = cv2.GaussianBlur(img, (11,11), 0)
	cv2.imshow("heol", blur)
	gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
	
	edges = cv2.Canny(blur,40,80)
	cv2.imshow("edges", edges)

	
	cnts, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	h,w,c = img.shape
	drawing = np.zeros((h,w), np.uint8)
	
	approxs = []
	for cnt in cnts:
		epsilon = cv2.arcLength(cnt, True) * 0.05
		approx = cv2.approxPolyDP(cnt, epsilon, True)
		if len(approx) == 4: 							
		 	if cv2.contourArea(cnt) > 2000:
				cv2.drawContours(drawing, [approx], -1, (255,0,0), -1)
	seg = cv2.bitwise_and(img, img, mask=drawing)
	cv2.imshow("draw", seg)	





	# frame_threshold = cv2.inRange(img,(0,0,0), (80,80,80) ,cv2.THRESH_BINARY)
	# cv2.imshow('threshold', frame_threshold)
	
	# cnts, hierarchy = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# i = 0
	# for cnt in cnts:
	# 	x,y,w,h = cv2.boundingRect(cnt)
	# 	if h <= height and w <= width and high_area>h*w>low_area :	# Use distance and size information
	# 		cv2.rectangle(img,(x-4,y-4),(x+w+4,y+h+4),(0,255,0),-1)
	# 		th = cv2.inRange(img,(0,250,0), (1,255,1) ,cv2.THRESH_BINARY)
	# 		c, hier = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# 		for con in c:
	# 			x1,y1,w1,h1 = cv2.boundingRect(con)
	# 			if w1 > 300:
	# 				cv2.rectangle(copy,(x1,y1),(x1+w1,y1+h1),(255,255,0),2)
	# 				plate = copy2[y1:y1+h1,x1:x1+w1]
	# counter+=1


	"""if counter % 50 == 0:
		# print(counter)
		x = str(pytesseract.image_to_string(plate))

	# print(x)			
		if x not in plate_number_list:
			plate_number_list.append(x)
			print(plate_number_list)"""

	# cv2.imshow('morphed', frame_morphed)
	# cv2.imshow("countour",img)
	# # cv2.imshow("th",th)
	# cv2.imshow("detected plate",plate)
	# cv2.imshow("boundind box",copy)

	if cv2.waitKey(30) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()