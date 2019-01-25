import numpy as np
import cv2

cap = cv2.VideoCapture('/home/rohit/Desktop/Video/VID_20190125_131806.mp4')
cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

while(cap.isOpened()):
	ret, img = cap.read()

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret2,frame_threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imshow('frame',gray)
	cv2.imshow("threshold", frame_threshold)
	cnts, hierarchy = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(img, cnts , -1, (0,255,0), 3)
	cv2.namedWindow("countour", cv2.WINDOW_NORMAL)
	cv2.imshow("countour",img)	

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
