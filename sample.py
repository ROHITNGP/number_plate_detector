import cv2
import numpy as np

cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
# cv2.namedWindow("1", cv2.WINDOW_NORMAL)
# cv2.namedWindow("2", cv2.WINDOW_NORMAL)
# cv2.namedWindow("3", cv2.WINDOW_NORMAL)


img = cv2.imread("/home/rohit/Downloads/0_578_872_0_70_http___cdni.autocarindia.com_ExtraImages_20180813013844_EVs-to-get-green-plates.jpg")

frame_threshold = cv2.inRange(img, (200,200,200), (255,255,255)) # Try to use DRGB colorspace, if not do it only for black

cv2.imshow("threshold", frame_threshold)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,3)) 	# Make a vertical ellipse using appropriate size
frame_threshold = cv2.dilate(frame_threshold,kernel,iterations = 2)
frame_threshold = cv2.erode(frame_threshold,kernel,iterations = 1)
cv2.imshow("1", frame_threshold)

cnts, hierarchy = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

bboxs = []

for cnt in cnts:
	x,y,w,h = cv2.boundingRect(cnt)
	if h <= 50 and w <= 50 and 5>=float(max(h, w)) / min(h, w)>=1 and h*w < 800 and h*w >= 100:	# Use distance and size information
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		bboxs.append([x,y,w,h])

cv2.imshow("3", img)
cv2.waitKey(0) 
cv2.destroyAllWindows()