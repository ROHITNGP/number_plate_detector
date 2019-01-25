import cv2
import numpy as np
import pytesseract
from PIL import Image

cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)


img = cv2.imread("/home/rohit/Pictures/vlcsnap-2019-01-25-22h47m45s335.png")
img = img[304:440,1067:1333,:]
# img = cv2.bilateralFilter(orig,10,100,100)
# img = cv2.GaussianBlur(img,(5,5),0)

# cv2.namedWindow("original",cv2.WINDOW_NORMAL)
# cv2.imshow("original",img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret2,frame_threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Try to use DRGB colorspace, if not do it only for black
# 
cv2.imshow("threshold", frame_threshold)

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,2)) 
# frame_morphed = cv2.dilate(frame_threshold,kernel,iterations = 2)

# cv2.namedWindow("morphed", cv2.WINDOW_NORMAL)
# cv2.imshow("morphed", frame_threshold)

cnts, hierarchy = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
approxs = []
for cnt in cnts:
	epsilon = 0.001*cv2.arcLength(cnt,True)
	approx = cv2.approxPolyDP(cnt,epsilon,True)
	if len(approx) == 4 and cv2.isContourConvex(cnt) == True: # and <= cv2.contourArea(cnt) <=:
		approxs.append(approx)

# Add area constraints in the area before appending it to approxs.
cv2.drawContours(img, cnts , -1, (0,255,0), 3)
cv2.namedWindow("countour", cv2.WINDOW_NORMAL)
cv2.imshow("countour",img)
print( approxs[1] )

# bboxs = []

# for cnt in cnts:
# 	x,y,w,h = cv2.boundingRect(cnt)
# 	if 20<h <= 50 and 20< w <= 50:	
# 		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
# 		bboxs.append([x,y,w,h])


print(pytesseract.image_to_string(frame_threshold))
# x,y,w,h = bboxs[2]

cv2.waitKey(0) 
cv2.destroyAllWindows()