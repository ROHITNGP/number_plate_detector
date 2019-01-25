import cv2
from PIL import Image
import pytesseract

print(cv2.useOptimized())
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
# cv2.namedWindow("frame_threshold",cv2.WINDOW_NORMAL)

img = cv2.imread("/home/rohit/Desktop/rgb01.png")
frame_threshold = cv2.inRange(img, (0,255,0), (0,255,0))
img = cv2.rectangle(img,(100,100),(200,200),(255,255,0),3)
cv2.putText(img,'OpenCV',(10,100), 3 , 2,(0,0,0),2,cv2.LINE_AA)


# cv2.imshow("frame_threshold",frame_threshold)
cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()