import pytesseract
from PIL import Image
import cv2
cv2.namedWindow("crop", cv2.WINDOW_NORMAL)
cv2.namedWindow("th2", cv2.WINDOW_NORMAL)

crop = cv2.imread("/home/rohit/Pictures/vlcsnap-2019-01-26-20h22m47s024.png")
crop = crop[400:430, 825:850, :]

h,w,c = crop.shape

crop = cv2.resize(crop,(28, h  * 28 // w))
cv2.imshow("crop",crop)

gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
ret2, th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow("th2", th2)

p = pytesseract.image_to_string(th2)
#print(p)

cv2.waitKey(0) 
cv2.destroyAllWindows()