import pytesseract
from PIL import Image
import cv2

crop = cv2.imread("/home/rohit/Pictures/vlcsnap-2019-01-25-22h47m45s335.png")
# crop = crop[483:534, 726:886, :]

h,w,c = crop.shape

#crop = cv2.resize(crop,(1000, h  * 1000 // w))
cv2.imshow("crop",crop)

gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
ret2, th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow(" ", th2)

#p = pytesseract.image_to_string(th2)
#print(p)

cv2.waitKey(0) 
cv2.destroyAllWindows()