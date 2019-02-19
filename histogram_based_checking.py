import cv2
import xml.etree.ElementTree as ET
import numpy as np
from scipy.ndimage import gaussian_filter
import os

csv_string = ""


def get_roi(path, filename):
    filename = os.path.join(path, filename + '.xml')
    tree = ET.parse(filename)
    root = tree.getroot()
    bboxs = []

    for object in root.iter('object'):
        if object.find('name').text == 'car':
            for m in object.iter('bndbox'):
                xmin = m.find('xmin').text
                xmax = m.find('xmax').text
                ymin = m.find('ymin').text
                ymax = m.find('ymax').text
                bboxs.append([int(xmin), int(ymin), int(ymax) - int(ymin), int(xmax) - int(xmin)])

        if object.find('name').text == 'numplate':
            for m in object.iter('bndbox'):
                xmin = m.find('xmin').text
                xmax = m.find('xmax').text
                ymin = m.find('ymin').text
                ymax = m.find('ymax').text
                cx, cy = (int(xmin) + int(xmax)) // 2, (int(ymin) + int(ymax)) // 2

    x, y, h, w = bboxs[0]
    key = [cx, cy]
    roi = img[y:y + h, x:x + w]
    return roi, key, bboxs[0]


def solve(img, key, bbox, filename):
    val = 0
    reduction = -1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("blur", gray)

    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
    #cv2.imshow("mask", mask)

    dx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    dx = cv2.convertScaleAbs(dx)
    #cv2.imshow("dx", dx)

    dx = cv2.bitwise_and(dx, mask)

    h, w = dx.shape
    bins = np.sum(dx, 1)
    smooth = gaussian_filter(bins, sigma=5)

    index = np.argmax(smooth) + 1
    mean = np.sum(smooth) / len(bins)
    # print(mean)

    drawing = np.zeros(dx.shape, np.uint8)
    indices = np.where(smooth > mean)
    for index in indices[0]:
        drawing[index:index + 1, :] = 255


    #cv2.imshow("drawing", drawing)
    # print([key[1] - bbox[1]], [key[0] - bbox[0]])
    # print(drawing[key[1] - bbox[1]][key[0] - bbox[0]])
    if drawing[key[1] - bbox[1]][key[0] - bbox[0]] == 255:
        val = 1
        reduction = cv2.countNonZero(drawing) * 1.0 / (h * w) * 100

    # else:
    #     print(filename)


    # plt.plot(smooth, range(h))
    # plt.show()
    #
    # cv2.waitKey(0)

    return val, reduction


def background():
    cap = cv2.VideoCapture('/home/aman/Videos/vlc-record-2019-02-17-18h00m44s-VID_20190217_115314.mp4-.mp4')
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=500)
    while (1):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        ###cv2.imshow('frame', fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()


path = "/home/aman/number_plate_detector/dataset"
filenames = os.listdir(path)
count = 0

for filename in filenames:
    filename, file_extension = os.path.splitext(filename)
    # print(filename)
    if file_extension == '.png':
        csv_string = csv_string + filename + ","

        filename = os.path.join(path, filename)
        # print(filename)
        img = cv2.imread(filename + ".png")
        roi, key, bbox = get_roi(path, filename)
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)

        #cv2.imshow("img", img)
        val, reduction = solve(roi, key, bbox, filename)

        if val == 1:
            count = count + 1
        # cv2.waitKey(0)
        csv_string = csv_string + str(reduction) + "\n"

print (count)
with open('output.csv', 'w') as f:
    f.write(csv_string)

