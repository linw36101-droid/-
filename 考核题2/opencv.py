import cv2
import numpy as np


img = cv2.imread("image1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


_, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)



contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


for contour in contours:
    area = cv2.contourArea(contour)
    if area < 100 or area > 10000:
        continue


    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)


    if 5 <= len(approx) <= 7:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.drawContours(img, [approx], 0, (0, 255, 0), 1)


cv2.imshow("Result", img)
cv2.imwrite("outputs/final_result.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()