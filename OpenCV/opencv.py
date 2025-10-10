import cv2
import numpy as np
import os


img_path = "image.png"
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise FileNotFoundError("找不到图片。")

cv2.imshow("Original", img_bgr)


img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)


lower_pink = np.array([140, 60, 80])
upper_pink = np.array([180, 255, 255])


mask = cv2.inRange(img_hsv, lower_pink, upper_pink)


result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)


os.makedirs("outputs", exist_ok=True)
cv2.imwrite("outputs/result_pink.png", result)

cv2.imshow("Mask", mask)
cv2.imshow("Result", result)
print("✅ 已保存结果到 outputs/result_pink.png")

cv2.waitKey(0)
cv2.destroyAllWindows()
