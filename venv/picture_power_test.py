#针对灰度图增强的测试，利用的技术是直方图正规化
import cv2 as cv
import numpy as np
img = cv.imread(r"C:\Users\zkzs5\PycharmProjects\LSTM\venv\grey_covid\2020.02.11.20022053-p12-67%2.png", 0)
# 计算原图中出现的最小灰度级和最大灰度级
# 使用函数计算
Imin, Imax = cv.minMaxLoc(img)[:2]
# 使用numpy计算
# Imax = np.max(img)
# Imin = np.min(img)
Omin, Omax = 0, 255
# 计算a和b的值
a = float(Omax - Omin) / (Imax - Imin)
b = Omin - a * Imin
out = a * img + b
out = out.astype(np.uint8)
cv.imwrite(r'C:\Users\zkzs5\PycharmProjects\LSTM\venv\grey_covid_power\2020.02.11.20022053-p12-67%2.png',out)
cv.imshow("img", img)
cv.imshow("out", out)
cv.waitKey()