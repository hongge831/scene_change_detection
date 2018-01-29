#encoding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys,os
from PIL import Image
result = []#用于记录所需要的图片块的位置坐标（注意不是像素坐标）
crop_dic = []#用于记录每一块图片的像素坐标
#打印鼠标点击的位置以及做标记
def location(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
        result.append((x//64,y//64))
        print(x//64,y//64)

#画网格函数
def drawGridLines(width,height):
    gridNum = 10#总共的网格数量是grid*grid，这里是100
    for i in range(gridNum):
        cv2.line(img, (0, i*64), (640, i*64), (255, 255, 255), 1, 1)
    for i in range(gridNum):
        cv2.line(img, (i*64, 0), (i*64, 640), (255, 255, 255), 1, 1)
#计算切分的区域矩形的左上和右下坐标值
def computeCutCordinate(width,height,num_part):
    w = width/num_part
    h = height/num_part
    for i in range(num_part):
        for j in range(num_part):
            crop_dic.append((j * h, i * w, j * h + h, i * w + w))

#cv初始设置
img = cv2.imread("./test.jpg")
img=cv2.resize(img,(640,640),interpolation=cv2.INTER_CUBIC)
drawGridLines(640,640)
cv2.namedWindow("testWindow",cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("testWindow", 640, 640)
cv2.setMouseCallback('testWindow', location)
computeCutCordinate(640,640,10)

###主函数入口
while (1):
    cv2.imshow('testWindow', img)
    if cv2.waitKey(1)&0xFF == ord('q'):#按q键退出
        break
cv2.destroyAllWindows()
###获取位置坐标结束
imgCut = Image.open("./test.jpg")
#这一步比较容易错，因为之前resize了后面的open图像也要resize
imgCut = imgCut.resize((640,640))
for index ,t in enumerate(result):
    imNew = imgCut.crop(crop_dic[t[0]+t[1]*10])
    imNew.save('./testNew'+str(index)+'.jpg')
    print(crop_dic[t[0]+t[1]*10])
