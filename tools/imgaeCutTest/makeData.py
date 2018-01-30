#encoding=utf-8
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys,os
from PIL import Image
##读取保存的列表文件
totalList = pickle.load(open("./totalList.txt",'rb'))
#patchInfo用于保存每一种场景有多少张图片以及每张图片有多少个patch
patchInfo = []
imageN = 0
patchN = 0
##保存截取的结果列表到本地的totalList.txt文件
fileSrc = "D:/PythonSpace/imgaeCutTest/data/"
fileList = os.listdir(fileSrc)
for fileN,f in enumerate(fileList):
    imgSrc = fileSrc+f+'/'
    cropList = totalList[fileN]
    for imgN,img in enumerate(os.listdir(imgSrc)):
        imgDir = imgSrc+img
        imgTemp = Image.open(imgDir).resize((640,640))
        imageN+=1
        for ptcN, t in enumerate(cropList):
            imgTemp.crop(t).save("./stored/"+f+'_'+str(imgN)+'_'+str(ptcN)+'.jpg')
            patchN+=1
    patchInfo.append((imageN,patchN))
    imageN = 0
    patchN = 0
    print("-->finish "+f)
pickle.dump(patchInfo,open("./patchInfo.txt","wb"),True)
print(patchInfo)







