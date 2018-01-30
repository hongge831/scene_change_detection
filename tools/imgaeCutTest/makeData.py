#encoding=utf-8
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys,os
from PIL import Image
##读取保存的列表文件
totalList = pickle.load(open("./totalList.txt",'rb'))
##保存截取的结果列表到本地的totalList.txt文件
fileSrc = "D:/PythonSpace/imgaeCutTest/data/"
fileList = os.listdir(fileSrc)
for f in fileList:
    imgSrc = fileSrc+f+'/'
    



