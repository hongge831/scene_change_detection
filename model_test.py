# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import numpy as np
import pylab
import random
import cv2

root='D:\VOC_HY/cjDATA/new_data/'   #根目录
deploy=root + 'caffe_model/Squeeze_SiameseNet_deploy.prototxt'    #deploy文件
caffe_model=root + 'parameters_saved/1213_squezenet/squeeze_siamese_iter_1581.caffemodel'   #训练好的 caffemodel
net = caffe.Net(deploy,caffe_model,caffe.TEST)   #加载model和network


#图片预处理设置
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,227,227)
transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(227,227,3)变为(3,227,227)
transformer.set_raw_scale('data', 225)    # 缩放到【0，255】之间
transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR

img_src = 'D:/VOC_HY/VIDEO&Images/1124/1124_images/1124rotation/'
new = np.zeros((1,16))
for file in os.listdir(img_src):
    img = img_src + file
    print img
    im = caffe.io.load_image(img)  # 加载图片
    # im = cv2.imread(img)
    # im = im/255
    net.blobs['data'].data[...] = transformer.preprocess('data', im)  # 执行上面设置的图片预处理操作，并将图片载入到blob中
    out = net.forward()
    pool10 = out['feat']
    pool10 = np.array(pool10).reshape((1,16))
    new = np.vstack((new, pool10))

new = new[1:,:]
totalNum = 45
trueNum = 0
falseNum = 0
for i in range(totalNum):
    x = np.sqrt(np.sum(np.square(new[i] - new[i+1])))
    print ((i),(i+1)),'  ',x,
    if(x>0.03):
        x=False
        falseNum = falseNum + 1
    else:
        x = True
        trueNum = trueNum + 1
    print x
print trueNum,falseNum