# -*- coding: utf-8 -*-
import cv2
import os
import caffe
import numpy as np
import time

root='D:\VOC_HY/cjDATA/new_data/'   #根目录
deploy=root + 'data/squeezeNet(new)/Squeeze_SiameseNet_deploy.prototxt'    #deploy文件
caffe_model=root + 'parameters_saved/20180104/squeeze_siamese_new_iter_2000.caffemodel'   #训练好的 caffemodel
net = caffe.Net(deploy,caffe_model,caffe.TEST)   #加载model和network

#图片预处理设置
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,28,28)
transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值，前面训练模型时没有减均值，这儿就不用
transformer.set_raw_scale('data', 1)    # 缩放
transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR

# 获得视频的格式''D:/VOC_HY/VIDEO&Images/1214_shortVideo/1214_2.mp4''
videoCapture = cv2.VideoCapture('D:/VOC_HY/VIDEO&Images/1214_shortVideo/test2.mp4')
# 获得码率及尺寸
# fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
# 读帧
x = 0
firstFrame = np.zeros((1,512))
secondFrame = np.zeros((1,512))
success, frame = videoCapture.read()
frame = frame / 255
frameNum = 1#帧计数初始化，不能从0开始
net.blobs['data'].data[...] = transformer.preprocess('data', frame)  # 执行上面设置的图片预处理操作，并将图片载入到blob中
out = net.forward()
feat = out['pool10_p']
# print feat.shape
feat = np.array(feat).reshape((1, 512))
firstFrame = feat
while success:
    cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
    cv2.imshow("Video", frame)  # 显示
    # im = caffe.io.load_image(frame)  # 加载图片
    if(frameNum%25==0):
        frameNum = 1
        frame = frame / 255
        net.blobs['data'].data[...] = transformer.preprocess('data', frame)  # 执行上面设置的图片预处理操作，并将图片载入到blob中
        out = net.forward()
        feat = out['pool10_p']
        feat = np.array(feat).reshape((1, 512))
        secondFrame = feat
        x = np.sqrt(np.sum(np.square(firstFrame - secondFrame)))
        print x*100
        if ((x*100) > 0.035):
            print 'false'
        firstFrame = secondFrame
    cv2.waitKey(40)  # 延迟
    # cv2.waitKey(1)
    success, frame = videoCapture.read()  # 获取下一帧
    frameNum = frameNum + 1