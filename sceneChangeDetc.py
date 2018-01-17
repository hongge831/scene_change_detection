# -*- coding: utf-8 -*-
#本代码是将场景图片分割成9*9网格后，分别进网络判断距离（基础网络是lenet）
import cv2
import os
import caffe
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
##将图片分成9*9个cell，计算每一个cell的feat值，用比较浅层的网络比较
def computeArrList(frame,num_part=9):
    im = Image.fromarray(frame)
    crop_dic = []  # put the coordiantes in the list
    width = im.size[0] / num_part
    height = im.size[1] / num_part
    for i in range(num_part):
        for j in range(num_part):
            crop_dic.append((i * width, j * height, i * width + width, j * height + height))
    i = 0
    imArrList = []  # 碎片图片转化成numpyarray后集中保存在列表中
    # print len(crop_dic)
    for t in crop_dic:
        im_new = im.crop(t)
        imArrList.append(np.asarray(im_new, float))
        # im_new.save('./new_file/im1_' + str(i) + '.jpg')
        i = i + 1
    return imArrList
#####函数结束
###加载模型&设置根目录路径
root='D:\VOC_HY/cjDATA/new_data/'   #根目录
deploy=root + 'caffe_model/mnist_siamese.prototxt'    #deploy文件
caffe_model=root + 'parameters_saved/siamese_lenet_tested/SiameseNet__iter_50000.caffemodel'   #训练好的 caffemodel
net = caffe.Net(deploy,caffe_model,caffe.TEST)   #加载model和network

#图片预处理设置
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,28,28)
transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
transformer.set_raw_scale('data', 1)    # 缩放
transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR
#src1 = 'D:/VOC_HY/VIDEO&Images/1214_shortVideo/test2.mp4'
videoCapture = cv2.VideoCapture('D:/VOC_HY/VIDEO&Images/object_tracking_video/AVSS_PV_Easy_Divx.avi')
# 获得码率及尺寸
# fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
# 读帧
##初始化变量的值
x = 0
frameNum = 1#设置帧计数初始值为1，不能设置为0
# new = np.zeros((1,10))#设置初始的结果矩阵
dis1 = []
dis2 = []
imArrList = []
##初始化变量结束
success, frame = videoCapture.read()
imArrList = computeArrList(frame, num_part=9)
for im in imArrList:
    im = im / 255
    net.blobs['data'].data[...] = transformer.preprocess('data', im)  # 执行上面设置的图片预处理操作，并将图片载入到blob中
    out = net.forward()
    feat = np.array(out['feat']).reshape((1, 10))
    dis1.append(feat)
while success:
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.imshow("Video", frame)  # 显示
    if (frameNum % 25 == 0):
        frameNum = 1
        imArrList = computeArrList(frame, num_part=9)
        for im in imArrList:
            im = im / 255
            net.blobs['data'].data[...] = transformer.preprocess('data', im)  # 执行上面设置的图片预处理操作，并将图片载入到blob中
            out = net.forward()
            feat = np.array(out['feat']).reshape((1, 10))
            dis2.append(feat)
        # imArrList1.extend(imArrList2)  ###将两个列表合并成一个列表
        totalNum = 81
        count = 0
        result = []
        for i in range(81):
            x = np.sqrt(np.sum(np.square(dis1[i] - dis2[i])))
            print ((i), (i + 81)), '  ', x,
            result.append(x)
            if (x > 0.06):
                x = False
            else:
                count = count + 1
                x = True
            print x
        if count<=45:
            print "ALERT!!!!!!"
        else:
            print "SAFE STATUS."
        print count
        ##计算结束，把imArrList2替换成imArrList1并且将imArrList2置空
        dis1 = dis2
        dis2 = []
    cv2.waitKey(40)  # 延迟
    success, frame = videoCapture.read()  # 获取下一帧
    frameNum = frameNum + 1














# frame = frame / 255
# frameNum = 1#帧计数初始化，不能从0开始
# net.blobs['data'].data[...] = transformer.preprocess('data', frame)  # 执行上面设置的图片预处理操作，并将图片载入到blob中
# out = net.forward()
# feat = out['feat']
# # print feat.shape
# feat = np.array(feat).reshape((1, 512))
# firstFrame = feat
# while success:
#     cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
#     cv2.imshow("Video", frame)  # 显示
#     # im = caffe.io.load_image(frame)  # 加载图片
#     if(frameNum%25==0):
#         frameNum = 1
#         print frame.dtype
#
#
#         frame = frame / 255
#         print type(frame)
#         net.blobs['data'].data[...] = transformer.preprocess('data', frame)  # 执行上面设置的图片预处理操作，并将图片载入到blob中
#         out = net.forward()
#         feat = out['feat']
#         feat = np.array(feat).reshape((1, 512))
#         secondFrame = feat
#         x = np.sqrt(np.sum(np.square(firstFrame - secondFrame)))
#         print x*100
#         if ((x*100) > 0.2):
#             print 'false'
#         firstFrame = secondFrame
#     cv2.waitKey(40)  # 延迟
#     # cv2.waitKey(1)
#     success, frame = videoCapture.read()  # 获取下一帧
#     frameNum = frameNum + 1