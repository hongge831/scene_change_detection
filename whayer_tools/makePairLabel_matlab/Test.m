clear;
clc;
txt_path = 'D:\VOC_HY\WHAYER_VOC2017\变电\把手（旋钮、钥匙旋钮、断开路指示）\JPEGImages\'; 
img_path = 'D:\VOC_HY\WHAYER_VOC2017\变电\把手（旋钮、钥匙旋钮、断开路指示）\Annotations\';

img_path_list = dir(strcat(img_path,'*.xml'));
len = length(img_path_list);
for j = 1:len 
    img_name = img_path_list(j).name;  
    img_name=img_name(1:end-4);     
    image1 = strcat(img_path,img_name,'.xml');
    txt1 =  strcat(txt_path,img_name,'.jpg');
    if(~exist(txt1,'file')) delete(image1,'file'); continue
    end
end
