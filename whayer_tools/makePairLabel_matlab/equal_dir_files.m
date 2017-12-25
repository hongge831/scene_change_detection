clear;
clc;
txt_path = 'D:\PythonSpace\ML_ZJU_Pytorch_Net\A\image\'; 
img_path = 'D:\PythonSpace\ML_ZJU_Pytorch_Net\A\mask\';

img_path_list = dir(strcat(img_path,'*.bmp'));
len = length(img_path_list);
for j = 1:len 
    img_name = img_path_list(j).name;  
    img_name=img_name(1:end-4);     
    image1 = strcat(img_path,img_name,'.bmp');
    txt1 =  strcat(txt_path,img_name,'.bmp');
    if(~exist(txt1,'file')) delete(image1,'file'); continue
    end
end
