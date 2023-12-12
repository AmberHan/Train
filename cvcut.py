# -*- coding:utf-8 -*-
# @Author  : Ricky Xu
# @Time    : 2023/9/20 5:57 下午
# @FileName: cvcut.py
# @Software: PyCharm
# 导入所需要的库
import cv2
import numpy as np

# 运行前先手动or用终端建立文件夹 mkdir -p ./video/cut/
# 将4.mp4换成你的视频路径和名字
# 修改timeF为你需要的间隔，假如视频是30帧/s， timeF=15就是0.5秒截图一次，timeF=30就是一秒截取一次。
# 修改./mypic/路径为你自己想存储的照片路径
# 修改j=0可以设置命名的起始点，可以不修改。

# 定义保存图片函数
# image:要保存的图片名字
# addr；图片地址与相片名字的前部分
# num: 相片，名字的后缀。int 类型
def save_image(image, addr, num):
    address = addr + str(num) + '.jpg'
    cv2.imwrite(address, image)


# 读取视频文件
videoCapture = cv2.VideoCapture("video/【昆曲】经典折子戏合集 p01 《牡丹亭·游园·惊梦》（孔爱萍、钱振荣）.mp4")
# 通过摄像头的方式
# videoCapture=cv2.VideoCapture(1)

# 读帧
success, frame = videoCapture.read()
i = 1000
timeF = 90
j = 0
while success:
    i = i + 1
    if (i % timeF == 0):
        j = j + 1
        save_image(frame, './video/cut/', j)
        print('save image:', i)
    success, frame = videoCapture.read()

