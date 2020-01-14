#1. Recode all examples;

import cv2
import numpy as np
import matplotlib.pyplot as plt
#查看opencv的版本
print(cv2.__version__)
image_gray = cv2.imread('lena.jpg',0)#0表示灰度图像，1表示彩色图像
image_RGB = cv2.imread('lena.jpg',1)
cv2.imshow('lena_gray',image_gray)
cv2.imshow('lena_RBG',image_RGB)
key = cv2.waitKey()
if key==27:
    cv2.destroyAllWindows()
# 用matplotlib显示图像的通道是RGB,而opencv的通道是BGR
# 方法一
image_RGB = cv2.cvtColor(image_RGB,cv2.COLOR_BGR2RGB)
# 方法二
B,G,R = cv2.split(image_RGB)
image_RGB = cv2.merge((R,G,B))
plt.imshow(image_RGB)
plt.show()

###图像颜色的改变
def color_chang(img,Bnum,Gnum,Rnum):
    '''
    :param img: 原始图像
    :param Bnum: 改变B通道的值
    :param Gnum: 改变G通道的值
    :param Rnum: 改变R通道的值
    :return: 改变颜色后的图像
    '''
    B,G,R = cv2.split(img)
    if Bnum==0:
        pass
    elif Bnum>0:
        de = 255-Bnum
        B[B>de] =255
        B[B<=de] =(Bnum+B[B<=de]).astype(img.dtype)
    elif Bnum<0:
        de = 0-Bnum
        B[B>=de] = (Bnum + B[B>=de]).astype(img.dtype)
        B[B<de] = 0

    if Gnum==0:
        pass
    elif Gnum>0:
        de = 255-Gnum
        B[B>de] =255
        B[B<=de] =(Gnum+B[B<=de]).astype(img.dtype)
    elif Gnum<0:
        de = 0-Gnum
        B[B>=de] = (Gnum + B[B>=de]).astype(img.dtype)
        B[B<de] = 0

    if Rnum==0:
        pass
    elif Rnum>0:
        de = 255-Rnum
        B[B>de] =255
        B[B<=de] =(Rnum+B[B<=de]).astype(img.dtype)
    elif Rnum<0:
        de = 0-Rnum
        B[B>=de] = (Rnum + B[B>=de]).astype(img.dtype)
        B[B<de] = 0

    img = cv2.merge((B,G,R))
    return img

change_color = color_chang(image_RGB,25,-5,50)
cv2.imshow('chang',change_color)

# ###################################
# #gamma矫正(gamma也大图片越亮对比度越高)
def gamma_adjust(img,gamma=1.0):
    tabel = []
    inv_gamma = 1.0/gamma
    for i in range(256):
        tabel.append(((i/255.0)**inv_gamma)*255)
    tabel = np.array(tabel).astype('uint8')
    image_gamma = cv2.LUT(img,tabel)
    return image_gamma
img = cv2.imread('timg.jpeg',0)
image_gamma = gamma_adjust(img,gamma=2)
cv2.imshow('gamma',image_gamma)
cv2.imshow('scr',img)

###############################
##image crop

print(img.shape)
img_crop = img[100:500,50:400]
cv2.imshow('crop',img_crop)

###直方图,以及直方图均衡（只能对单通道进行直方图均衡）
img_equalize = cv2.equalizeHist(img)
cv2.imshow('equalize',img_equalize)
hist = img.flatten()
plt.hist(hist,256,[0,255],color='r')
plt.show()
hist1 = img_equalize.flatten()
plt.hist(hist1,256,[0,255],color='b')
plt.show()

###################################
# scale+rotation+translation = similarity transform
M= cv2.getRotationMatrix2D((image_RGB.shape[1]/2,image_RGB.shape[0]/2),30,0.8)
img_rotate = cv2.warpAffine(image_RGB,M,(image_RGB.shape[1],image_RGB.shape[0]))
cv2.imshow('retation',img_rotate)


####仿射变换
rows,cols,c = image_RGB.shape
pos1 = np.float32([[0,0],[cols-1,0],[0,rows-1]])
pos2 = np.float32([[cols*0.2,rows*0.1],[cols*0.9,rows*0.2],[cols*0.1,rows*0.9]])
M = cv2.getAffineTransform(pos1,pos2)
dst = cv2.warpAffine(image_RGB,M,(cols,rows))
cv2.imshow('affine',dst)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

############
#投影变换
def random_warp(img, row, col):
    height, width, channels = img.shape

    random_margin = 60
    x1 = np.random.randint(-random_margin, random_margin)
    y1 = np.random.randint(-random_margin, random_margin)
    x2 = np.random.randint(width - random_margin - 1, width - 1)
    y2 = np.random.randint(-random_margin, random_margin)
    x3 = np.random.randint(width - random_margin - 1, width - 1)
    y3 = np.random.randint(height - random_margin - 1, height - 1)
    x4 = np.random.randint(-random_margin, random_margin)
    y4 = np.random.randint(height - random_margin - 1, height - 1)

    dx1 = np.random.randint(-random_margin, random_margin)
    dy1 = np.random.randint(-random_margin, random_margin)
    dx2 = np.random.randint(width - random_margin - 1, width - 1)
    dy2 = np.random.randint(-random_margin, random_margin)
    dx3 = np.random.randint(width - random_margin - 1, width - 1)
    dy3 = np.random.randint(height - random_margin - 1, height - 1)
    dx4 = np.random.randint(-random_margin, random_margin)
    dy4 = np.random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return  img_warp
img_warp = random_warp(image_RGB,image_RGB.shape[0],image_RGB.shape[1])
cv2.imshow('00',img_warp)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()
