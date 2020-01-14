import cv2
import numpy as np


def random_rotation(image,scale):
    '''
    对图片进行随机的旋转
    :param image: 输入读入后的图像
    :param scale: 旋转后的图像的尺度
    :return: 旋转后的图像
    '''
    angle = np.random.randint(0,90)#随机的角度
    M = cv2.getRotationMatrix2D((image.shape[1]/2,image.shape[0]/2),angle,scale)#旋转矩阵
    image_rotation = cv2.warpAffine(image,M,(image.shape[0],image.shape[1]))
    return image_rotation
def random_color_shift(img):
    '''
    对图片进行随机的颜色变换
    :param img:读入后的图片
    :return:随机改变颜色后的图片
    '''
    B, G, R = cv2.split(img)#分离通道
    b_rand = np.random.randint(-50, 50)
    '''分别对RGB三通道进行操作，注意数值越界'''
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

    g_rand = np.random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    r_rand = np.random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)
    img_merge = cv2.merge((B, G, R))
    return img_merge
def random_image_crop (image):
    '''
    随机的裁剪
    :param image: 读入后的图片
    :return: 裁剪后的图片
    '''
    media0 = image.shape[0]/4
    media1 = image.shape[1]/4
    x1 = np.random.randint(0,media0)
    x2 = np.random.randint(image.shape[0]-media0-1,image.shape[0]-1)
    y1 = np.random.randint(0,media1)
    y2 = np.random.randint(image.shape[0]-media1-1,image.shape[1]-1)
    image_crop = image[x1:x2,y1:y2]#裁剪图片
    return image_crop
def random_warp(img):
    '''
    随机投影变换
    :param img: 读入后的图片
    :return:投影变换后的图片
    '''
    height, width, channels = img.shape
    random_margin1 = height/4
    random_margin2 = width/4
    '''随机取四对点构成投影矩阵'''
    x1 = np.random.randint(0, random_margin1)
    y1 = np.random.randint(0, random_margin2)
    x2 = np.random.randint(width - random_margin1 - 1, width - 1)
    y2 = np.random.randint(0, random_margin2)
    x3 = np.random.randint(width - random_margin1 - 1, width - 1)
    y3 = np.random.randint(height - random_margin2 - 1, height - 1)
    x4 = np.random.randint(0, random_margin1)
    y4 = np.random.randint(height - random_margin2 - 1, height - 1)

    dx1 = np.random.randint(0, random_margin1)
    dy1 = np.random.randint(0, random_margin2)
    dx2 = np.random.randint(width - random_margin1 - 1, width - 1)
    dy2 = np.random.randint(0, random_margin2)
    dx3 = np.random.randint(width - random_margin1 - 1, width - 1)
    dy3 = np.random.randint(height - random_margin2 - 1, height - 1)
    dx4 = np.random.randint(0, random_margin1)
    dy4 = np.random.randint(height - random_margin2 - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)#投影矩阵
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return img_warp
if __name__=='__main__':
    image = cv2.imread('pp.jpg')#读入图片
    image_crop = random_image_crop(image)#裁剪
    image_color_shift = random_color_shift(image)#颜色变换
    image_rotation = random_rotation(image,1)#旋转
    image_perspective_transform = random_warp(image)#投影变换
    '''显示图片'''
    cv2.imshow('image',image)
    cv2.imshow('image_crop',image_crop)
    cv2.imshow('image_color_shift',image_color_shift)
    cv2.imshow('image_rotatin',image_rotation)
    cv2.imshow('image_perspective_transform ',image_perspective_transform )
    key = cv2.waitKey()
    if key ==27:
        cv2.destroyAllWindows()
