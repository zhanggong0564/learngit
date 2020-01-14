import cv2
import numpy as np

def chang_color(img):
    Y,U,V = cv2.split(img)
    Y_rand = np.random.randint(-30,30)
    if Y_rand ==0:
        pass
    elif Y_rand>0:
        de = 255-Y_rand
        Y[Y>de]=255
        Y[Y<=de] = (Y[Y<=de]+Y_rand).astype(img.dtype)
    elif Y_rand<0:
        de = 0-Y_rand
        Y[Y>de] = (Y[Y>de] + Y_rand).astype(img.dtype)
        Y[Y<=de] = 0
    
    U_rand = np.random.randint(-30,30)
    if U_rand ==0:
        pass
    elif U_rand>0:
        de = 255-U_rand
        U[U>de]=255
        U[U<=de] = (U[U<=de]+U_rand).astype(img.dtype)
    elif U_rand<0:
        de = 0-U_rand
        U[U>de] = (U[U>de] + U_rand).astype(img.dtype)
        U[U<=de] = 0
    V_rand = np.random.randint(-30,30)
    if V_rand ==0:
        pass
    elif V_rand>0:
        de = 255-V_rand
        V[V>de]=255
        V[V<=de] = (V[V<=de]+V_rand).astype(img.dtype)
    elif V_rand<0:
        de = 0-V_rand
        V[V>de] = (V[V>de] + V_rand).astype(img.dtype)
        V[V<=de] = 0
    change_img = cv2.merge((Y,U,V))
    return change_img 

if __name__ == '__main__':
    #读入图片
    img = cv2.imread('imge/pp.jpg')
    #转换YUV
    img_YUV = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    #改变YUV颜色
    change_imge = chang_color(img_YUV)
    #将YUV转换回RGB
    chang_BGR = cv2.cvtColor(change_imge,cv2.COLOR_YUV2BGR)
    #显示改变后的YUV图片
    cv2.imshow('change_YUV',change_imge)
    #显示改变后的RGB图像
    cv2.imshow('BGR',chang_BGR)
    #显示原始图像
    cv2.imshow('src',img)
    key = cv2.waitKey()
    if key==27:
        cv2.destroyAllWindows()
