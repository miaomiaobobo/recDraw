import cv2
import numpy as np
import math
import sys, os
from PIL import Image, ImageDraw
import heapq
#轮廓框取
image = cv2.imread('picture\\13.JPG')
image=cv2.resize(image,(1000,800))
#灰度处理
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#去噪
img = cv2.fastNlMeansDenoising(img,None,10,7,21)
#记录灰度
gray=0#用于记录点击某一点的灰度
#记录鼠标点击的位置
#mouseX=0
#mouseY=0
#a=[]
#b=[]

#画矩形框，dewa_Rec（已经去噪灰度图，彩色图，鼠标点击的x坐标，鼠标点击的y坐标）
def draw_Rec(img,image,mouseX,mouseY):
    #对图片进行画框处理
    #进行黑白处理
    ret, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
    #对黑白后的图像检测轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    listx=[]
    listy=[]
    listw=[]
    listh=[]
    listGap=[]
    i=0
    for c in contours:
        # 找到边界坐标
        x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
        print("x:{}，y:{} ； 长：{}，宽：{}".format(x ,y ,w ,h))
        xCentre=int(x+0.5*w)
        yCentre=int(y+0.5*h)
        #两点之间距离
        p1 = np.array([mouseX, mouseY])
        p2 = np.array([xCentre, yCentre])
        p3 = p2 - p1
        p4 = math.hypot(p3[0], p3[1])
        # 限定矩形的大小（w:长；h:宽）
        if w>50 and h>50:
            listx.append(x)
            listy.append(y)
            listw.append(w)
            listh.append(h)
            listGap.append(p4)
            print('第{}个 符合条件矩形与鼠标点击处距离：'.format(i),p4)
            i=i+1
    #找出距离最小值在距离列表中的索引
    temp = map(listGap.index, heapq.nsmallest(1, listGap))
    temp = list(temp)
    n=listGap.index(min(listGap))
    print('第{}个符合条件的矩形为最小距离'.format(n))
    print('最小距离为:',listGap[n])

    #去拿到距离最近的方框坐标
    x=listx[n]
    y=listy[n]
    w=listw[n]
    h=listh[n]
    print('最小距离矩形框x:{}，y:{} ； 长：{}，宽：{}'.format(x ,y ,w ,h))
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    '''
        # 找面积最小的矩形
        rect = cv2.minAreaRect(c)
        # 得到最小矩形的坐标
        box = cv2.boxPoints(rect)
        # 标准化坐标到整数
        box = np.int0(box)
        # 画出边界
        #cv2.drawContours(img, [box], 0, (0, 0, 255), 3)
    '''
    cv2.imshow("image",image)
    cv2.imwrite("result.jpg", image)

    cv2.waitKey(0)


def on_EVENT_LBUTTONDOWN(event, x, y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        draw_Rec(img,image,x,y)
        #a.append(x)
        #b.append(y)
        #cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        #cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
        #           1.0, (0, 0, 0), thickness=1)
        #cv2.imshow("image", image)

cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", image)
cv2.waitKey(0)
#print(a[0],b[0])
#mouseX=a[0]
#mouseY=b[0]
#a=[]
#b=[]
