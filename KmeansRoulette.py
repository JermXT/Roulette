import cv2,random
import numpy as np
import math,time

def avgBlueIntensity(img):
    test = int(len(img)*len(img[0])*0.1)
    r=0
    g=0
    b=0
    count = 0;

    for p in range(test):
        pix = img[random.randint(0,len(img)-1)][random.randint(0,len(img[0])-1)]
        if(pix[0]*0.8>pix[1] and pix[0]*0.8>pix[2]):
            count = count+1;
            b = pix[0]+b
            g = pix[1]+g
            r = pix[2] +r
    b = b/count
    r = r/count
    g = g/count
    return b,g,r

def Blur(img):
    closing = cv2.GaussianBlur(img,(5,5),10)
    closing = cv2.medianBlur(img,23)

    return closing
    ###closing kernel using morphological transform, not used
    '''
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    np.array([[0, 0, 1, 0, 0],
              [0, 1, 1, 1, 0],
              [1, 1, 1, 1, 1],
              [0, 1, 1, 1, 0],
              [0, 0, 1, 0, 0]], dtype=np.uint8)
    for x in range(10):
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, ellipse)
    
    return closing
    '''
def Scan(img):
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ###amount of colors in img, use at least 6 colors and not reccomended over 16
    K = 16
    #K = 8
    #K = 6
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

def findCircleCenter(img):
    loopjump = 40
    ###how wide space between pix are
    xt = 0
    yt = 0
    c = 0
    xleft = 2000000
    yleft = 0
    ###sector vars
    r = 0
    g=0
    b=0
    ###debug counter
    leftcounter=0
    rightcounter=0

    for y in range(len(img)/loopjump):
        for x in range(len(img[0])/loopjump):
            ### may need to multiply green by a bit
            #if(img[y*10][x*10][1] > 100):
                
            if(img[y*loopjump][x*loopjump][1] >=int(img[y*loopjump][x*loopjump][2])*0.5+int(img[y*loopjump][x*loopjump][0])):
                #print img[y*10][x*10][1],img[y*10][x*10][2],img[y*10][x*10][0],int(img[y*10][x*10][2])+int(img[y*10][x*10][0])
                #img[y*10][x*10][1]=0
                xt = xt + x*loopjump
                yt = yt +y*loopjump
                b = img[y*loopjump][x*loopjump][0]
                g = img[y*loopjump][x*loopjump][1]
                r = img[y*loopjump][x*loopjump][2]
                img[y*loopjump][x*loopjump][2] =255
                if(x*loopjump>467):
                    rightcounter +=1
                else:
                    leftcounter +=1
                if(x*loopjump<xleft):
                    xleft =x*loopjump
                    yleft = y*loopjump
                c = c+1
                
    radius = int(math.sqrt((xleft-xt/c)*(xleft-xt/c)+(yleft-yt/c)*(yleft-yt/c)))
    cv2.circle(img,(xt/c,yt/c),radius,(255,255,255))
    cv2.circle(img,(xt/c,yt/c),5,(255,255,255))
    print "coordinate center : ", xt/c,yt/c
    print "green pixels counted to right then left : ", rightcounter, leftcounter
    return img,xt/c,yt/c,radius,b,g,r
    

    

def findSectors(img,x,y,radius,b,g,r):
    colorDict = {}
    for a in range(90):
        xc = int(round(math.cos(a*4)*radius/2))
        yc = int(round(math.sin(a*4)*radius/2))
        print tuple(img[y+yc][x+xc])
        if(tuple(img[y+yc][x+xc]) in colorDict):
            colorDict[tuple(img[y+yc][x+xc])] += 1
        else:
            colorDict[tuple(img[y+yc][x+xc])] = 1
    print colorDict

def main():
    start = time.time()

    #img = cv2.imread("Images/downWheelS.png")
    #img = cv2.imread("Images/IMG_0840.jpg")
    img = cv2.imread("Images/IMG_0830.jpg")
  
    ###not used cus comp pool floor hella sus(not blue)
    #b,g,r = avgBlueIntensity(img)
    #print(b,g,r)
    
    close = Blur(img)
    show = Scan(close)
    center,x,y,radius,b,g,r = findCircleCenter(show)
    #findSectors(center,x,y,radius,b,g,r)
    end = time.time()
    print "time : ", end-start

    #cv2.imshow("original", img)
    cv2.imshow("close",close)
    cv2.imshow("thing", show)
    cv2.imshow("circle",center)
    #cv2.imshow("final",)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
