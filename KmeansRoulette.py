import cv2,random
import numpy as np
import math,time,sys

DEBUG = True
#sys.setrecursionlimit = 200000
order = [[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1]]
prev = [1,0]

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
    #median -> 23 low
    return closing
###closing kernel using morphological transform, not used
def Closing(closing):
    rect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    np.array([[1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1]], dtype=np.uint8)

    closing = cv2.dilate(closing,rect,iterations = 10)
    closing = cv2.erode(closing,rect,iterations = 10)
    cv2.imshow("dialate",closing)
    return closing
    
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
    ###how wide space between pix are
    loopjump = 40
    
    
    greenPoints=[]
    xt = 0
    yt = 0
    c = 0
    xleft = 2000000
    yleft = 0
    ###sector vars(possibly unimportant)
    r = 0
    g=0
    b=0
    ###Debug vars
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
                
                greenPoints.append([x*loopjump,y*loopjump])
                
                ###possibly unimportant
                b = img[y*loopjump][x*loopjump][0]
                g = img[y*loopjump][x*loopjump][1]
                r = img[y*loopjump][x*loopjump][2]
                
                ###use seperate image to place dots on
                #if(DEBUG):
                    #img[y*loopjump][x*loopjump][2] =255
                if(x*loopjump>467):
                    rightcounter +=1
                else:
                    leftcounter +=1
                if(x*loopjump<xleft):
                    xleft =x*loopjump
                    yleft = y*loopjump
                c = c+1
                
    radius = int(math.sqrt((xleft-xt/c)*(xleft-xt/c)+(yleft-yt/c)*(yleft-yt/c)))
    #cv2.circle(img,(xt/c,yt/c),radius,(255,255,255))
    #cv2.circle(img,(xt/c,yt/c),5,(255,255,255))
    if(DEBUG):
        print "coordinate center : ", xt/c,yt/c
        print "green pixels counted to right then left : ", rightcounter, leftcounter
    #return img,xt/c,yt/c,radius,b,g,r
    return img, xt/c, yt/c, greenPoints

def sectors(img, xc, yc, points):
    ###mebbe 3 -> 1 grayscale faster process?
    sector = np.zeros((len(img),len(img[0]),3), np.uint8)

    left = True
    right = True
    
    
    ###line equations to seperate two sectors
    if(DEBUG):
        print "---line---"
        print points[0]
        print xc,yc
        print -1/(float(points[0][1]-yc)/float(points[0][0]-xc))
        print yc-(-1/(float(points[0][1]-yc)/float(points[0][0]-xc)))*xc
        print points
        print "---line---"    
    m = -1/(float(points[0][1]-yc)/float(points[0][0]-xc)) 
    c = int(yc-m*xc)

    ###! delete 2 lines below
    #cv2.line(img,(307,720),(547,0),(255,255,255),1)
    #cv2.line(img,(int(((len(img)-1)-c)/m),(len(img)-1)),(len(img[0])-1,int((len(img[0])-1)*m)-c),(255,255,255),1)
    for i in range(len(points)):
        
        if(left and points[i][1]>points[i][0]*m+c):
            left = False
            borderPath(img, sector, points[i][0], points[i][1])
        
        if(right and points[i][1]<points[i][0]*m+c):
            right= False
            borderPath(img, sector, points[i][0], points[i][1])
        #fill(img,sector,points[x][0], points[x][1],img[points[x][1]][points[x][0]])
    print right, left
    return img,sector, xc,yc

def borderPath(img, canvas,x,y):
    ###! note: left side must not be edge
    while(img[y][x-1][1] >= int(img[y][x-1][2]*0.5)+int(img[y][x-1][0])):
        x-=1
    grace = 100
    exit = False
    xst = x
    yst = y
    contour = []

    while(exit ==False or grace >0):
    #while(grace >0):
        #print exit, grace
        #print "x: ", xst, x
        #print "y: ", yst, y
        #print "abs: ", abs(xst-x), abs(yst-y)
        if(grace >0):
            grace -=1

        if(abs(xst-x) <3 and abs(yst-y)<3):
            exit = True
        else:
            exit = False
        index = order.index(prev)
        for i in range((index+1), (index+9)):
            point = img[y+order[i%8][1]][x+order[i%8][0]]
            if(point[1] > int(point[2]*0.5)+int(point[0])):
                ###test line
                img[y+order[i%8][1]][x+order[i%8][0]][1] = int(img[y+order[i%8][1]][x+order[i%8][0]][0])+int(img[y+order[i%8][1]][x+order[i%8][0]][2]*0.5)
                canvas[y+order[i%8][1]][x+order[i%8][0]] = [255,255,255]
                prev[0] = order[i%8][0]
                prev[1] = order[i%8][1]
                prev[0]=prev[0]*-1
                prev[1]=prev[1]*-1
                x=x+order[i%8][0]
                y=y+order[i%8][1]
                contour.append([x+order[i%8][0],y+order[i%8][1]])
                break
            
            if(point[1] == int(point[2]*0.5)+int(point[0])):
                img[y+order[i%8][1]][x+order[i%8][0]][1]-=1
                canvas[y+order[i%8][1]][x+order[i%8][0]] = [255,255,255]
                prev[0] = order[i%8][0]
                prev[1] = order[i%8][1]
                prev[0]=prev[0]*-1
                prev[1]=prev[1]*-1
                x=x+order[i%8][0]
                y=y+order[i%8][1]
                break
    ctr = np.array(contour).reshape((-1,1,2)).astype(np.int32)
    fit = cv2.fitEllipse(ctr)
    canvas = Closing(canvas)
    cv2.ellipse(canvas,fit,(0,255,0),2,cv2.LINE_AA)

###not used
def detectWhite(canvas,x,y):
    colorpixs = 0
    for i in range(8):
        if(order[i][0]+x >=0 and order[i][0]+x < len(canvas[0]) and order[i][1]+y >=0 and order[i][1]+y < len(canvas)):
           if(canvas[y+order[i][1]][x+order[i][0]][0] == [255]):
               colorpixs +=1
    return colorpixs
           
###Keeps exceeding recursion limit, switching to border detection
def fill(image, canvas, x, y,color):
    #if(all(canvas[y][x] != [255,255,255])): print True
    if(canvas[y][x][0] != 255 and image[y][x][0] == color[0] and image[y][x][1] == color[1] and image[y][x][2] == color[2]):
        print canvas[y][x]
        canvas[y][x] = [255,255,255]
        
        if(x<len(image[0])-1): fill(image, canvas, x+1, y,color)
        if(x>0):               fill(image, canvas, x-1, y,color)
        if(y<len(image)-1):    fill(image, canvas, x, y+1,color)
        if(y>0):               fill(image, canvas, x, y-1,color)
    
    

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
    center,x,y,greenPoints = findCircleCenter(show)
    img,sector, x,y = sectors(center,x,y,greenPoints)
    
    #findSectors(center,x,y,radius,b,g,r)
    end = time.time()
    if(DEBUG):
        print "time : ", end-start

    #cv2.imshow("original", img)
    cv2.imshow("close",close)
    cv2.imshow("thing", show)
    cv2.imshow("circle",center)
    cv2.imshow("sector",sector)
    #cv2.imshow("final",)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
