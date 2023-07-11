# plot 100 new points of the database load from UIUC
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

f = 'data0'
b = np.loadtxt(f, skiprows=1)
b1=np.insert(b, [1,-1], [(1-1*0.001),b[0,1]], axis=0) # add two points in the data make the spline good

imgName=f'{f.split(".")[0]}_spline.png'
imgTestName=f'{f.split(".")[0]}_splineTest.png'
file=f'{f.split(".")[0]}'

# split b1 to x-dir and y-dir
b0=np.split(b1, 2, 1)
b1x=b0[0]
b1y=b0[1]

# print the database points from UIUC and plot the spline
list1=b1
list_to_tuple = tuple(list1)

QUADRUPLE_SIZE: int=4

def num_segments(points: tuple)-> int:
        return len(points)-(QUADRUPLE_SIZE-1)
        # return len(points)

def catmull_rom_spline(
    P0: tuple,
    P1: tuple,
    P2: tuple,
    P3: tuple,
    num_points: int,
    alpha: float=0,
):

    def tj(ti: float, pi: tuple, pj: tuple)-> float:
        xi, yi = pi
        xj, yj = pj
        dx, dy = xj-xi, yj-yi
        1+(dx**2+dy**2)**0.5
        return ti+1**alpha
    
    t0: float=0.0
    t1: float=tj(t0,P0,P1)
    t2: float=tj(t1,P1,P2)
    t3: float=tj(t2,P2,P3)
    t=np.linspace(t1,t2, num_points).reshape(num_points, 1)

    A1=(t1-t)/(t1-t0)*P0+(t-t0)/(t1-t0)*P1
    A2=(t2-t)/(t2-t1)*P1+(t-t1)/(t2-t1)*P2
    A3=(t3-t)/(t3-t2)*P2+(t-t2)/(t3-t2)*P3
    B1=(t2-t)/(t2-t0)*A1+(t-t0)/(t2-t0)*A2
    B2=(t2-t)/(t2-t1)*A2+(t-t1)/(t2-t1)*A3
    points=(t2-t)/(t2-t1)*B1+(t-t1)/(t2-t1)*B2
    return points
    
def catmull_rom_chain(points: tuple, num_points: int)-> list:
    point_quadruples=(
        (points[idx_segment_start +d] for d in range(QUADRUPLE_SIZE))
        for idx_segment_start in range(num_segments(points))
        )

    all_spline=(catmull_rom_spline(*q, num_points)for q in point_quadruples ) 

    chain=[chain_point for spline in all_spline for chain_point in spline ]
    return chain

if __name__=="__main__":
    POINTS: tuple=list_to_tuple
    NUM_POINTS: int=50

    chain_points: list=catmull_rom_chain(POINTS, NUM_POINTS)
    assert len(chain_points)==num_segments(POINTS)*NUM_POINTS
    plt.figure()
    plt.plot(*zip(*chain_points), c='blue')
    plt.plot(*zip(*POINTS), c='red', linestyle='none', marker='o')
    plt.grid(True)
    # plt.savefig(imgName)


    # split the array of chain_pointsp
    c=np.array_split(chain_points,2 ,1)
    # del array([,shape=(6000, 0)].....)
    d=c[0]   
    # print(d)                
    # 將d array 分割成兩項(分成x項與y項)
    d1=np.split(d, 2, 1)     
    xc=d1[0]   
    # print(xc)              
    yc=d1[1] 
    # 將xc再分成兩份(分成上下表面)              
    xc=np.split(xc, 2, 0)
    xc1=xc[0]
    xc2=xc[1]
    # print(xc1)
    # xc分成兩份，yc也要兩份
    yc=np.split(yc, 2, 0)
    yc1=yc[0]
    yc2=yc[1]                        

    # define range(list) of x(topper surface of airfoil)
    x=np.arange(0.97, 0.09, -0.03)
    x11=np.arange(0.1, 0, -0.005)
    # print(x[0])
    # define range of x0(lower surface of airfoil)
    x0=np.arange(0, 0.1, 0.005)
    x01=np.arange(0.1, 1, 0.03)
    # y1 mean topper surface of airfoil
    y1=np.zeros((30, 1))
    y11=np.zeros((20,1)) 
    # y2 mean lower surface of airfoil
    y2=np.zeros((20, 1)) 
    y21=np.zeros((31, 1))

    # def the new points of each database
    coords= np.zeros((101, 2))
    coords[0]=[b1x[0],b1y[0]]
                   

# plot the new point of topper surface of airfoil
    for i in range(len(x)):
        if  x[i] in xc1:              
            result=np.where(xc1==x[i])
            # print(result)
            if (result[0].shape[0])==2:
                result=result[0][0]
            elif(result[0].shape[0])==1:
                if (result[0][0])>(int((xc1.shape[0]))):
                    result=(xc1.shape[0])-result[0][0]-1
                else:
                    result=result[0][0]
            else:
                result=result[0][0]
            y1[i]=yc1[result]
        else:
            temp=xc1[-1][0]
            for x1 in xc1:
                # print(x1)
                x1 = x1[0]
                if x1>x[i]:
                    temp=x1
                else:
                    result1=np.where(xc1==x1)
                    result2=np.where(xc1==temp)
                    # print(result1[0], result2[0])
                    if (result1[0].shape[0])==1:
                        result1=result1[0][0]
                    elif (result1[0].shape[0])==2:    
                        result1=result1[0][0]
                    else:
                        result1=result1[0][0]

                    if (result2[0].shape[0])==1:
                        result2=result2[0][0]
                    elif (result2[0].shape[0])==2:
                        result2=result2[0][0]
                    else:
                        result2=result2[0][0]
                    y1[i]=(yc1[result1])-((yc1[result1]-yc1[result2])*((x1-x[i])/(x1-temp)))                     
                    break
        plt.plot(x[i], y1[i], 'y-o')
        coords[i+1] = [round(x[i], 5), round(y1[i][0], 5)]

    for i in range(len(x11)):
        if  x11[i] in xc1:              
            result=np.where(xc1==x11[i])
            # print(result)
            if (result[0].shape[0])==2:
                result=result[0][0]
            elif(result[0].shape[0])==1:
                if (result[0][0])>(int((xc1.shape[0]))):
                    result=(xc1.shape[0])-result[0][0]-1
                else:
                    result=result[0][0]
            else:
                result=result[0][0]
            y11[i]=yc1[result]
        else:
            temp=xc1[-1][0]
            for x1 in xc1:
                # print(x1)
                x1 = x1[0]
                if x1>x11[i]:
                    temp=x1
                else:
                    result1=np.where(xc1==x1)
                    result2=np.where(xc1==temp)
                    # print(result1[0], result2[0])
                    if (result1[0].shape[0])==1:
                        result1=result1[0][0]
                    elif (result1[0].shape[0])==2:    
                        result1=result1[0][0]
                    else:
                        result1=result1[0][0]

                    if (result2[0].shape[0])==1:
                        result2=result2[0][0]
                    elif (result2[0].shape[0])==2:
                        result2=result2[0][0]
                    else:
                        result2=result2[0][0]
                    y11[i]=(yc1[result1])-((yc1[result1]-yc1[result2])*((x1-x11[i])/(x1-temp)))                     
                    break
        plt.plot(x11[i], y11[i], 'y-o')
        coords[i+30] = [round(x11[i], 5), round(y11[i][0], 5)]
        # print(coords)

# plot the new point of lower surface of airfoil
    for j in range(len(x0)):
        if  x0[j] in xc2:              
            result4=np.where(xc2==x0[j])
            if (result4[0].shape[0])==2:
                if (result4[0][0])>(int((xc2.shape[0]))):
                    result4=(xc2.shape[0])-result4[0][1]-1
                else:
                    result4=result4[0][0]
                y2[j]=yc2[result4]
            elif(result4[0].shape[0])==1:
                if (result4[0][0])>(int((xc2.shape[0]))):
                    result4=(xc2.shape[0])-result4[0][0]-1
                else:
                    result4=result4[0][0]
                y2[j]=yc2[result4]
            else:
                y2[j]=yc2[result4[0][0]]
        else:
            temp=xc2[1][0]
            for x2 in xc2:
                x2 = x2[0]
                if x2<x0[j]:
                    temp=x2
                else:
                    result5=np.where(xc2==x2)
                    result6=np.where(xc2==temp)
                    if (result5[0].shape[0])==1:
                        result5=result5[0][0]
                    elif (result5[0].shape[0])==2:    
                        result5=result5[0][0]
                    else:
                        result5=result5[0][0]

                    if (result6[0].shape[0])==1:
                        result6=result6[0][0]
                    elif (result6[0].shape[0])==2:
                        result6=result6[0][1]
                    else:
                        result6=result6[0][1]
                    y2[j]=(yc2[result5])-((yc2[result5]-yc2[result6])*((x2-x0[j])/(x2-temp)))                      
                    break
        plt.plot(x0[j], y2[j], 'y-o')
        coords[j+50] = [round(x0[j], 5), round(y2[j][0], 5)]

    for j in range(len(x01)):
        if  x01[j] in xc2:              
            result4=np.where(xc2==x01[j])
            if (result4[0].shape[0])==2:
                if (result4[0][0])>(int((xc2.shape[0]))):
                    result4=(xc2.shape[0])-result4[0][1]-1
                else:
                    result4=result4[0][0]
                y21[j]=yc2[result4]
            elif(result4[0].shape[0])==1:
                if (result4[0][0])>(int((xc2.shape[0]))):
                    result4=(xc2.shape[0])-result4[0][0]-1
                else:
                    result4=result4[0][0]
                y21[j]=yc2[result4]
            else:
                y21[j]=yc2[result4[0][0]]
        else:
            temp=xc2[1][0]
            for x2 in xc2:
                x2 = x2[0]
                if x2<x01[j]:
                    temp=x2
                else:
                    result5=np.where(xc2==x2)
                    result6=np.where(xc2==temp)
                    if (result5[0].shape[0])==1:
                        result5=result5[0][0]
                    elif (result5[0].shape[0])==2:    
                        result5=result5[0][0]
                    else:
                        result5=result5[0][0]

                    if (result6[0].shape[0])==1:
                         result6=result6[0][0]
                    elif (result6[0].shape[0])==2:
                        result6=result6[0][1]
                    else:
                        result6=result6[0][1]
                    y21[j]=(yc2[result5])-((yc2[result5]-yc2[result6])*((x2-x01[j])/(x2-temp)))                      
                    break
        plt.plot(x01[j], y21[j], 'y-o')
        coords[j+70] = [round(x01[j], 5), round(y21[j][0], 5)]
        coords[-1]=[b1x[-1],b1y[-1]]
    np.save(file, coords)




    plt.plot(b1x[-1],b1y[-1], 'y-o')
    # plt.plot(x0[j], y2[j], 'y-o')
    plt.xlabel("x")
    plt.ylabel("y[i]")
    # plt.savefig(imgTestName)
    # plt.show()

            
            