import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def interpolate_point(lst_x, lst_shear, xpoint):
    n = len(lst_x)
    if xpoint < lst_x[0] or xpoint > lst_x.iloc[-1]:
        lst_x[0] = lst_shear[0]
        lst_x.iloc[-1] = lst_shear.iloc[-1]

    for i in range(n-1):
        if xpoint >= lst_x[i] and xpoint <= lst_x.iloc[i+1]:
            y = linear_interpolation(lst_x[i], lst_shear[i], lst_x[i+1], lst_shear[i+1], xpoint)

    return y

def linear_interpolation(x1, x2, y1, y2, x):
    y = y1 + (y2 - y1)/(x2 - x1) * (x - x1)
    return y

upperdf = pd.read_csv('temps/shear_upper.csv')
bottomdf = pd.read_csv('temps/shear_bottom.csv')

upperlst_x = upperdf['x']
upperlst_shear = upperdf['shear stress']
bottomlst_x = bottomdf['x']
bottomlst_shear = bottomdf['shear stress']

# print(upperlst_x.iloc[-1])

xp_front = np.arange(0.005, 0.1, 0.005)
xp_back = np.arange(0.1, 0.99, 0.03)

with open('temps/shear', 'wt') as of:
    of.write('x' + ',' + 'shear stress' '\n')
    of.write('0' + ', ' + '0' '\n')
    for i in range(len(xp_front)):
        line = str(round(xp_front[i], 5))
        y = interpolate_point(upperlst_x, upperlst_shear, xp_front[i])
        line1 = str(round(y, 5))
        of.write(line + ', ' + line1 + '\n')
    
    for i in range(len(xp_back)):
        line = str(round(xp_back[i], 5))
        y = interpolate_point(upperlst_x, upperlst_shear, xp_back[i])
        line1 = str(round(y, 5))
        of.write(line + ', ' + line1 + '\n')

    linex0 = str(1)
    liney0 = str(round(upperlst_shear.iloc[-1], 5))
    of.write(linex0 + ', ' + liney0 + '\n')

    for i in range(len(xp_front)):
        line = str(round(xp_front[i], 5))
        y = interpolate_point(bottomlst_x, bottomlst_shear, xp_front[i])
        line1 = str(round(y, 5))
        of.write(line + ', ' + line1 + '\n')

    for i in range(len(xp_back)):
        line = str(round(xp_back[i], 5))
        y = interpolate_point(bottomlst_x, bottomlst_shear, xp_back[i])
        line1 = str(round(y, 5))
        of.write(line + ', ' + line1 + '\n')
    linex1 = str(1)
    liney1 = str(round(bottomlst_shear.iloc[-1], 5))
    of.write(linex0 + ', ' + liney0 + '\n')
