import os
import numpy as np
import pandas as pd

dataDir = 'data/data/'
valDir = 'data/val/'

aoa_lst = []
cl_lst = []
for data in os.listdir(dataDir):
    aoa = data.split("AOA_")[1].split(".npz")[0]
    aoa_lst.append(int(aoa))
    content = np.load(os.path.join(dataDir, data))
    cl_lst.append(float(content["CL"]))

aoa_lst.sort()
cl_lst.sort()

content = pd.read_csv(os.path.join(valDir, "CL_alpha_M7.csv"), names=['Alpha', 'CL'])
val_aoa = content['Alpha']
val_cl = content['CL']

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
plt.scatter(val_aoa, val_cl, edgecolor='red', facecolor='none', s=50, linewidths=2.5, label='Exp', marker='s')
plt.plot(aoa_lst, cl_lst, color='blue', label='CFD', marker='v', markeredgecolor='blue', markerfacecolor='blue', markersize=8)

plt.legend()
plt.ylabel(r"$C_L$")
plt.xlabel(r"$\alpha$ (degree)")
plt.grid()
plt.savefig("CL vs alpha.png")