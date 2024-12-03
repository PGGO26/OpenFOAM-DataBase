import os
import pandas as pd
import numpy as np

v = 223.6688
p_inf = 94410
rho = 1.3613

# os.system("postProcess -latestTime -func boundaryCloud > LOG.log")

# BoundaryCloud
file_path = 'postProcessing/boundaryCloud/1200/surface_p.xy'
content = pd.read_csv(file_path, delimiter='\s+', skiprows=0, header=None, names=['x','y','z','p']).round(5)

x = content['x']
y = content['y']
z = content['z']
p = content['p']

normalized_x = (x + 0.15) / 0.8
normalized_y = (y + 0.4) / 0.8
normalized_p = (p - p_inf) / (0.5 * rho * v**2)

res = 512
Output = np.full((res, res), np.nan)
Output[(normalized_x * (res-1)).astype(int), (normalized_y * (res-1)).astype(int)] = normalized_p

field = np.copy(Output)
field = np.flipud(field.transpose())

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.figure(figsize=(8,6))

colors = ["#0000F0", "#0073FF", "#00FDFB", "#00FF8A", "#01FF07", "#89FF00", "#FAFF00", "#FF7900", "#FF0000"]
n_bins = 50
cmap_name = "SACCON"
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
plt.imshow(field, cmap=custom_cmap, vmin=-1.5, vmax=0.2, interpolation='nearest')
plt.colorbar(label='Cp')
plt.title('Upper surface Pressure distribution')
plt.xlabel('X pixel')
plt.ylabel('Y pixel')
plt.savefig('Global result.png')

# # InternalCloud
# file_path = 'postProcessing/internalCloud/1200/AIP_p_U.xy'
# content = pd.read_csv(file_path, delimiter='\s+', skiprows=0, header=None, names=['x','y','z','p','u','v','w']).round(5)

# x = content['x']
# y = content['y']
# z = content['z']
# p = content['p']
# u = content['u']

# normalized_x = (x + 1) / 3
# normalized_z = (z + 1.5) / 3
# normalized_p = (p - p_inf) / (0.5 * rho**2 * v)

# res = 1024
# Output = np.zeros((res, res))
# Output[(normalized_x * (res-1)).astype(int), (normalized_z * (res-1)).astype(int)] = normalized_p

# field = np.copy(Output)
# field = np.flipud(field.transpose())
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8,6))
# plt.imshow(field, cmap='jet', interpolation='nearest')
# plt.colorbar(label='value')
# plt.title('Local Feature')
# plt.xlabel('X pixel')
# plt.ylabel('Y pixel')
# plt.savefig('Result.png')