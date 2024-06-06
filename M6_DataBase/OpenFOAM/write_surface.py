import numpy as np
x_min, x_max = -0.5, 1.5
z_min, z_max = -1, 1
y_min, y_max = 0, 1
res = 128

x_values = np.linspace(x_min, x_max, res)
z_values = np.linspace(z_min, z_max, res)
y_values = np.linspace(y_min, y_max, res)

write_path_boundary = 'system/include/boundary'
with open(write_path_boundary, 'w') as file:
    file.write("pts\n(\n")
    for z in z_values:
        for y in y_values:
            y *= 1.1963
            for x in x_values:
                file.write(f"({x} {y} {z})\n")
    file.write(");")
print(f'Write the surface points in {write_path_boundary} file.')

y_values_AIP = [0.2, 0.65, 0.8, 0.9, 0.96]
res_AIP = 512

x_values_AIP = np.linspace(x_min, x_max, res)
z_values_AIP = np.linspace(z_min, z_max, res)
write_path_AIP = 'system/include/AIP'
with open(write_path_AIP, 'w') as file:
    file.write("pts\n(\n")
    for z in z_values_AIP:
        for y in y_values_AIP:
            y *= 1.1963
            for x in x_values_AIP:
                file.write(f"({x} {y} {z})\n")
    file.write(");")
print(f'Write the surface points in {write_path_AIP} file.')