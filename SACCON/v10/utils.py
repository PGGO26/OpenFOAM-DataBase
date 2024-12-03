import os, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def write_pts(mode="boundary_upper", x_min=-0.15, xy_range=0.8, res=512, z_range=0.05, layer=11):
    x_min, x_max = x_min, x_min + xy_range
    y_min, y_max = -xy_range/2, xy_range/2

    x_values = np.linspace(x_min, x_max, res)
    y_values = np.linspace(y_min, y_max, res)

    if mode == "boundary_upper":
        z_min, z_max = 0.0, z_range
        z_values = np.linspace(z_min, z_max, layer)
    elif mode == "boundary_lower":
        z_min, z_max = -z_range, 0.0
        z_values = np.linspace(z_min, z_max, layer)

    elif mode == "internal":
        z_min, z_max = -z_range, z_range
        y_values = np.linspace(y_max, y_max, layer)
        z_values = np.linspace(z_min, z_max, res)

    with open("system/include/pts", 'w') as file:
        file.write("pts\n(\n")
        for z in z_values:
            for y in y_values:
                for x in x_values:
                    file.write(f"({x} {y} {z})\n")
        file.write(");")
        print(f"Write the Surface points fille : system/include/pts.")

def outputPlot(output, fileName):
    field = np.copy(output)
    field = np.flipud(field.transpose())

    plt.figure(figsize=(8,6))
    plt.imshow(field, cmap='jet', interpolation='nearest')
    plt.colorbar(ticks=np.linspace(np.nanmin(field), np.nanmax(field), 12),label='Cp')
    plt.title('Pressure coefficient distribution')
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.savefig(fileName)
    print("Saving plot in ", fileName)

def outputProcess(mode="boundaryProbes", res=256, surface="Upper", x_min=-0.15, xy_range=0.8, pinf=94410, rho=1.3613, v=223.6688):
    # Finding postProcess file path
    post_path = os.path.join("postProcessing/", mode)
    final_step = max(step for step in os.listdir(post_path))
    file_path = os.path.join(post_path, str(final_step), "points.xy")

    if mode == "boundaryProbes":
        content = pd.read_csv(file_path, delimiter='\s+', comment="#", skiprows=0, header=None, names=['x','y','z','p']).round(5)
    elif mode == "internalProbes":
        content = pd.read_csv(file_path, delimiter='\s+', comment="#", skiprows=0, header=None, names=['x','y','z','p','u','v','w']).round(5)

    x = content['x'][1:].astype(float)
    y = content['y'][1:].astype(float)
    z = content['z'][1:].astype(float)
    p = content['p'][1:].astype(float)

    normalized_x = (x -x_min) / xy_range
    normalized_y = (y + xy_range/2) / xy_range
    normalized_p = (p - pinf) / (0.5 * rho * v**2)

    Output = np.full((res, res), np.nan)
    Output[(normalized_x * (res-1)).astype(int), (normalized_y * (res-1)).astype(int)] = normalized_p
    outputPlot(Output, fileName=f"{surface}_pressure.png")

def outputProcess_Coeffs(Output_DIR, Case_DIR, FileName):
    time_step = os.listdir("OpenFOAM/postProcessing/forceCoeffsCompressible/")[-1]
    with open(f"OpenFOAM/postProcessing/forceCoeffsCompressible/{time_step}/forceCoeffs.dat") as inFile:
        content = inFile.readlines()
    last_line = content[-1].strip().split()

    Cd = float(last_line[1])
    Cl = float(last_line[4])
    print(f"Final time : {time_step}, Cd : {Cd}, Cl : {Cl}")
    npz_PATH = os.path.join(Output_DIR, FileName)
    np.savez_compressed(npz_PATH, Time=time_step, CL=Cl, CD=Cd)

    case_PATH = os.path.join(Case_DIR, FileName)
    if os.path.exists(case_PATH):
        shutil.rmtree(case_PATH)
    shutil.copytree(f"OpenFOAM/{time_step}/", case_PATH)
    return npz_PATH