# 使用跟 /mesh_test/mesh_test.py 相同的代碼，避免混淆
# 2025/04/09 優化 boundaryProbes 中的 mode 參數，能分成 "All", 及 "latestTime"

import os, time, shutil
import numpy as np
import pandas as pd

def runSim(freestream_x, freestream_z, P_inf, T_inf):
    os.chdir("OpenFOAM/")
    with open("0/include/template", "rt") as inFile:
        with open("0/include/freestreamConditions", "wt") as outFile:
            for line in inFile:
                line = line.replace("VEL_X", f"{freestream_x}")
                line = line.replace("VEL_Z", f"{freestream_z}")
                line = line.replace("P_INF", f"{P_inf}")
                line = line.replace("T_INF", f"{T_inf}")
                outFile.write(line)
    os.system("./Allclean")
    os.system("decomposePar > LOG.log")
    os.system("mpirun -np 16 hisa -parallel > HiSA.log")
    os.system("reconstructPar > LOG.log")
    os.system("rm -rf proc*")
    os.chdir("..")

def casebackup(output_dir="case", filename="MACH_0.7_AOA_8"):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    final_step = None
    for step in os.listdir("OpenFOAM/"):
        if step.isdigit():
            step = int(step)
            if final_step is None or step > final_step:
                final_step = step
            
    source_path = os.path.join("OpenFOAM/", f"{final_step}")
    destination_path = os.path.join(output_dir, f"{filename}_{final_step}")
    shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
    print(f"Backup step data into {destination_path}")

def wrtie_pts(xmin, xmax, ymin, ymax, zmin=-0.05, zmax=0.05, plane_res=[512, 512, 11]):
    if not os.path.exists("OpenFOAM/system/include/"):
        os.mkdir("OpenFOAM/system/include/")

    x_values = np.linspace(xmin, xmax, plane_res[0])
    y_values = np.linspace(ymin, ymax, plane_res[1])
    z_values = np.linspace(zmin, zmax, plane_res[2])

    with open('OpenFOAM/system/include/pts', 'w') as f:
        f.write("pts\n(\n")
        for z in z_values: 
            for y in y_values:
                for x in x_values:
                    f.write(f"({x} {y} {z})\n")
        f.write(");")

def forceCoeffsCompressible(MagU_inf, Lift_Dir, Drag_Dir):
    os.chdir("OpenFOAM/")
    print("Running post-Process : forceCoeffsCompressible :", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    with open("system/forceCoeffsCompressible_template", "rt") as inFile:
        with open("system/forceCoeffsCompressible", "wt") as outFile:
            for line in inFile:
                line = line.replace("MAGUINF", f"{MagU_inf}")
                line = line.replace("LIFT_DIR", Lift_Dir)
                line = line.replace("DRAG_DIR", Drag_Dir)
                outFile.write(line)

    os.system("hisa -postProcess -latestTime -func forceCoeffsCompressible > LOG.log")
    step_path = os.path.join("postProcessing/forceCoeffsCompressible/", os.listdir("postProcessing/forceCoeffsCompressible/")[-1])
    file_path = os.path.join(step_path, os.listdir(step_path)[0])
    with open(file_path, "rt") as inFile:
        content = inFile.readlines()
    last_line = content[-1].strip().split()
    Step = float(last_line[0])
    Cm = float(last_line[1])
    Cd = float(last_line[2])
    Cl = float(last_line[3])
    Clf = float(last_line[4])
    Clr = float(last_line[5])

    data_dict = {"Step":Step, "Cm":Cm, "Cd":Cd, "Cl":Cl, "Clf":Clf, "Clr":Clr}
    os.chdir("..")
    # print(f"Step : {Step}, Cm : {Cm}, Cd : {Cd}, Cl : {Cl}, Clf : {Clf}, Clr : {Clr}")
    return data_dict

def outputProcessing(step_path, xmin, xmax, ymin, ymax, zmin, zmax, res):
    mask_output = np.full((res, res), np.nan)
    p_output = np.zeros((res, res))
    shearstress_output = np.zeros((res, res))
    content = pd.read_csv(step_path, delimiter="\s+", skiprows=0, header=1, names=['x','y','z','p','p_x','p_y','p_z']).round(5)
    content['WallShearStress'] = np.sqrt(content['p_x']**2 + content['p_y']**2 + content['p_z']**2)
    x = content['x']
    y = content['y']
    norm_x = (x - xmin) / (xmax - xmin)
    norm_y = (y - ymin) / (ymax - ymin)

    mask_output[(norm_x * (res-1)).astype(int), (norm_y * (res-1)).astype(int)] = 0
    p_output[(norm_x * (res-1)).astype(int), (norm_y * (res-1)).astype(int)] = content['p']
    shearstress_output[(norm_x * (res-1)).astype(int), (norm_y * (res-1)).astype(int)] = content['WallShearStress']
    
    return mask_output, p_output, shearstress_output

def boundaryProbes(xmin, xmax, ymin, ymax, zmin, zmax, res, mode=None):
    wrtie_pts(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax, plane_res=[512, 512, 3])
    os.chdir("OpenFOAM/")
    if mode == "All":
        os.system("hisa -postProcess -func wallShearStress > LOG.log")
        os.system("postProcess -func boundaryProbes > LOG.log")
        os.chdir("..")
        step_files = [f for f in os.listdir("OpenFOAM") if f.isdigit() and int(f) != 0]

    elif mode == "latestTime":
        os.system("hisa -postProcess -latestTime -func wallShearStress > LOG.log")
        os.system("postProcess -latestTime -func boundaryProbes > LOG.log")
        os.chdir("..")
        step_files = [f for f in os.listdir("OpenFOAM/postProcessing/boundaryProbes/") if f.isdigit() and int(f) != 0]

    boundary_outputs = np.full((len(step_files), 3, res, res), np.nan)
    step_idx = 0
    for step in step_files:
        step_path = os.path.join("OpenFOAM/postProcessing/boundaryProbes/", step)
        file_path = os.path.join(step_path, os.listdir(step_path)[0])
        mask_output, p_output, shearstress_output = outputProcessing(file_path, xmin, xmax, ymin, ymax, zmin, zmax, res)
        boundary_outputs[step_idx][0] = mask_output
        boundary_outputs[step_idx][1] = p_output
        boundary_outputs[step_idx][2] = shearstress_output
        step_idx += 1

    return boundary_outputs

# MACH = [0.5] #dataGen TEST
MACH = [0.6, 0.7, 0.8]
AOA = [-5, 0, 5, 10]
RES = 256
OUTPUT_DIR = "data/"
CASE_DIR = "case/"

T_INF = 254.1
VEL = np.sqrt(1.4 * 287 * T_INF)
mu = 1.716E-5 * np.power((T_INF / 273), 1.5) * (273 + 110.4) / (T_INF + 110.4)
P_INF = 94410

X_MIN_MAX = [-0.15, 0.65]
Y_MIN_MAX = [-0.40, 0.40]
Zu_MIN_MAX = [0.050, 0.005]
Zl_MIN_MAX = [-0.005, -0.050]

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

for mach in MACH:
    for aoa in AOA:
        fileName = f"MACH_{mach}_AOA_{aoa}"
        print("*"*80, f"\nRunning case : {fileName}")

        vel = VEL * mach
        rad_aoa = np.deg2rad(aoa)
        fsX = vel * np.cos(rad_aoa)
        fsZ = vel * np.sin(rad_aoa)
        print("Running simulation :", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        runSim(freestream_x=fsX, freestream_z=fsZ, P_inf=P_INF, T_inf=T_INF)
        casebackup(output_dir=CASE_DIR, filename=fileName)

        lift_dir = f"({-1 * (np.sin(rad_aoa))} 0 {np.cos(rad_aoa)})"
        drag_dir = f"({np.cos(rad_aoa)} 0 {np.sin(rad_aoa)})"
        forceCoeffs_dict = forceCoeffsCompressible(MagU_inf=vel, Lift_Dir=lift_dir, Drag_Dir=drag_dir)

        print("Post-processing boundaryProbes upper surface :", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        upper_outputs = boundaryProbes(xmin=X_MIN_MAX[0], xmax=X_MIN_MAX[1], ymin=Y_MIN_MAX[0], ymax=Y_MIN_MAX[1], zmin=Zu_MIN_MAX[0], zmax=Zu_MIN_MAX[1], res=RES, mode='latestTime')

        print("Post-processing boundaryProbes lower surface :", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        lower_outputs = boundaryProbes(xmin=X_MIN_MAX[0], xmax=X_MIN_MAX[1], ymin=Y_MIN_MAX[0], ymax=Y_MIN_MAX[1], zmin=Zl_MIN_MAX[0], zmax=Zl_MIN_MAX[1], res=RES, mode='latestTime')

        np.savez(f"{OUTPUT_DIR}/{fileName}.npz", Coeffs=forceCoeffs_dict, Upper_surface=upper_outputs, Lower_surface=lower_outputs)
        print("Done :", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
