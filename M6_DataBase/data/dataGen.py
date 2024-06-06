import os
import numpy as np
import pandas as pd

outputDir = 'train/'
V_sonic = 319.083
Mach_values = [0.84]
angle_values = [3.06, 6.06]

def runSim(freestreamX, freestreamZ):
    with open("0/include/template", "rt") as inFile:
        with open("0/include/freestreamConditions", "wt") as outFile:
            for line in inFile:
                line = line.replace("VEL_X", "{}".format(freestreamX))
                line = line.replace("VEL_Z", "{}".format(freestreamZ))
                outFile.write(line)

    folders = [f'processor{i}' for i in range(10)]
    all_folders_exist = all(os.path.isdir(folder) for folder in folders)
    if all_folders_exist:
        print("All processor folders exist. Skipping decomposePar.")
    else:
        print("Not all processor folders exist. Running decomposePar.")
        os.system("decomposePar > log/decomposePar.log")
    print("Runing HiSa.")
    os.system('mpirun -n 8 hisa -parallel > log/hisa.log')
    print("Simulation done.")
    os.system("reconstructPar > log/reconstructPar.log")
    print(f"Case {fileName} runSim is done.")
    os.system("postProcess -latestTime -func boundaryCloud > log/boundaryCloud.log")
    os.system("postProcess -latestTime -func internalCloud > log/internalCloud.log")
    os.chdir("..")
    outputProcessing(baseName=fileName, dataDir=outputDir, Vel=Vel)
    os.system("./Allclean")
    print('\tdone')

def outputProcessing(baseName, dataDir, Vel):
    # boundaryCloud
    timestep = os.listdir(f"OpenFOAM/postProcessing/boundaryCloud/")[0]
    timestep_path = f"OpenFOAM/postProcessing/boundaryCloud/{timestep}"
    boundaryCase_name = os.listdir(f"{timestep_path}/")[0]
    boundaryCase_path = f"{timestep_path}/{boundaryCase_name}"
    print(f"Case path : {boundaryCase_path}")
    boundary_content = pd.read_csv(boundaryCase_path, delimiter='\s+', skiprows=0, header=None, names=['x','y','z','p'])
    rho = 1.184
    pv = 0.5 * rho * (Vel**2)
    boundary_content['Cp'] = (boundary_content['p']-98858.97) / pv
    upper = boundary_content[(boundary_content['z'] >= 0.)]
    x = upper['x']
    y = upper['y']
    z = upper['z']
    cp = upper['Cp']
    normalized_x = (x - x.min()) / (x.max() - x.min())
    normalized_y = (y - y.min()) / (y.max() - y.min())
    normalized_z = (z - z.min()) / (z.max() - z.min())
    normalized_Cp = (cp - cp.min()) / (cp.max() - cp.min())
    res = round(len(x) ** (1/3))
    GF_Output= np.ones((3,res,res))
    GF_Output[0][(normalized_x * (res-1)).astype(int), (normalized_y * (res-1)).astype(int)] = normalized_z
    GF_Output[1][(normalized_x * (res-1)).astype(int), (normalized_y * (res-1)).astype(int)] = normalized_Cp

    lower = boundary_content[(boundary_content['z'] <= 0.)]
    x = lower['x']
    y = lower['y']
    z = lower['z']
    cp = lower['Cp']
    normalized_x = (x - x.min()) / (x.max() - x.min())
    normalized_y = (y - y.min()) / (y.max() - y.min())
    normalized_z = (z - z.min()) / (z.max() - z.min())
    normalized_Cp = (cp - cp.min()) / (cp.max() - cp.min())
    res = round(len(x) ** (1/3))
    GF_Output[2][(normalized_x * (res-1)).astype(int), (normalized_y * (res-1)).astype(int)] = normalized_Cp

    # internalCloud
    internalCase_name = os.listdir(f'OpenFOAM/postProcessing/internalCloud/{timestep}/')[0]
    internalCase_path = f"OpenFOAM/postProcessing/internalCloud/{timestep}/{internalCase_name}"
    print(f"Case path : {internalCase_path}")
    internal_content = pd.read_csv(internalCase_path, delimiter='\s+', skiprows=0, header=None, names=['x','y','z','p','u','v','w'])
    section_lst = [0.2, 0.65, 0.8, 0.9, 0.96]
    Output_lst = []
    for section_value in section_lst:
        section_df = internal_content[(internal_content['y'] >= section_value * 1.1963 - 1e-2) & (internal_content['y'] <= section_value * 1.1963 + 1e-2)]
        x = section_df['x']
        z = section_df['z']
        normalized_x = (x - x.min()) / (x.max() - x.min())
        normalized_z = (z - z.min()) / (z.max() - z.min())
        res = round(len(x) ** (1/2))
        section_Output = np.ones((4,res,res))
        section_Output[0][(normalized_x * (res-1)).astype(int), (normalized_z * (res-1)).astype(int)] = section_df['p']
        section_Output[1][(normalized_x * (res-1)).astype(int), (normalized_z * (res-1)).astype(int)] = section_df['u']
        section_Output[2][(normalized_x * (res-1)).astype(int), (normalized_z * (res-1)).astype(int)] = section_df['v']
        section_Output[3][(normalized_x * (res-1)).astype(int), (normalized_z * (res-1)).astype(int)] = section_df['w']
        Output_lst.append(section_Output)

    fileName = dataDir + baseName
    print("\tsaving in " + fileName + ".npz")
    np.savez_compressed(fileName, Global=GF_Output[0],Upper=GF_Output[1],Lower=GF_Output[2],
                                  Local_20pc=Output_lst[0],Local_65pc=Output_lst[1],Local_80pc=Output_lst[2],
                                  Local_90pc=Output_lst[3],Local_96pc=Output_lst[4])

for Mach in Mach_values:    
    for angle in angle_values:
        fileName = f"M6_{Mach}_{angle}"
        angle_randiant = angle / 180 * np.pi
        Vel = V_sonic * Mach
        fsX = Vel * np.cos(angle_randiant)
        fsZ = Vel * np.sin(angle_randiant)
        
        os.chdir("OpenFOAM/")
        runSim(fsX,fsZ)
        os.system("pwd")