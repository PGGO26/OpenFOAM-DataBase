import time, os
import utils
import numpy as np

VEL = 223.6688
OUTPUT_DIR = 'data/data/'
CASE_DIR = 'data/caseDict/'

AOA_LST = range(0, 11, 2)
print("Numbers of total case : ", len(AOA_LST))

def runSim(freestream_x, freestream_z):
    os.chdir("OpenFOAM/")
    with open("0/include/template", 'rt') as inFile:
        with open("0/include/freestreamConditions", "wt") as outFile:
            for line in inFile:
                line = line.replace("VEL_X", f"{freestream_x}")
                line = line.replace("VEL_Z", f"{freestream_z}")
                outFile.write(line)

    print("Running simulation : ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    os.system("./Allclean")
    os.system("decomposePar > LOG.log")
    os.system("mpirun -np 16 hisa -parallel > HiSA.log")
    os.system("reconstructPar > LOG.log")
    os.chdir("..")
    print("Simulation done : ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))) 

def outputProcess(FileName):
    print("Running post-Process : focesCoeffs.", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    os.chdir("OpenFOAM/")
    os.system("hisa -postProcess -latestTime -func forceCoeffsCompressible > LOG.log")
    os.chdir("..")
    utils.outputProcess_Coeffs(Output_DIR=OUTPUT_DIR, Case_DIR=CASE_DIR, FileName=FileName)
    # print("Running post-Process : boundaryProbes upper surface.")
    # utils.write_pts(mode="boundary_upper", xy_range=0.8, ratio=0.2, res=512, z_range=0.05, layer=11)
    # os.system("postProcess -latestTime -func boundaryProbes > LOG.log")
    # utils.outputProcess(mode="boundaryProbes", res=256, surface="Upper", x_min=-0.15, xy_range=0.8)
    # print("Upper boundaryProbes done : ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    # os.chdir("..")


for aoa in AOA_LST:
    print("-"*80, "\nRunning case at AOA : ", aoa, "degree.")
    fileName = f"AOA_{aoa}"
    aoa_rad = np.deg2rad(aoa)
    fsX = VEL * np.cos(aoa_rad)
    fsZ = VEL * np.sin(aoa_rad)
    runSim(fsX, fsZ)
    outputProcess(FileName=fileName)