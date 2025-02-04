import time, os
import utils
import numpy as np

VEL = 319.5269
OUTPUT_DIR = 'data/data/'
CASE_DIR = 'data/caseDict/'

AOA_LST = range(-5, 11, 2)
MACH_LST = [0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25]
# AOA_LST = [10]
# MACH_LST=[0.7]
print("Numbers of total case : ", len(AOA_LST) * len(MACH_LST))

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

def outputProcess(FileName, QINF):
    output_Processing = utils.postProcessing(OUTPUT_DIR, CASE_DIR, FileName, RES=512, X_MIN=-0.15, XY_RANGE=0.8)
    ## forceCoeffs
    print("Running post-Process : focesCoeffs : ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    os.chdir("OpenFOAM/")
    os.system("hisa -postProcess -latestTime -func forceCoeffsCompressible > LOG.log")
    os.chdir("..")
    output_Processing.outputProcess(MODE="forceCoeffs")
    ## boundaryProbes
    print("Running post-Process : boundaryProbes upper surface :", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    os.chdir("OpenFOAM/")
    output_Processing.write_pts(MODE="boundary_upper", RES=800, Z_RANGE=0.05, LAYER=11)
    # output_Processing.write_pts(MODE="boundary_upper", RES=512, Z_RANGE=0.05, LAYER=3)
    os.system("postProcess -latestTime -func boundaryCloud > LOG.log")
    os.chdir("..")
    output_Processing.outputProcess(MODE="boundaryCloud", SURFACE="Upper", RES=512, PINF=94410, QINF=QINF)
    # output_Processing.outputProcess(MODE="boundaryCloud", SURFACE="Upper", RES=256, PINF=94410, QINF=QINF)

    print("Running post-Process : boundaryProbes lower surface.", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    os.chdir("OpenFOAM/")
    output_Processing.write_pts(MODE="boundary_lower", RES=800, Z_RANGE=0.05, LAYER=11)
    # output_Processing.write_pts(MODE="boundary_lower", RES=512, Z_RANGE=0.05, LAYER=3)
    os.system("postProcess -latestTime -func boundaryCloud > LOG.log")
    os.chdir("..")
    output_Processing.outputProcess(MODE="boundaryCloud", SURFACE="Lower", RES=512, PINF=94410, QINF=QINF)
    # output_Processing.outputProcess(MODE="boundaryCloud", SURFACE="Lower", RES=256, PINF=94410, QINF=QINF)
    print("Upper boundaryProbes done : ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    output_Processing.output()

for mach in MACH_LST:
    for aoa in AOA_LST:
        print("-"*80, "\nRunning case at AOA : ", aoa, "degree.")
        fileName = f"MACH_{mach}_AOA_{aoa}"
        vel = VEL * mach
        q_inf = 0.5 * 1.3613 * vel**2
        aoa_rad = np.deg2rad(aoa)
        fsX = vel * np.cos(aoa_rad)
        fsZ = vel * np.sin(aoa_rad)
        runSim(fsX, fsZ)
        outputProcess(FileName=fileName, QINF=q_inf)