################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Generate training data via OpenFOAM
#
################

import os, math, uuid, sys, random, time, re
import numpy as np
import pandas as pd
import utils 
import multiprocessing as mp

samples   = 8           # no. of datasets to produce
cpu_to_use = 4
freestream_angle  = math.pi / 18.  # -angle ... angle
freestream_length = 10.           # len * (1. ... factor)
# freestream_length_factor = 6.    # length factor
freestream_length_factor = 5

# Mutiprocessing using
cpu_total = mp.cpu_count()
print(f'Using cpu core : {cpu_to_use} / {cpu_total}')

airfoil_database  = "./airfoil_database/"
output_dir        = "./train/"

seed = random.randint(0, 2**32 - 1)
np.random.seed(seed)
print("Seed: {}".format(seed))


def newPoint():   
    
    os.system('python3 spline.py')
    a = np.load('data0.npy')
    output1 = ""
    for n in range(a.shape[0]):
        output1 += "( {}  {}  0.005)\n".format(a[n][0], a[n][1])
    
    # Find these internalCloud points, which may be used later in surface pressure or shear stress
    with open('system/internalCloudFoam_temp', "rt") as inFile:
        with open("system/foamtest", "wt") as outFile:
            for line in inFile:
                    line = line.replace("points","{}".format(output1))
                    outFile.write(line)

    return(0)

# Find the last file that the openfoam write in id

def find(id):
    file_list_total = os.listdir(f'{id}')

    file_list = []

    for i in range(len(file_list_total)):
        if len(file_list_total[i]) > 4:
            continue

        else:
            file_name = str(file_list_total[i])
            file_list.append(int(file_name))

    file_list.sort()

    for file in file_list:
        if file % 100 == 0:
            file_last = file
        else:
            exit

    return file_last

# print(find('id_temp'))

def shear(filename, fsX, fsY):
    with open(r'constant/polyMesh/boundary', 'r') as fboundary:
        temp = re.split('(aerofoil)', fboundary.read())[2]
        for item in re.finditer('(nFaces[\s]*)([\w]*)', temp):
            nFaces = int(item.group(2))
        for item in re.finditer('(startFace[\s]*)([\w]*)', temp):
            startFace = int(item.group(2))
        endFace = startFace + nFaces
    
    with open(r'constant/polyMesh/faces', 'r') as ffaces:
        faces = re.findall('\(.*\)', ffaces.read())
        face_total = []; Faces = []
        for item in faces:
            face_total.append(int(re.findall('[\w]*', item)[1]))
        Faces = face_total[startFace:endFace]

    with open(r'constant/polyMesh/points', 'r') as fpoints:
        points = re.findall('\(.*\)', fpoints.read())
        x_total = []; x = []
        y_total = []; y = []
        for index in range(len(points)):
            for item in re.finditer('(\()([\S ]*)(\))', points[index]):
                x_total.append(float(re.findall('[\S]*', item.group(2))[0]))
                y_total.append(float(re.findall('[\S]*', item.group(2))[2]))
        for item in Faces:
            x.append(x_total[item])
            y.append(y_total[item])
    
    with open(f'{filename}/wallShearStress', 'r') as fstress:
        stress = fstress.read().replace('(0 0 0)', '')
        stress = re.findall('\(.*\)', stress)
        Tau_x = []; Tau_y = []
        for item in stress:
            for item in re.finditer('(\()([\S ]*)(\))', item):
                Taudata = item.group(2)
                Tau_x.append(float(re.split('[\ ]', Taudata)[0]))
                Tau_y.append(float(re.split('[\ ]', Taudata)[1]))

    df = pd.DataFrame(x, columns={'x'})
    df['y'] = y
    df['shear stress x'] = Tau_x
    df['shear stress y'] = Tau_y
    df['shear stress'] = np.sqrt(np.square(df['shear stress x']) + np.square(df['shear stress y']))
    q = 0.5 * (np.square(fsX) + np.square(fsY))
    df['shear stress'] = df['shear stress'] / q

    df_up = df[df['y'] >= 0]
    df_bottom = df[df['y'] <= 0]
    df_up = df_up.sort_values('x')
    df_bottom = df_bottom.sort_values('x')

    os.mkdir('temps')
    df_up.to_csv('temps/shear_upper.csv')
    df_bottom.to_csv('temps/shear_bottom.csv')

    return(0)

def tidy(stepfile):
    newCloud = os.listdir('postProcessing/newCloud')
    boundaryCloud = os.listdir('postProcessing/boundaryProbes')

    for file in newCloud:
        if file != str(stepfile):
            os.system(f'rm -r postProcessing/newCloud/{file}')
        else:
            continue

    for file in boundaryCloud:
        if file != str(stepfile):
            os.system(f'rm -r postProcessing/boundaryProbes/{file}')
        else:
            continue 

    return(0)

def shearInterpolation():
    status = os.system('python3 interpolation.py')
    return status

def genMesh(airfoilFile):
    ar = np.loadtxt(airfoilFile, skiprows=1)

    # There are two different shapes of airfoils, with different mesh making methods
    if ar[0][1]-ar[-1][1] != 0:
        if np.max(np.abs(ar[0]-ar[(ar.shape[0]-2)]))<1e-6:
            ar = ar[:-1]

        output = ""
        pointIndex = 1000
        for n in range(ar.shape[0]):
            output += "Point({}) = {{ {}, {}, 0.000000, 0.0025}};\n". format(pointIndex, ar[n][0], ar[n][1])
            pointIndex += 1

        with open("airfoil_template.geo", "rt") as inFile:
            with open("airfoil.geo", "wt") as outFile:
                for line in inFile:
                    line = line.replace("POINTS", "{}".format(output))
                    line = line.replace("LAST_POINT_INDEX", "{}".format(pointIndex-1))
                    line = line.replace("final", "{}".format(pointIndex-2))
                    outFile.write(line)

    else:
        if np.max(np.abs(ar[0]-ar[(ar.shape[0]-2)]))<1e-6:
            ar = ar[:-1]

        output = ""
        pointIndex = 1000
        for n in range(ar.shape[0]):
            output += "Point({}) = {{ {}, {}, 0.000000, 0.0025}};\n". format(pointIndex, ar[n][0], ar[n][1])
            pointIndex += 1

        with open("airfoil_template1.geo", "rt") as inFile:
            with open("airfoil.geo", "wt") as outFile:
                for line in inFile:
                    line = line.replace("POINTS", "{}".format(output))
                    line = line.replace("LAST_POINT_INDEX", "{}".format(pointIndex-1))
                    line = line.replace("final", "{}".format(pointIndex-2))
                    outFile.write(line)

    # Convert the built .geo file to .msh file
    if os.system( "gmsh airfoil.geo -3 -format msh2 airfoil.msh > /dev/null" ) != 0:   # if it error during mesh creation, /dev/null(導向垃圾桶) it==0(correct)
        print(f"airfoil {airfoil} : error during mesh creation!")
        return(-1)
    
    # Convert the .msh file to the polyMesh folder to be used
    if os.system("gmshToFoam airfoil.msh > /dev/null")!=0:
        print(f"airfoil {airfoil} : error during coversion to OpenFoam mesh")
        return(-1)

    # Change patch to match empty and wall
    with open("constant/polyMesh/boundary", "rt") as iF:
        with open("constant/polyMesh/boundaryTemp", "wt") as oF:
            inArea = False
            inAerofoil = False
            for line in iF:
                if "front" in line or "back" in line:
                    inArea = True
                elif "aerofoil" in line:
                    inAerofoil = True
                elif inArea and "type" in line:
                    line = line.replace("patch", "empty")
                    inArea = False
                elif inAerofoil and "type" in line:
                    line = line.replace("patch", "wall")
                    inAerofoil = False
                oF.write(line)
    os.rename("constant/polyMesh/boundaryTemp", "constant/polyMesh/boundary")
    return(0)

def runSim(freestreamX, freestreamY, pressure = 1e5):
    # This list contains Ux, Uy and P 
    list = [f'Vx     {freestreamX: .2f};', f'Vy           {freestreamY:.2f};',  f'Pressure     {pressure};' ]

    # in data 'ICnBC' contaons Ux,Uy and P
    with open("0/ICnBC", "wt") as oF:
        oF.write('//Initial and boundary conditions for flow field\n')
        for i in range(len(list)):
            oF.write(list[i]+'\n')
        oF.write('#inputMode merge\n')

    status = os.system("./Allclean && simpleFoam > foam.log")
    return status

def outputProcessing(basename, stepfile, freestreamX, freestreamY, cpu_id, dataDir=output_dir, res=256):
    pCoe = np.loadtxt(f'{cpu_id}/postProcessing/boundaryProbes/{stepfile}/points.xy')
    p_U = np.loadtxt(f'{cpu_id}/postProcessing/newCloud/{stepfile}/points.xy')
    LnD = np.loadtxt(f'{cpu_id}/postProcessing/forceCoeffs_airfoil/0/forceCoeffs.dat')
    point = np.load(f'{cpu_id}/data0.npy')
    shear = pd.read_csv(f'{cpu_id}/temps/shear')

    mapOutput1 = np.zeros((6, res, res))
    mapInput = np.zeros((1, 103, 1))
    mapOutput2 = np.zeros((2, 101, 1))
    mapOutput3 = np.zeros((1, 2, 1))
    curIndex1 = 0
    curIndex2 = 0
    curIndex3 = 0

    for x in range(res):
        for y in range(res):
            xf = (((x/(res-1))-0.5)*2)+0.5
            yf = (((y/(res-1))-0.5)*2)
            if abs(p_U[curIndex1][0] - xf)<1e-5 and abs(p_U[curIndex1][1] - yf)<1e-5:
                mapOutput1[0][x][y] = freestreamX
                mapOutput1[1][x][y] = freestreamY
                mapOutput1[2][x][y] = 0
                mapOutput1[3][x][y] = p_U[curIndex1][3]
                mapOutput1[4][x][y] = p_U[curIndex1][4]
                mapOutput1[5][x][y] = p_U[curIndex1][5]
                curIndex1 += 1 
            else:
                mapOutput1[2][x][y] = 1.0

    for x2 in range(len(point)):
        mapInput[0][x2] = point[curIndex2][1]
        mapInput[0][-2] = freestreamX
        mapInput[0][-1] = freestreamY
        curIndex2 += 1

    for x3 in range(len(pCoe)):
        mapOutput2[0][x3] = pCoe[curIndex3][3]
        mapOutput2[1][x3] = shear['shear stress'].iloc[curIndex3]
        curIndex3 += 1

    mapOutput3[0][0] = LnD[stepfile][3] # Cl
    mapOutput3[0][1] = LnD[stepfile][2] # Cd
    if stepfile != 4800:
        fileName = dataDir + '%s_%d_%d' %(basename, int(freestreamX*100), int(freestreamY*100))
    else:
        fileName = 'dataset_4800/' + '%s_%d_%d' %(basename, int(freestreamX*100), int(freestreamY*100))
    print("\tsaving in " + fileName + ".npz")
    np.savez_compressed(fileName, output=mapOutput1, input = mapInput, CpnCf = mapOutput2 ,LnD=mapOutput3)

files = os.listdir(airfoil_database)
files.sort()
if len(files)==0:
	print("error - no airfoils found in %s" % airfoil_database)
	exit(1)

utils.makeDirs( ["./data_pictures", "./train", "./OpenFOAM/constant/polyMesh/sets", "./OpenFOAM/constant/polyMesh"] )
    

def full_process(airfoil, fsX, fsY, n):

    startime_case = time.time()
    id = os.getpid()
    if not os.path.exists(f'{id}'):
        os.system(f'cp -r OpenFOAM {id}')
        os.system(f'cp -r {airfoil_database}{airfoil} {id}/data0')
        print(f'case {id} is created.')
    
    basename = os.path.splitext( os.path.basename(airfoil) )[0]

    os.chdir(f'{id}')
    # path = os.path.join('..', airfoil_database, f'{basename}.dat')
    # print(path)

    if newPoint() != 0:
        print('\tnewPoint failed')
        os.chdir('..')
        return(-1)
  
    if genMesh("../" + airfoil_database + airfoil) != 0:
        print("\tmesh generation failed, aborting")
        os.chdir("..")
        return -1
    print(f'{id} : genMesh is done.')
    
    runSim(fsX, fsY)
    print(f'{id} : runSim is done.')

    os.chdir('..')
    stepfilename = find(id)
    print(stepfilename)
    os.chdir(f'{id}')

    if shear(stepfilename, fsX, fsY) != 0: # 給入 id 來存取 id 資料夾中的檔案
        print('cannot import shearstress')
        os.chdir('..')
        return(-1)
    print('shear is imported.')

    if tidy(stepfilename) != 0: # 給入穩態的步數資料夾名稱(stepfilename)
        print('\tcannot tidy the data of internalCloud and boundaryCloud')
        os.chdir('..')
        return(-1)
    print('tidy is done.')

    if shearInterpolation() != 0:
        print('\tcan not interpolation the wall shear stress.')
        os.chdir('..')
        return(-1)
    print('shearInterpolation is done.')

    os.chdir('..')
    n = int(n)
    outputProcessing(basename, stepfilename, fsX, fsY, id)

    
    totaltime_case = time.time() - startime_case

    with open('log', 'a+') as infile:
        infile.write(f'(cpu {cpu_to_use} / samples {samples}) - {stepfilename} : {(totaltime_case/60):.2f} min\n')

    print("%s : " %(time.strftime('%X')) + f'case-{id}  is done. using time : {(totaltime_case/60):.2f} min')
    if os.system(f'rm -r {id}') !=0:
        print(f'can not remove case {id} ')
        os.chdir('..')
        return(-1)

    return 0


samplelst = []
def progress(status):
    if status == 0:
        samplelst.append(status)
    print(f'current sample is : {len(samplelst)} / {samples}')

if __name__=='__main__':
    pool = mp.Pool(cpu_to_use)
    startTime = time.time()

    for n in range(samples):
        
        print("%s: Start the task" %(time.strftime('%X')))
        print("Run {}:".format(n))

        fileNumber = np.random.randint(0, len(files))
        airfoil = files[fileNumber]
        basename = os.path.splitext( os.path.basename(airfoil) )[0]
        print("\tusing {}".format(airfoil))

        length = freestream_length * np.random.uniform(2.,freestream_length_factor)
        angle  = np.random.uniform(-freestream_angle, freestream_angle)
        fsX =  math.cos(angle) * length
        fsY = -math.sin(angle) * length

        print("\tUsing len %5.3f angle %+5.3f " %( length,angle )  )
        print("\tResulting freestream vel x,y: {},{}".format(fsX,fsY))

        pool.apply_async(full_process, args=(airfoil, fsX, fsY, n), callback=progress)
        #r.wait()

    pool.close()
    pool.join()
    totalTime = (time.time() - startTime)/60
    print(f'Final time elapsed: {totalTime:.2f} minutes')
    with open('log', 'a+') as infile:
        infile.write(f'(total  running samples : {len(samplelst)}) / cpu core : {cpu_to_use}) total time : {(totalTime/60):.2f} min\n')