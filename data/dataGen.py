import os, re, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp

cpu_to_use = 4
output_dir = 'data/'
fs = 30
airfoil_Lst = os.listdir('airfoil/')
# AOA = [-5, -2.5, 0, 2.5, 5]
AOA = [-5]

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
        print("Error during coversion to OpenFoam mesh")
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
                    
    return 0

def runSim(freestreamX, freestreamY):
    with open("U_template", "rt") as inFile:
        with open("0/U", "wt") as outFile:
            for line in inFile:
                line = line.replace("VEL_X", "{}".format(freestreamX))
                line = line.replace("VEL_Y", "{}".format(freestreamY))
                outFile.write(line)

    os.system("simpleFoam > foam.log")
    return 0

def find(caseName):
    file_list_total = os.listdir(caseName)
    # print(f'case : {caseName}')
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

def shear(stepIndex, fsX, fsY):
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
    
    with open(f'{stepIndex}/wallShearStress', 'r') as fstress:
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

def outputProcessing(CaseName,fsX, fsY, dataDir=output_dir):
    shearUpper = f'{CaseName}/temps/shear_upper.csv'
    shearLower = f'{CaseName}/temps/shear_bottom.csv'
    df_upper = pd.read_csv(shearUpper)
    df_lower = pd.read_csv(shearLower)

    # upper
    section_upper = int(np.round(len(df_upper) / 10))
    section_upper_1 = df_upper[:section_upper]
    section_upper_2 = df_upper[section_upper:-section_upper]
    section_upper_3 = df_upper[-section_upper:]
    x_upper = [section_upper_1.loc[0,'x']]
    upper = [section_upper_1.loc[0,'shear stress']]

    step_1 = np.linspace(1, len(section_upper_1) - 1, 15)
    for i in step_1:
        index = int(i)
        x_upper.append(section_upper_1.loc[index,'x'])
        upper.append(section_upper_1.loc[index,'shear stress'])

    section_upper_2 = section_upper_2.reset_index()
    step_2 = np.linspace(1, len(section_upper_2) - 1, 20)
    for j in step_2:
        index = int(j)
        x_upper.append(section_upper_2.loc[index,'x'])
        upper.append(section_upper_2.loc[index,'shear stress'])

    section_upper_3 = section_upper_3.reset_index()
    step_3 = np.linspace(1, len(section_upper_1) - 1, 15)
    for k in step_3:
        index = int(k)
        x_upper.append(section_upper_3.loc[index,'x'])
        upper.append(section_upper_3.loc[index,'shear stress'])

    # lower
    section_lower = int(np.round(len(df_lower) / 10))
    section_lower_1 = df_lower[:section_lower]
    section_lower_2 = df_lower[section_lower:-section_lower]
    section_lower_3 = df_lower[-section_lower:]
    x_lower = [section_lower_1.loc[0,'x']]
    lower = [section_lower_1.loc[0,'shear stress']]

    step_1 = np.linspace(1, len(section_lower_1) - 1, 15)
    for i in step_1:
        index = int(i)
        x_lower.append(section_lower_1.loc[index,'x'])
        lower.append(section_lower_1.loc[index,'shear stress'])

    section_lower_2 = section_lower_2.reset_index()
    step_2 = np.linspace(1, len(section_lower_2) - 1, 20)
    for j in step_2:
        index = int(j)
        x_lower.append(section_lower_2.loc[index,'x'])
        lower.append(section_lower_2.loc[index,'shear stress'])

    section_lower_3 = section_lower_3.reset_index()
    step_3 = np.linspace(1, len(section_lower_1) - 1, 15)
    for k in step_3:
        index = int(k)
        x_lower.append(section_lower_3.loc[index,'x'])
        lower.append(section_lower_3.loc[index,'shear stress'])


    Input1 = np.asarray([x_upper,upper])
    Input2 = np.asarray([x_lower,lower])
    Input3 = np.asarray([fsX, fsY])
    fileName = dataDir + f'{CaseName}'
    print(f'fileName : {fileName}.npz')
    np.savez_compressed(fileName, upper = Input1, lower=Input2, Vel=Input3)
    print("\tsaving in " + fileName + '.npz')

def fullProcess(caseName,fsX,fsY):
    startime_case = time.time()

    # Mesh generation
    os.chdir(f'{caseName}')
    if genMesh(airfoilFile='data0') != 0:
        print("\tmesh generation failed, aborting")
        os.chdir('..')
        return -1
    print(f'{caseName} : Doing runSim.')
    # runSim
    if runSim(fsX,fsY) != 0:
        print('runSim is failed')
        return -1

    os.chdir('..')
    print(f'case - {caseName} runSim id done.')

    # OutputProcess
    stepIndex = find(caseName)
    os.chdir(caseName)
    print(f'case: {caseName} - final step : {stepIndex}')
    if shear(stepIndex, fsX, fsY) != 0:
        print('cannot import shearstress')
        os.chdir('..')
    os.chdir('..')
    print(f'case : {caseName} shear processing is done')
    totaltime_case = time.time() - startime_case
    usingTime = np.round((totaltime_case/60),2)
    outputProcessing(CaseName=caseName,fsX=fsX, fsY=fsY)
    os.system(f'rm -r {caseName}')

    print("%s : " %(time.strftime('%X')) + f'case-{caseName}  is done. using time : {usingTime} min\n')
    return 0

samplelst = []
def progress(status):
    samplelst.append(status)
    print(f'{samplelst}\t{len(samplelst)} / {len(AOA) * len(airfoil_Lst)}')

if __name__=='__main__':
    pool = mp.Pool(cpu_to_use)
    startTime = time.time()

    # for airfoil in airfoil_Lst:
    #     basename = os.path.splitext(os.path.basename(airfoil))[0]
    #     print("\tusing {}".format(airfoil))
    #     for aoa in AOA:
    #         print("%s: Start the task" %(time.strftime('%X')))
    #         caseName = f'{basename}_{aoa}'
    #         angle = (np.pi / 180) * aoa
    #         fsX = fs * np.cos(angle)
    #         fsY = fs * np.sin(angle)
    #         if not os.path.exists(f'{caseName}'):
    #             os.system(f'cp -r OpenFOAM {caseName}')
    #             os.system(f'cp -r airfoil/{airfoil} {caseName}/data0')
    #             print(f'case {caseName} is created.')
    #         pool.apply_async(fullProcess, args=(caseName,fsX,fsY), callback=progress)

    for aoa in AOA:
        for airfoil in airfoil_Lst:
            basename = os.path.splitext(os.path.basename(airfoil))[0]
            caseName = f'{basename}_{aoa}'
            angle = (np.pi / 180) * aoa
            fsX = fs * np.cos(angle)
            fsY = fs * np.sin(angle)
            if not os.path.exists(f'{caseName}'):
                os.system(f'cp -r OpenFOAM {caseName}')
                os.system(f'cp -r airfoil/{airfoil} {caseName}/data0')
                print(f'case {caseName} is created.')
            pool.apply_async(fullProcess, args=(caseName,fsX,fsY), callback=progress)

    pool.close()
    pool.join()
    totalTime = (time.time() - startTime) / 60
    print(f'Final time elapsed: {np.floor(totalTime / 60)} hour {np.floor(totalTime % 60)} minutes')