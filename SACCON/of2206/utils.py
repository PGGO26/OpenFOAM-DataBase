import os, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class postProcessing:
    def __init__(self, OUTPUT_DIR, CASE_DIR, FILENAME, RES, X_MIN, XY_RANGE):
        self.output_Dir = OUTPUT_DIR
        self.case_Dir = CASE_DIR
        self.fileName = FILENAME
        self.x_min = X_MIN
        self.xy_range = XY_RANGE

        self.Mask = np.zeros((4, RES, RES))
        self.Output = np.zeros((2, RES, RES))
    
    def write_pts(self, MODE, RES, Z_RANGE, LAYER):
        x_min, x_max = self.x_min, self.x_min + self.xy_range
        y_min, y_max = -self.xy_range/2, self.xy_range/2

        x_values = np.linspace(x_min, x_max, RES)
        y_values = np.linspace(y_min, y_max, RES)

        if MODE == "boundary_upper":
            z_min, z_max = 0.0, Z_RANGE
            z_values = np.linspace(z_min, z_max, LAYER)
        elif MODE == "boundary_lower":
            z_min, z_max = -Z_RANGE, 0.0
            z_values = np.linspace(z_min, z_max, LAYER)

        elif MODE == "internal":
            z_min, z_max = -Z_RANGE, Z_RANGE
            y_values = np.linspace(y_max, y_max, LAYER)
            z_values = np.linspace(z_min, z_max, RES)
        else:
            print("No matching MODE found.")

        with open("system/include/pts", 'w') as file:
            file.write("pts\n(\n")
            for z in z_values:
                for y in y_values:
                    for x in x_values:
                        file.write(f"({x} {y} {z})\n")
            file.write(");")
            print(f"Write the Surface points fille : system/include/pts.")

    def outputPlot(self, OUTPUT, CMAP, LABEL, TITLE):
        field = np.copy(OUTPUT)
        field = np.flipud(field.transpose())

        plt.figure(figsize=(8,6))
        plt.imshow(field, cmap=CMAP, interpolation='nearest')
        plt.colorbar(ticks=np.linspace(np.nanmin(field), np.nanmax(field), 12),label=LABEL)
        plt.title(TITLE)
        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')
        plt.savefig(self.fileName)
        print("Saving plot in ", self.fileName)

    def outputProcess(self, MODE, SURFACE="None", RES=512, PINF=94410, QINF=3.4E5):

        if MODE == "forceCoeffs":
            with open(f"OpenFOAM/postProcessing/forceCoeffsCompressible/0/coefficient.dat") as inFile:
                content = inFile.readlines()
            last_line = content[-1].strip().split()
            self.Cd = float(last_line[1])
            self.Cl = float(last_line[4])

        elif MODE == "boundaryCloud":
            post_path = os.path.join("OpenFOAM/postProcessing/", MODE)
            self.final_step = max(step for step in os.listdir(post_path))
            step_path = os.path.join(post_path, str(self.final_step))
            file_path = os.path.join(step_path, os.listdir(step_path)[-1])
            content = pd.read_csv(file_path, delimiter='\s+', comment="#", skiprows=0, header=None, names=['x','y','z','p']).round(5)
        
            x = content['x'][1:].astype(float)
            y = content['y'][1:].astype(float)
            z = content['z'][1:].astype(float)
            p = content['p'][1:].astype(float)

            normalized_x = (x - self.x_min) / self.xy_range
            normalized_y = (y + self.xy_range/2) / self.xy_range
            normalized_p = (p - PINF) / QINF

            if SURFACE == "Upper":
                self.Mask[0][(normalized_x * (RES-1)).astype(int), (normalized_y * (RES-1)).astype(int)] = z
                self.Output[0][(normalized_x * (RES-1)).astype(int), (normalized_y * (RES-1)).astype(int)] = normalized_p

            elif SURFACE == "Lower":
                self.Mask[1][(normalized_x * (RES-1)).astype(int), (normalized_y * (RES-1)).astype(int)] = z
                self.Mask[2] = np.abs(self.Mask[0] - self.Mask[1])
                self.Mask[3] = 0.5 * (self.Mask[0] + self.Mask[1])
                self.Output[1][(normalized_x * (RES-1)).astype(int), (normalized_y * (RES-1)).astype(int)] = normalized_p
            else:
                print("No matching surface found.")
        else:
            print("No matching mode found.")

    def output(self):
        npz_path  = os.path.join(self.output_Dir, self.fileName)
        np.savez_compressed(npz_path, time=self.final_step, Cl=self.Cl, Cd=self.Cd, Map=self.Mask[2:], Cp=self.Output)

        case_path = os.path.join(self.case_Dir, self.fileName)
        if os.path.exists(case_path):
            shutil.rmtree(case_path)
        shutil.copytree(f"OpenFOAM/{self.final_step}/", case_path)