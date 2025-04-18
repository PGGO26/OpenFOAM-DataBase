import re
import os, shutil
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR, VISUAL_DIR = "data/", "visualizations/"
shutil.rmtree(VISUAL_DIR, ignore_errors=True)
os.mkdir(VISUAL_DIR)

def residual_plotting(filepath, output_dir):
    with open(filepath,'r') as file:
        log_content = file.read()
    GMRES_pattern = r"GMRES iteration: 0\s+Residual: [\d.]+ \(([\deE.-]+) ([\deE.-]+) ([\deE.-]+)\)"
    GMRES_matches = re.findall(GMRES_pattern, log_content)
    omega_pattern = r"Solving for omega, Initial residual = ([0-9.eE+-]+)"
    omega_matches = re.findall(omega_pattern, log_content)
    k_pattern = r"smoothSolver:  Solving for k, Initial residual = ([0-9.eE+-]+)"
    k_matches = re.findall(k_pattern, log_content)

    if GMRES_matches:
        rho = [float(value[0]) for value in GMRES_matches]
        rhoU = [float(value[1]) for value in GMRES_matches]
        rhoE = [float(value[2]) for value in GMRES_matches]
    else:
        print("No GMRES matches found")

    if omega_matches:
        omega_residual = [float(value) for value in omega_matches]
    else:
        print("No matches found for Omega Initial residual")

    if k_matches:
        k_residual = [float(value) for value in k_matches]
    else:
        print("No matches found for K Initial residual")

    plt.figure(figsize=(8,6))
    plt.semilogy(rho, label='rho')
    plt.semilogy(rhoU, label='rhoU')
    plt.semilogy(rhoE, label='rhoE')
    plt.semilogy(omega_residual, label='omega')
    plt.semilogy(k_residual, label='k')

    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title("Residual over iteration")
    plt.legend()
    output_path = os.path.join(output_dir, "HiSA_Residuals.png")
    plt.savefig(output_path)
    print(f"Saving residual figure at {output_path}.")

residual_plotting(filepath="OpenFOAM/HiSA.log", output_dir=VISUAL_DIR)

for f in os.listdir(DATA_DIR):
    if f.endswith(".npz"):
        data = np.load(os.path.join(DATA_DIR, f), allow_pickle=True)
        steps = len(data['Upper_surface'])
        for step in range(steps):
            upper_mask = data['Upper_surface'][step][0]
            lower_mask = data['Lower_surface'][step][0]
            upper_p_field = data['Upper_surface'][step][1] + upper_mask
            upper_s_field = data['Upper_surface'][step][2] + upper_mask
            lower_p_field = data['Lower_surface'][step][1] + lower_mask
            lower_s_field = data['Lower_surface'][step][2] + lower_mask
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fields = [upper_p_field, lower_p_field, upper_s_field, lower_s_field]
            titles = ["p-upper", "p-lower", "shear-upper", "shear-lower"]
            
            for i, (field, title) in enumerate(zip(fields, titles)):
                im = axes[i//2, i%2].imshow(field, cmap='jet')
                axes[i//2, i%2].set_title(title)
                fig.colorbar(im, ax=axes[i//2, i%2])
            
            plt.tight_layout()
            plt.savefig(os.path.join(VISUAL_DIR, f.replace(".npz", f"_{step}.png")))
            plt.close()
