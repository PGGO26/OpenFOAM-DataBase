import re, time

log_path = 'HiSA.log'
with open(log_path,'r') as file:
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


import matplotlib.pyplot as plt
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
plt.savefig('plot_Resuidual.png')

print("Saving residual plot", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))