import os, time

print("Constructing mesh...", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
os.system("./Allclean")
os.system("blockMesh > LOG.log")
os.system("surfaceFeatures > LOG.log")
os.system("decomposePar > LOG.log")
os.system("mpirun -np 16 snappyHexMesh -overwrite -parallel > MESHING.log")
os.system("reconstructParMesh -constant > LOG.log")
os.system("rm -rf proc*")
os.system("checkMesh > MESH.log")
print("Mesh done : ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))