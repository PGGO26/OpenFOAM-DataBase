# OpenFOAM-DataBase
- `SACCON` 是建立模擬 SACCON 外型在穿音速條件的模擬資料
- `M6` 是建立 Onera M6 機翼在穿音速條件的模擬資料
1. 目前分別使用兩種版本 `of2206` 及 `version10`
2. 在各版本中會有 `data` 及 `OpenFOAM` 兩個資料夾，`data` 存放生成的資料，`OpenFOAM`作為模擬使用的 case template
## SACCON case
在執行 `Allrun.py` 生成資料前需要先根據 `constant/triSurface/` 資料夾中的外型建立網格，照目前的設置會生成的網格約350萬個 cells，可以在 `system/surfaceFeatureExtract`/`system/surfaceFeatures` 及 `system/snappyHexMesh` 這兩個文件調整表面網格
### 生成指令
```bash
blockMesh
surfaceFeatureExtract
decomposePar
mpirun -np 16 snappyHexMesh -overwrite -parallel
reconstructParMesh -constant
```
>根據版本不同生成的指令會不同
>在 `version10` 中 `surfaceFeatureExtract` 要換成 `surfaceFeatures`

### 生成數據
在建立好網格後，調整 `Allrun.py` 文件中需要生成資料的數量及範圍後就可以執行開始生成數據了
