# OpenFOAM-DataBase
- `SACCON` 是建立模擬 SACCON 外型在穿音速條件的模擬資料
- `M6` 是建立 Onera M6 機翼在穿音速條件的模擬資料
1. 目前分別使用兩種版本 `of2206` 及 `version10`，目前 `version` 的結果不準確
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

# SACCON case
目前包含使用兩個 OpenFOAM 版本 (v2206, version10) 的 SACCON case
1. 使用 HiSA 求解器
2. 使用 snappyHexMesh 方法建立網格
## OpenFOAM version10 case
> 需要注意在模擬開始前 `0/` 資料夾中不能有 `wallShearStress` 的檔案存在
- `mesh_test` 測試網格的模擬結果
- `dataGen` 建立資料庫
### `mesh_test`
調整 `system/` 資料夾中的 `blockMeshDict` 或是 `snappyHexMeshDict` 文件來生成不同的網格
- 先進入 OpenFOAM 資料夾中執行 `meshing.py` 代碼，建立網格
- 再使用 `test_mesh.py` 代碼進行模擬，根據需求調整內部的後處理函數參數 (`boundaryProbes`, `outputProcessing`)
### `dataGen`
1. 同樣先使用 `meshing.py` 建立網格
2. 再使用 `dataGen.py` 建立資料庫
