# OpenFOAM-DataBase
1. 下載翼型 `./download_airfoils.sh`
2. `runTrain.py`
    1. function `data_load` 修改 `data_dir` 為存放訓練用的 npz 資料
    2. 裡面內容 `input` 及 `target` 讀取的 key 為 npz 資料中給的名稱 (input 給自由流、翼剖面) (target 給升阻力係數)
    3. `cnn = model.fit` 中 `batch_size` 為多少筆資料擬合一次，`epochs` 為要重複訓練多少遍
