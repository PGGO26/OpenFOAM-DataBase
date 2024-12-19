import cv2
import os

# 定義圖片所在資料夾路徑和輸出影片的檔案名稱
image_folder = 'plots'  # 替換成你的圖片資料夾路徑
output_video = 'output_video.mp4'  # 輸出的影片名稱

# 定義一個函數來提取圖片檔名中 "_" 後的數字
def get_number_from_filename(filename):
    # 假設圖片檔名格式為 "image_123.jpg"
    try:
        # 提取 "_" 後的數字部分，並轉換為整數
        return int(filename.split('_')[-1].split('.')[0])
        # return float(filename.split('_')[-3])
    except ValueError:
        # 如果無法轉換成數字，則返回一個極小的數，以防止排序錯誤
        return -1

# 獲取所有圖片的檔案名稱並根據 "_" 後的數字進行排序
images = [img for img in os.listdir(image_folder) if img.startswith("Result_upper") and img.endswith(".png")]
images.sort(key=get_number_from_filename)

# 讀取第一張圖片來獲取影片的寬高資訊
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_image.shape

# 定義影片的格式與參數 (影片名稱, 編碼方式, fps, 影片尺寸)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定義mp4編碼
video = cv2.VideoWriter(output_video, fourcc, 2, (width, height))  # fps設為30

# 逐一將圖片寫入影片
for image in images:
    img_path = os.path.join(image_folder, image)
    img = cv2.imread(img_path)
    video.write(img)

# 釋放 VideoWriter 物件
video.release()
print("影片生成完成！")
