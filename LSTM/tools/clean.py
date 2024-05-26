from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np

root_dir = './train'

# 遍歷root_dir下的所有子資料夾
for folder in os.listdir(root_dir+'/images'):
    dir_path = root_dir+'/images/'+folder
    txt_path = root_dir+'/labels/'+folder
    # 獲取資料夾中的所有圖片檔案，並按照檔案名稱排序
    image_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.jpg') or f.endswith('.png')])
    i = 0
    while i < len(image_files) - 1:
        # 打開並讀取圖片
        image1 = Image.open(os.path.join(dir_path, image_files[i]))
        image2 = Image.open(os.path.join(dir_path, image_files[i + 1]))

        # 將圖片轉換為灰度
        image1 = image1.convert('L')
        image2 = image2.convert('L')

        # 將圖片轉換為numpy數組
        image1 = np.array(image1)
        image2 = np.array(image2)

        # 計算兩張圖片的SSIM
        s = ssim(image1, image2)

        # 如果SSIM大於閾值，則刪除第二張圖片
        if s > 0.90:
            os.remove(os.path.join(dir_path, image_files[i + 1]))
            os.remove(os.path.join(txt_path, image_files[i + 1].split('.')[0]+'.txt'))
            # 更新image_files列表
            image_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.jpg') or f.endswith('.png')])
        else:
            i += 1