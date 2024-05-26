import cv2
import matplotlib.pyplot as plt

def draw_bounding_boxes(image_path, bounding_boxes, colors):
    # 讀取圖片
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 獲取圖片的寬度和高度
    img_height, img_width = img.shape[:2]
    
    # 繪製每一個 bounding box
    for i, box in enumerate(bounding_boxes):
        x, y, width, height = box
        
        # 將座標轉換為整數
        left = int((x - width / 2) * img_width)
        right = int((x + width / 2) * img_width)
        top = int((y - height / 2) * img_height)
        bottom = int((y + height / 2) * img_height)
        
        # 繪製矩形
        cv2.rectangle(img, (left, top), (right, bottom), colors[i], 2)
    
    # 顯示圖片
    plt.imshow(img)
    plt.show()

# 使用範例
def test1():
    image_path = 'D:\\AIcup\\32_33_train_v2\\train\\images\\0902_150000_151900\\0_00004.jpg'
    label = [
        [0.4717245578765869, 0.19566739400227864, 0.05939416885375977, 0.09901534186469184],
        [0.43120614290237425, 0.27706132464938693, 0.08367326259613037, 0.14628035227457684]
    ]
    predict = [
        [0.37279016, 0.33207834, 0.1182334,  0.17362374],
    ]
    real = [
        [0.3770164132118225, 0.39042001300387913, 0.12061488628387451, 0.21938054826524522]
    ]
    bounding_boxes=label+predict+real
    colors = [(0, 255, 0)] * len(label) + [(255, 0, 0)]*len(predict)+ [(0, 0, 255)]*len(real)  # 前四個框使用綠色，最後一個框使用紅色
    draw_bounding_boxes(image_path, bounding_boxes, colors)

def test2():
    image_path = 'D:\\AIcup\\32_33_train_v2\\train\\images\\0902_150000_151900\\0_00011.jpg'
    label = [
        [0, 0, 0, 0],
        [0.40684821605682375, 0.17069200636848572, 0.06779799461364741, 0.09316972702268564],
    ]
    predict = [
        [0.36344436, 0.20601615, 0.07222768, 0.10224696],
    ]
    real = [
        [0.2833589792251587, 0.2923420906066895, 0.11559467315673828, 0.15447478824191624],
    ]
    bounding_boxes=label+predict+real
    colors = [(0, 255, 0)] * len(label) + [(255, 0, 0)]*len(predict)+ [(0, 0, 255)]*len(real)  # 前四個框使用綠色，最後一個框使用紅色
    draw_bounding_boxes(image_path, bounding_boxes, colors)

test2()
test1()