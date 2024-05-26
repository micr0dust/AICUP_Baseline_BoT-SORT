import os

root_dir = './train'
dict_id = {}

if not os.path.exists(f"{root_dir}/lstm_data"):
    os.mkdir(f"{root_dir}/lstm_data")

distribute = [0]*11

# 遍歷root_dir下的所有子資料夾
for folder in os.listdir(root_dir+'/labels'):
    dir_path = root_dir+'/labels/'+folder
    # 獲取資料夾中的所有 label 檔案，並按照檔案名稱排序
    label_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.txt')])

    prev_cam = None
    for file in label_files:
        cam = file.split('_')[0]
        if cam != prev_cam:
            dict_id.clear()
        prev_cam = cam

        with open(os.path.join(dir_path, file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                _, x, y, w, l, ID = line.split()
                if ID in dict_id:
                    dict_id[ID].append((cam, x, y, w, l))
                else:
                    dict_id[ID] = [(cam, x, y, w, l)]

        # 檢查所有字典內的ID，如果該文件沒有出現該ID，就將該ID紀載的(cam,x,y,w,l)list輸出到文件f"{cam}_{ID}.txt"
        id_list = [line.split()[-1] for line in lines]
        for ID in list(dict_id.keys()):
            if ID not in id_list:
                with open(f"{root_dir}/lstm_data/{cam}_{ID}.txt", 'w') as f:
                    for record in dict_id[ID][:10]:
                        f.write(' '.join(record) + '\n')
                # if len(dict_id[ID]) > 100:
                #     print(f"in folder {folder}, {cam}_{ID} has {len(dict_id[ID])} records")
                distribute[min(10, len(dict_id[ID]))]+=1
                del dict_id[ID]

print(distribute)