def process_input(input_data):
    # 將輸入數據分割成列表
    input_list = input_data.split()
    
    # 將每個元素轉換為浮點數
    input_list = [float(item) for item in input_list]
    
    # 將列表分割成子列表，每個子列表有5個元素
    output = [input_list[i:i+5] for i in range(0, len(input_list), 5)]
    
    # 移除每個子列表的第一個元素
    output = [row[1:] for row in output]
    
    return output

if __name__ == "__main__":
    results = []
    while True:
        try:
            # 讀取一行輸入
            input_data = input()
            result = process_input(input_data)
            results.extend(result)
        except EOFError:
            print(results)
            break