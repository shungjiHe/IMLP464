import os

# 自動定位路徑
current_dir = os.path.dirname(__file__)
eng_path = os.path.join(current_dir, 'data', 'english_list.csv')
math_path = os.path.join(current_dir, 'data', 'math_list.csv')
output_path = os.path.join(current_dir, 'Score.csv')

scores = {}

# 1. 讀取英文成績
if os.path.exists(eng_path):
    with open(eng_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.strip().split(',')
            if not data or len(data) < 2: continue
            
            # 檢查第二欄是否為數字
            if not data[1].isdigit(): 
                continue
                
            name = data[0]
            scores[name] = int(data[1])

# 2. 讀取數學成績並加總
if os.path.exists(math_path):
    with open(math_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.strip().split(',')
            if not data or len(data) < 2: continue
            
            # 檢查第二欄是否為數字
            if not data[1].isdigit():
                continue
                
            name = data[0]
            scores[name] = scores.get(name, 0) + int(data[1])

# 3. 寫入與列印
with open(output_path, 'w', encoding='utf-8') as f_out:
    f_out.write("姓名,總分\n")
    for name, total in scores.items():
        f_out.write(f"{name},{total}\n")
        print(f"{name} {total}")