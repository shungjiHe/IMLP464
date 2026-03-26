import os

# 1. 取得路徑
yourPath = input("請輸入查詢路徑: ")
allFileList = os.listdir(yourPath)

# 準備一個清單來存放學號
student_ids = []

for file in allFileList:
    full_path = os.path.join(yourPath, file)
    
    if os.path.isdir(full_path):
        print(f"I'm a directory: {file}")
    elif os.path.isfile(full_path):
        print(f"I'm a File: {file}")
        
        # 進階：提取學號 (假設格式為 hw_學號.副檔名)
        if file.startswith("hw_"):
            # 做法：先用 '.' 分割取檔名，再用 '_' 分割取後半段
            # 例如: hw_07945001.txt -> hw_07945001 -> 07945001
            student_id = file.split('.')[0].split('_')[1]
            student_ids.append(student_id)

# 2. 將學號存成 CSV 檔
output_file = "student_list.csv"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("學號\n") # 寫入標題
    for s_id in student_ids:
        f.write(s_id + "\n")

print(f"\n--- 任務完成 ---")
print(f"已從檔案中提取 {len(student_ids)} 筆學號，並存至 {output_file}")