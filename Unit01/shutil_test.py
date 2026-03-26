import os
import shutil

# 1. 在目前目錄下建立 files 資料夾
if os.path.exists('files'):
    shutil.rmtree('files')  # 如果已存在則先刪除，確保練習環境乾淨
os.mkdir('files')

# 2. 讓使用者輸入數字 N
n = int(input("請輸入要建立的資料夾數量 N: "))

# 切換到 files 資料夾內進行操作
os.chdir('files')

# 建立 f1, f2... fN 資料夾
for i in range(1, n + 1):
    os.mkdir(f'f{i}')

print(f"建立 {n} 個資料夾後：", sorted(os.listdir('.')))
input("按 Enter 鍵繼續（重新命名 f1）...")

# 3. 將 f1 重新命名為 folder1
if os.path.exists('f1'):
    os.rename('f1', 'folder1')
print("重新命名 f1 後：", sorted(os.listdir('.')))
input("按 Enter 鍵繼續（移除 folder1）...")

# 4. 移除 folder1
if os.path.exists('folder1'):
    # 如果是空資料夾可用 os.rmdir，若內部可能有檔案建議用 shutil.rmtree
    shutil.rmtree('folder1') 
print("移除 folder1 後：", sorted(os.listdir('.')))
input("按 Enter 鍵繼續（移除整個 files 資料夾）...")

# 5. 最後移除整個 files 資料夾
# 必須先退出 files 資料夾回到上一層
os.chdir('../')
shutil.rmtree('files')

print("已成功移除 files 資料夾，程式結束。")