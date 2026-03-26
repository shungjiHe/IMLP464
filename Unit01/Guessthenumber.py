import random

def guess_number_game():
    # 1. 初始化遊戲參數
    answer = random.randint(1, 100)  # 產生 1 到 100 的隨機整數
    low, high = 1, 100              # 設定初始範圍
    count = 0                        # 嘗試次數計數器
    
    print("======= 猜數字遊戲 =======")

    while True:
        # 2. 顯示目前的範圍並取得輸入
        try:
            user_input = input(f"猜數字範圍 {low} < ? < {high} ：")
            guess = int(user_input)
        except ValueError:
            print("請輸入有效的整數數字！")
            continue

        # 3. 檢查輸入是否在範圍內
        if guess <= low or guess >= high:
            print(f"錯誤提示：請輸入介於 {low} 到 {high} 之間的數字！")
            continue

        count += 1

        # 4. 判斷勝負與邏輯分支
        if guess == answer:
            print(f"賓果！猜對了，答案是 {answer}")
            print(f"您總共嘗試了 {count} 次。")
            break
        else:
            # 根據猜測縮小顯示範圍
            if guess > answer:
                print("再小一點！！")
                high = guess
            else:
                print("再大一點！！")
                low = guess
            
            print(f"您猜了 {count} 次。")

            # 5. 詢問是否繼續或退出
            choice = input("想繼續玩嗎？(按 Enter 繼續，輸入 'n' 退出): ").lower()
            if choice == 'n':
                print(f"遊戲結束，正確答案是 {answer}。")
                break

if __name__ == "__main__":
    guess_number_game()