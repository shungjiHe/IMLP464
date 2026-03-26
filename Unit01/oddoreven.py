try:
    num = int(input("請輸入一個 1 到 1000 之間的數字: "))

    if num < 1 or num > 1000:
        print("錯誤：輸入的數字超出範圍，請輸入 1 到 1000 之間的數。")
    else:
        if num % 2 == 0:
            print(f"{num} 偶數")
        else:
            print(f"{num} 奇數")

except ValueError:
    print("錯誤：請務必輸入整數數字。")