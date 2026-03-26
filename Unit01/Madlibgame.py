while True:
    course_period = input("請輸入課程期數: ")
    course_name = input("請輸入課程名稱: ")
    name = input("請輸入姓名: ")
    background = input("請輸入您的背景(科系或職業): ")
    expectation = input("請簡述對於本課程的期望(列出幾點修完課程想得到的技能/對自己未來的規劃): ")

    print("------------------------------------------")
    print(f"第{course_period}期 - {course_name}")
    print(f"姓名： {name} ,")
    print(f"學生背景： {background}")
    print("==========================================")
    print(f"{name} 修習完課程[ {course_name} ]後，")
    print(f"結業成果獲得: {expectation} 。")
    print("Well Done.")
    print("------------------------------------------")

    again = input("是否要再建立一次模板故事？(y/n): ")
    if again.lower() != 'y':
        break