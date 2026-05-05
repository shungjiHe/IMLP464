# pandas_practice_all_in_one.py
# 依照「2.2.2_Pandas數據分析.ipynb」整理成單一 Python 檔案

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 0. 輔助：如果沒有 CSV，就產生練習資料
# =========================

def make_fake_grades(n=100, seed=42):
    rng = np.random.default_rng(seed)
    last_names = list("王李張劉陳楊黃吳趙周徐孫朱馬胡郭林何高梁鄭羅宋謝唐許韓馮鄧曹彭曾蕭田董潘袁蔡蔣余杜葉程魏蘇呂丁任沈姚盧姜崔鍾譚陸汪范金石廖賈夏韋傅方白鄒孟熊秦邱江尹薛閻段雷侯龍史陶黎賀顧毛郝龔邵萬錢嚴賴莫孔")
    given_a = list("俊玉淑上靜正怡亦淳威心士志雅慧文凱建君偉鈴潔婷峰希遠德瑄伯")
    given_b = list("安華婷紫成偉君瑄茜德怡賢遠鈴潔婷峰希琳亨慧君伯")
    names = [
        last_names[i % len(last_names)] + given_a[(i * 3) % len(given_a)] + given_b[(i * 7) % len(given_b)]
        for i in range(n)
    ]
    grades = rng.integers(8, 16, size=(n, 5))
    return pd.DataFrame(grades, columns=["國文", "英文", "數學", "自然", "社會"]).assign(姓名=names)[
        ["姓名", "國文", "英文", "數學", "自然", "社會"]
    ]


def load_grades():
    candidates = [
        "data/grades.csv",
        "grades.csv",
    ]
    for path in candidates:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            pass
    return make_fake_grades()


def show(title, obj=None):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    if obj is not None:
        print(obj)


# =========================
# 1. 開始使用 pandas
# =========================

df = load_grades()

show("df 的型態", type(df))
show("df 前五筆", df.head())


# =========================
# 2. Pandas 基本資料結構：DataFrame / Series
# =========================

show("取出國文欄位：df['國文']", df["國文"])
show("取出國文欄位：df.國文", df.國文)

# 畫出國文成績折線圖
df.國文.plot(title="國文成績")
plt.show()

# 畫出國文成績直方圖
df.國文.hist()
plt.title("國文成績分布")
plt.xlabel("分數")
plt.ylabel("人數")
plt.show()


# =========================
# 3. 一些基本的資料分析
# =========================

show("國文平均", df.國文.mean())
show("國文標準差", df.國文.std())
show("基本統計 describe()", df.describe(numeric_only=True))
show("相關係數矩陣 corr()", df.corr(numeric_only=True))
show("國文和數學的相關係數", df.國文.corr(df.數學))


# =========================
# 4. 增加一行，也就是增加欄位
# =========================

subject_cols = ["國文", "英文", "數學", "自然", "社會"]

# 總級分：五科相加
df["總級分"] = df[subject_cols].sum(axis=1)

# 加權：國文 + 英文 + 數學 * 2
df["加權"] = df.國文 + df.英文 + df.數學 * 2

show("新增總級分與加權欄位", df.head())


# =========================
# 5. 排序和 index 重設
# =========================

show("依總級分由高到低排序", df.sort_values(by="總級分", ascending=False).head(10))

show(
    "加權分最高；同分才看總級分",
    df.sort_values(by=["加權", "總級分"], ascending=False).head(10),
)

# 排序後重設 index，從 1 開始編號
df = df.sort_values(by=["加權", "總級分"], ascending=False)
df.index = range(1, len(df) + 1)
show("排序後重設 index", df.head())


# =========================
# 6. 篩出我們要的資料
# =========================

show("找出數學滿級分的同學", df[df.數學 == 15])

show(
    "找出數學和英文都滿級分的同學",
    df[(df.數學 == 15) & (df.英文 == 15)],
)

show(
    "找出數學或英文滿級分的同學",
    df[(df.數學 == 15) | (df.英文 == 15)],
)


# =========================
# 7. 刪除一行或一列
# =========================

# 刪掉一行，也就是刪掉欄位：axis=1
show("刪掉總級分欄位，但不改變原本 df", df.drop("總級分", axis=1))

# 真正改變原本 DataFrame，要加 inplace=True
# 這裡先複製一份，避免後面範例不能繼續使用總級分
df_no_total = df.copy()
df_no_total.drop("總級分", axis=1, inplace=True)
show("使用 inplace=True 刪掉總級分欄位", df_no_total.head())

# 刪掉一列，也就是刪掉指定 index 的資料
df_drop_index_5 = df.drop(5)
show("刪掉 index = 5 的那一列", df_drop_index_5.head())

# 刪掉符合條件的列：例如刪掉姓名等於某人的資料
name_to_drop = df.loc[5, "姓名"]
rows_to_drop = df[df.姓名 == name_to_drop].index
df_drop_by_condition = df.drop(rows_to_drop)
show(f"刪掉姓名為 {name_to_drop} 的資料", df_drop_by_condition)


# =========================
# 8. 真實股價資料：Apple 股價
# =========================

# 注意：pandas_datareader / Yahoo Finance 可能因網路或套件版本失效。
# 如果有 data/aapl.csv，建議直接讀本機資料。

try:
    stock = pd.read_csv("data/aapl.csv", index_col="Date", parse_dates=True)
except FileNotFoundError:
    try:
        import pandas_datareader.data as web

        stock = web.DataReader("AAPL", "yahoo")
    except Exception as e:
        print("\n無法讀取 AAPL 線上資料，改用假資料示範。原因：", e)
        dates = pd.date_range("2020-01-01", periods=400, freq="B")
        rng = np.random.default_rng(7)
        close = 100 + np.cumsum(rng.normal(0, 1, len(dates)))
        stock = pd.DataFrame(
            {
                "Open": close + rng.normal(0, 0.5, len(dates)),
                "High": close + rng.normal(1, 0.5, len(dates)),
                "Low": close - rng.normal(1, 0.5, len(dates)),
                "Close": close,
                "Volume": rng.integers(10_000_000, 200_000_000, len(dates)),
                "Adj Close": close,
            },
            index=dates,
        )
        stock.index.name = "Date"

show("AAPL 股價前五筆", stock.head())

# 只取最後 300 個交易日
stock = stock[-300:]
show("最後 300 個交易日", stock.head())

# 收盤價
stock.Close.plot(legend=True, title="AAPL Close")
plt.show()

# 20 日移動平均
stock.Close.plot(legend=True)
stock.Close.rolling(20).mean().plot(label="MA20", legend=True)
plt.title("AAPL Close with MA20")
plt.show()

# 20 日和 60 日移動平均
stock.Close.plot(legend=True)
stock.Close.rolling(20).mean().plot(label="MA20", legend=True)
stock.Close.rolling(60).mean().plot(label="MA60", legend=True)
plt.title("AAPL Close with MA20 and MA60")
plt.show()

# 準備做預測：用昨天收盤價預測今天收盤價
x = stock.Close.values[:-1]
y = stock.Close.values[1:]

show("stock 筆數", len(stock))
show("x 筆數：昨天收盤價", len(x))
show("y 筆數：今天收盤價", len(y))

plt.scatter(x, y)
plt.xlabel("昨天收盤價")
plt.ylabel("今天收盤價")
plt.title("昨天收盤價 vs 今天收盤價")
plt.show()


# =========================
# 9. 手工打造一個 DataFrame
# =========================

np.random.seed(42)

mydata = np.random.randn(4, 3)
show("原始二維資料 mydata", mydata)

df2 = pd.DataFrame(mydata, columns=list("ABC"))
show("df2", df2)

df3 = pd.DataFrame(np.random.randn(3, 3), columns=list("ABC"))
show("df3", df3)

# 兩個表格上下貼起來
df4 = pd.concat([df2, df3])
show("上下合併 df2 與 df3", df4)

# 重設 index
df4 = df4.reset_index(drop=True)
show("重設 index 後的 df4", df4)

# 橫向貼起來
# axis=1 表示左右合併；列數不同時，缺少的位置會補 NaN
df5 = pd.concat([df2, df3], axis=1)
show("橫向合併 df2 與 df3", df5)


# =========================
# 10. 大一點的例子：製作假的學測資料
# =========================

# 產生姓名表
names = pd.DataFrame({"姓名": make_fake_grades(n=100, seed=100)["姓名"]})
show("姓名資料", names)

# 產生成績表
df_grades = pd.DataFrame(
    np.random.randint(6, 16, (100, 5)),
    columns=["國文", "英文", "數學", "社會", "自然"],
)
show("成績資料", df_grades)

# 合併姓名與成績
final_grades = pd.concat([names, df_grades], axis=1)
show("合併姓名與成績", final_grades)

# 輸出成 CSV，index=False 可以避免多出 Unnamed: 0 欄位
final_grades.to_csv("grades_output.csv", index=False, encoding="utf-8-sig")
print("\n已輸出 grades_output.csv")
