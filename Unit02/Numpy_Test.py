# Unit02內Numpy, Pandas, Matplolib, Seaborn實作完成
# Seaborn的實作藏在Data Visualization Package影片後半段哦

import numpy as np

# 1. 定義成績清單並轉換為 NumPy 數組 (ndarray)
grades_list = [77, 85, 56, 90, 66]
grades = np.array(grades_list)

# 2. 使用 np.mean() 直接計算平均值
average = np.mean(grades)

# 3. 其他常見的統計資訊 (選配)
total_score = np.sum(grades)    # 總分
max_score = np.max(grades)      # 最高分
min_score = np.min(grades)      # 最低分
std_dev = np.std(grades)        # 標準差 (了解成績離散程度)

# 4. 輸出結果
print(f"成績數組：{grades}")
print(f"總分：{total_score}")
print(f"平均成績：{average}")
print(f"最高分：{max_score} / 最低分：{min_score}")
print(f"標準差：{std_dev:.2f}")

#####

usd_to_twd = 31.71
prices = np.array([1096.95, 596.95, 896.95])

twd_prices = prices * usd_to_twd

print(twd_prices)

###

grades = np.array([85, 70, 80])
weights = np.array([0.2, 0.35, 0.45])

weighted_average = np.sum(grades * weights)

print(weighted_average)

###
A = np.array([
    0.7372234, 0.6126442, 0.27626397, 0.28984569, 0.4988311,
    0.42021495, 0.4515489, 0.15328583, 0.93455268, 0.23537373,
    0.24761985, 0.41945762, 0.44871061, 0.66726833, 0.15622678,
    0.70315927, 0.98138833, 0.8233088, 0.85394749, 0.14797509,
    0.32816167, 0.64607213, 0.96849395, 0.90799427, 0.19579351,
    0.35618479, 0.30311169, 0.32568614, 0.66885309, 0.67844806,
    0.18201376, 0.69592441, 0.11077285, 0.22884185, 0.29488536,
    0.90457337, 0.19872841, 0.38361489, 0.35731267, 0.76200102,
    0.92178147, 0.44190395, 0.07941469, 0.03253205, 0.44805018,
    0.38885104, 0.15473767, 0.55240742, 0.80182196, 0.70921062
])
print(A.shape)

B = A.reshape(5, 10)

print(B)
print(B.shape)

B = A.reshape(5, 10)

E = B.ravel()

print(E)
print(E.shape)

ZeroArray = np.zeros(10)
print(ZeroArray)

oneArray = np.ones(10)
print(oneArray)

EyeArray = np.eye(5)
print(EyeArray)

LinearSpaceArray = np.linspace(0, 10, 100)
print(LinearSpaceArray)

ArangeArray = np.arange(1, 10, 0.2)

print(ArangeArray)

A = np.array([
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9]
])

B=np.sum(A, axis=0)
print(B)

C=np.sum(A, axis=0)
print(C)

D=A.sum()
print(D)

L = np.array([3, -2, -1, 5, 7, -3])

result = L[L > 0]

print(result)