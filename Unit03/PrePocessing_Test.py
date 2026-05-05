import os
from io import StringIO

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


# =============================================================================
# 工具函式
# =============================================================================

def print_title(title: str) -> None:
    """印出段落標題，方便閱讀輸出結果。"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_subtitle(title: str) -> None:
    """印出小標題。"""
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


# =============================================================================
# 1. 缺失值處理 Missing Data
# =============================================================================

print_title("1. 缺失值處理 Missing Data")

csv_data = """A,B,C,D,E
5.0,2.0,3.0,,6
1.0,6.0,,8.0,5
0.0,11.0,12.0,4.0,5
3.0,,3.0,5.0,
5.0,1.0,4.0,2.0,4
"""

# notebook 中有讀取 preText.csv，這裡做成可獨立執行：
# 如果同資料夾有 preText.csv 就讀檔，否則用內建練習資料。
if os.path.exists("preText.csv"):
    df = pd.read_csv("preText.csv")
else:
    df = pd.read_csv(StringIO(csv_data))

print_subtitle("原始資料")
print(df)

print_subtitle("檢查每個欄位是否為空值：df.isnull()")
print(df.isnull())

print_subtitle("每個欄位的空值數量：df.isnull().sum()")
print(df.isnull().sum())

print_subtitle("整份資料總共有幾個空值：df.isnull().sum().sum()")
print(df.isnull().sum().sum())


# -----------------------------------------------------------------------------
# 1-1 dropna：刪除缺失值
# -----------------------------------------------------------------------------

print_subtitle("dropna()：只要該列有任一空值，就刪除整列")
print(df.dropna())

print_subtitle("dropna(axis=1)：只要該欄有任一空值，就刪除整欄")
print(df.dropna(axis=1))

print_subtitle("dropna(how='all')：只有整列全部都是空值時才刪除")
print(df.dropna(how="all"))

print_subtitle("dropna(thresh=4)：至少要有 4 個非空值，該列才保留")
print(df.dropna(thresh=4))

print_subtitle("dropna(subset=['C'])：只有 C 欄為空時才刪除該列")
print(df.dropna(subset=["C"]))


# -----------------------------------------------------------------------------
# 1-2 fillna：補缺失值
# -----------------------------------------------------------------------------

print_subtitle("fillna(0)：所有空值補 0")
print(df.fillna(0))

print_subtitle("用每個欄位的平均數補值")
print(df.fillna(df.mean(numeric_only=True)))

print_subtitle("用每個欄位的中位數補值")
print(df.fillna(df.median(numeric_only=True)))

print_subtitle("用每個欄位的眾數補值")
mode_values = df.mode().iloc[0]
print(df.fillna(mode_values))

print_subtitle("用前一筆資料補值：ffill")
print(df.ffill())

print_subtitle("用後一筆資料補值：bfill")
print(df.bfill())


# =============================================================================
# 2. 類別資料處理 Categorical Data
# =============================================================================

print_title("2. 類別資料處理 Categorical Data")

# color：無序類別，例如 red / green / blue 沒有大小順序
# size：有序類別，例如 M < L < XL
# classlabel：類別標籤

df2 = pd.DataFrame(
    [
        ["green", "M", 10.1, 1],
        ["red", "L", 13.5, 2],
        ["blue", "XL", 15.3, 1],
    ],
    columns=["color", "size", "price", "classlabel"],
)

print_subtitle("原始類別資料")
print(df2)


# -----------------------------------------------------------------------------
# 2-1 有序類別：手動 mapping
# -----------------------------------------------------------------------------

print_subtitle("有序類別 size：使用 mapping 轉成數值")
size_mapping = {
    "M": 1,
    "L": 2,
    "XL": 3,
}

df2_size_encoded = df2.copy()
df2_size_encoded["size"] = df2_size_encoded["size"].map(size_mapping)
print(df2_size_encoded)

print_subtitle("將 size 數值轉回原本的文字類別")
inv_size_mapping = {value: key for key, value in size_mapping.items()}
df2_size_decoded = df2_size_encoded.copy()
df2_size_decoded["size"] = df2_size_decoded["size"].map(inv_size_mapping)
print(df2_size_decoded)


# -----------------------------------------------------------------------------
# 2-2 類別標籤 Label Encoding
# -----------------------------------------------------------------------------

print_subtitle("classlabel 轉成從 0 開始的類別編號")
class_mapping = {label: idx for idx, label in enumerate(np.unique(df2["classlabel"]))}
print("class_mapping:", class_mapping)

df2_class_encoded = df2.copy()
df2_class_encoded["classlabel"] = df2_class_encoded["classlabel"].map(class_mapping)
print(df2_class_encoded)

print_subtitle("使用 sklearn LabelEncoder")
label_encoder = LabelEncoder()
encoded_class = label_encoder.fit_transform(df2["classlabel"])
print("encoded_class:", encoded_class)
print("classes_:", label_encoder.classes_)
print("inverse_transform:", label_encoder.inverse_transform(encoded_class))


# -----------------------------------------------------------------------------
# 2-3 無序類別：One-hot Encoding
# -----------------------------------------------------------------------------

print_subtitle("pandas get_dummies：只針對 color 做 One-hot Encoding")
color_dummies = pd.get_dummies(df2["color"], prefix="color", dtype=int)
print(color_dummies)

print_subtitle("把 color 的 One-hot Encoding 結果合併回 df2")
df2_onehot = pd.concat([df2.drop("color", axis=1), color_dummies], axis=1)
print(df2_onehot)

print_subtitle("pandas get_dummies：直接對整個 DataFrame 做 One-hot Encoding")
print(pd.get_dummies(df2, dtype=int))

print_subtitle("get_dummies(drop_first=True)：避免 Dummy Variable Trap")
print(pd.get_dummies(df2, drop_first=True, dtype=int))


# -----------------------------------------------------------------------------
# 2-4 sklearn OneHotEncoder / ColumnTransformer
# -----------------------------------------------------------------------------

print_subtitle("sklearn OneHotEncoder：將 color 欄位轉成 One-hot")
onehot_encoder = OneHotEncoder(sparse_output=False)
encoded_color = onehot_encoder.fit_transform(df2[["color"]])
print(encoded_color)
print("categories_:", onehot_encoder.categories_)

print_subtitle("ColumnTransformer：只轉換 color，其餘欄位保留")
column_transformer = ColumnTransformer(
    transformers=[
        ("color_onehot", OneHotEncoder(sparse_output=False), ["color"]),
    ],
    remainder="passthrough",
)

transformed_data = column_transformer.fit_transform(df2)
print(transformed_data)

feature_names = column_transformer.get_feature_names_out()
print("feature_names:", feature_names)


# =============================================================================
# 3. 資料特徵縮放 Feature Scaling
# =============================================================================

print_title("3. 資料特徵縮放 Feature Scaling")

iris = datasets.load_iris()
x = pd.DataFrame(iris["data"], columns=iris["feature_names"])
y = pd.DataFrame(iris["target"], columns=["target_names"])
data = pd.concat([x, y], axis=1)

print_subtitle("Iris target_names")
print(iris["target_names"])

print_subtitle("Iris 原始資料前 5 筆")
print(data.head())

print_subtitle("Iris 特徵資料 x 前 5 筆")
print(x.head())


# -----------------------------------------------------------------------------
# 3-1 Normalization：0 到 1 區間縮放
# 公式：x_norm = (x - x_min) / (x_max - x_min)
# -----------------------------------------------------------------------------

print_subtitle("手動計算 Normalization：0 到 1 區間縮放")
x_norm_manual = (x - x.min()) / (x.max() - x.min())
print(x_norm_manual.head())

print_subtitle("檢查 Normalization 後每個欄位的最小值")
print(x_norm_manual.min())

print_subtitle("檢查 Normalization 後每個欄位的最大值")
print(x_norm_manual.max())

print_subtitle("使用 sklearn MinMaxScaler 做 Normalization")
minmax_scaler = MinMaxScaler()
x_norm_sklearn = pd.DataFrame(
    minmax_scaler.fit_transform(x),
    columns=x.columns,
)
print(x_norm_sklearn.head())


# -----------------------------------------------------------------------------
# 3-2 Standardization：標準化
# 公式：x_std = (x - mean) / std
# 標準化後平均值接近 0，標準差接近 1
# -----------------------------------------------------------------------------

print_subtitle("手動計算 Standardization：平均值 0，標準差 1")
x_std_manual = (x - x.mean()) / x.std()
print(x_std_manual.head())

print_subtitle("檢查 Standardization 後每個欄位的平均值")
print(x_std_manual.mean())

print_subtitle("檢查 Standardization 後每個欄位的標準差")
print(x_std_manual.std())

print_subtitle("使用 sklearn StandardScaler 做 Standardization")
standard_scaler = StandardScaler()
x_std_sklearn = pd.DataFrame(
    standard_scaler.fit_transform(x),
    columns=x.columns,
)
print(x_std_sklearn.head())

print_subtitle("sklearn StandardScaler 後的平均值")
print(x_std_sklearn.mean())

print_subtitle("sklearn StandardScaler 後的標準差，ddof=0 才會接近 1")
print(x_std_sklearn.std(ddof=0))


# =============================================================================
# 4. 小型整合練習：缺值 + 類別 + 縮放
# =============================================================================

print_title("4. 小型整合練習：缺值 + 類別 + 縮放")

practice_df = pd.DataFrame(
    {
        "color": ["red", "green", "blue", "green", np.nan],
        "size": ["M", "L", "XL", np.nan, "M"],
        "price": [10.5, 13.2, np.nan, 15.6, 9.9],
        "weight": [1.2, np.nan, 1.7, 1.5, 1.1],
    }
)

print_subtitle("練習資料：原始資料")
print(practice_df)

print_subtitle("步驟 1：數值欄位用平均數補值")
practice_processed = practice_df.copy()
numeric_columns = ["price", "weight"]
practice_processed[numeric_columns] = practice_processed[numeric_columns].fillna(
    practice_processed[numeric_columns].mean()
)
print(practice_processed)

print_subtitle("步驟 2：類別欄位用眾數補值")
categorical_columns = ["color", "size"]
for col in categorical_columns:
    practice_processed[col] = practice_processed[col].fillna(practice_processed[col].mode()[0])
print(practice_processed)

print_subtitle("步驟 3：有序類別 size 做 mapping")
practice_processed["size"] = practice_processed["size"].map(size_mapping)
print(practice_processed)

print_subtitle("步驟 4：無序類別 color 做 One-hot Encoding")
practice_processed = pd.get_dummies(practice_processed, columns=["color"], dtype=int)
print(practice_processed)

print_subtitle("步驟 5：數值欄位做 Standardization")
scale_columns = ["size", "price", "weight"]
practice_scaled = practice_processed.copy()
practice_scaled[scale_columns] = StandardScaler().fit_transform(practice_scaled[scale_columns])
print(practice_scaled)

print_title("完成：PrePocessing_Test.py 已執行完畢")
