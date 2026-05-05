# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 0. 基本設定
# ============================================================

np.random.seed(42)

OUTPUT_DIR = "scikit_learn_basic_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# 1. 載入 Iris 資料集
# ============================================================

print("=" * 80)
print("1. 載入 Iris 資料集")
print("=" * 80)

iris = datasets.load_iris()

print("iris 的 key：")
print(iris.keys())

print("\nIris 資料集說明：")
print(iris.DESCR[:1000])
print("...略")


# ============================================================
# 2. 觀察原始資料
# ============================================================

print("\n" + "=" * 80)
print("2. 觀察原始資料")
print("=" * 80)

data = iris.data
target = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("data shape:", data.shape)
print("target shape:", target.shape)

print("\n特徵名稱：")
print(feature_names)

print("\n類別名稱：")
print(target_names)

print("\n前 10 筆特徵資料：")
print(data[:10])

print("\n前 10 筆目標類別：")
print(target[:10])

print("\n資料型別：")
print(type(data))
print(type(target))


# ============================================================
# 3. 建立 pandas DataFrame
# ============================================================

print("\n" + "=" * 80)
print("3. 建立 pandas DataFrame")
print("=" * 80)

x = pd.DataFrame(data, columns=feature_names)
y = pd.DataFrame(target, columns=["target"])

print("特徵資料 x 前 10 筆：")
print(x.head(10))

print("\n目標資料 y 前 10 筆：")
print(y.head(10))

# 合併特徵與目標欄位
iris_df = pd.concat([x, y], axis=1)

# 建立可讀性較高的 target_name 欄位
iris_df["target_name"] = iris_df["target"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

print("\n合併後的 iris_df 前 10 筆：")
print(iris_df.head(10))

print("\niris_df 基本資訊：")
print(iris_df.info())

print("\niris_df 統計摘要：")
print(iris_df.describe())


# ============================================================
# 4. 資料視覺化
# ============================================================

print("\n" + "=" * 80)
print("4. 資料視覺化")
print("=" * 80)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=iris_df,
    x="sepal length (cm)",
    y="sepal width (cm)",
    hue="target_name"
)
plt.title("Iris: Sepal Length vs Sepal Width")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "iris_sepal_scatter.png"))
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=iris_df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="target_name"
)
plt.title("Iris: Petal Length vs Petal Width")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "iris_petal_scatter.png"))
plt.close()

print(f"圖片已輸出到資料夾：{OUTPUT_DIR}")


# ============================================================
# 5. 只使用前兩個特徵
# ============================================================

print("\n" + "=" * 80)
print("5. 只使用前兩個特徵")
print("=" * 80)

# Notebook 中提到：
# we only take the first two features.
X_two_features = iris_df[["sepal length (cm)", "sepal width (cm)"]]
y_all = iris_df["target"]

print("X_two_features 前 5 筆：")
print(X_two_features.head())

print("\ny_all 前 5 筆：")
print(y_all.head())


# ============================================================
# 6. 只選擇 target 為 0 與 1 的資料
# ============================================================

print("\n" + "=" * 80)
print("6. 只選擇 target 為 0 與 1 的資料")
print("=" * 80)

binary_df = iris_df[iris_df["target"].isin([0, 1])].copy()

X = binary_df[["sepal length (cm)", "sepal width (cm)"]]
y = binary_df["target"]

print("binary_df shape:", binary_df.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nbinary_df 前 10 筆：")
print(binary_df.head(10))

print("\ny 類別數量：")
print(y.value_counts())


# ============================================================
# 7. 切分訓練集與測試集
# ============================================================

print("\n" + "=" * 80)
print("7. 切分訓練集與測試集")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\nX_train 前 5 筆：")
print(X_train.head())

print("\nX_test 前 5 筆：")
print(X_test.head())


# ============================================================
# 8. 手動 Standardization / Z-score
# ============================================================

print("\n" + "=" * 80)
print("8. 手動 Standardization / Z-score")
print("=" * 80)

def norm_stats(dfs):
    """
    計算 DataFrame 每個欄位的：
    - 最小值
    - 最大值
    - 平均值
    - 標準差
    """
    minimum = np.min(dfs)
    maximum = np.max(dfs)
    mu = np.mean(dfs)
    sigma = np.std(dfs)
    return minimum, maximum, mu, sigma


def z_score(col, stats):
    """
    使用訓練資料的平均值與標準差做標準化：
    z = (x - mean) / std
    """
    minimum, maximum, mu, sigma = stats

    df = pd.DataFrame(index=col.index)

    for c in col.columns:
        df[c] = (col[c] - mu[c]) / sigma[c]

    return df


stats = norm_stats(X_train)

X_train_z = z_score(X_train, stats)
X_test_z = z_score(X_test, stats)

arr_x_train = np.array(X_train_z)
arr_y_train = np.array(y_train)

print("手動 z-score 後的 X_train 前 5 筆：")
print(X_train_z.head())

print("\n轉成 numpy array 後，arr_x_train 前 5 筆：")
print(arr_x_train[:5])

print("\narr_y_train 前 5 筆：")
print(arr_y_train[:5])


# ============================================================
# 9. 使用 sklearn StandardScaler
# ============================================================

print("\n" + "=" * 80)
print("9. 使用 sklearn StandardScaler")
print("=" * 80)

sc = StandardScaler().fit(X_train)

print("StandardScaler mean：")
print(sc.mean_)

print("\nStandardScaler scale / standard deviation：")
print(sc.scale_)

# transform: (x - mean) / std
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

print("\nX_train_std 前 5 筆：")
print(X_train_std[:5])

print("\nX_test_std 前 5 筆：")
print(X_test_std[:5])

print("\nmean of X_train_std:", np.round(X_train_std.mean(), 4))
print("std of X_train_std:", X_train_std.std())

# 注意：
# 正確流程是：
# 1. scaler.fit(X_train)
# 2. scaler.transform(X_train)
# 3. scaler.transform(X_test)
#
# 不建議對 X_test 使用 fit_transform，因為這會讓測試資料自己的統計值參與轉換，
# 造成資料洩漏 data leakage。


# ============================================================
# 10. 使用 fit_transform 簡化訓練資料標準化
# ============================================================

print("\n" + "=" * 80)
print("10. 使用 fit_transform 簡化訓練資料標準化")
print("=" * 80)

sc2 = StandardScaler()

X_train_std_2 = sc2.fit_transform(X_train)
X_test_std_2 = sc2.transform(X_test)

print("X_train_std_2 前 5 筆：")
print(X_train_std_2[:5])

print("\nX_test_std_2 前 5 筆：")
print(X_test_std_2[:5])


# ============================================================
# 11. Min-Max Normalization 手動練習
# ============================================================

print("\n" + "=" * 80)
print("11. Min-Max Normalization 手動練習")
print("=" * 80)

x1 = np.random.normal(50, 6, 100)
y1 = np.random.normal(5, 0.5, 100)

x2 = np.random.normal(30, 6, 100)
y2 = np.random.normal(4, 0.5, 100)

plt.figure(figsize=(8, 6))
plt.scatter(x1, y1, c="b", marker="s", s=20, alpha=0.8, label="Group 1")
plt.scatter(x2, y2, c="r", marker="^", s=20, alpha=0.8, label="Group 2")
plt.title("Random Data Before Min-Max Scaling")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "random_data_before_minmax.png"))
plt.close()

print("x1 平均值:", np.sum(x1) / len(x1))
print("x2 平均值:", np.sum(x2) / len(x2))

x_val = np.concatenate((x1, x2))
y_val = np.concatenate((y1, y2))

print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)


def minmax_norm(X):
    """
    Min-Max Normalization：
    X' = (X - X_min) / (X_max - X_min)
    """
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))


x_val_minmax_manual = minmax_norm(x_val)

print("\n手動 minmax_norm(x_val[:10])：")
print(x_val_minmax_manual[:10])

plt.figure(figsize=(8, 6))
plt.scatter(x_val_minmax_manual[:100], y_val[:100], c="b", marker="s", s=20, alpha=0.8, label="Group 1")
plt.scatter(x_val_minmax_manual[100:], y_val[100:], c="r", marker="^", s=20, alpha=0.8, label="Group 2")
plt.title("X Value After Manual Min-Max Scaling")
plt.xlabel("x normalized")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "random_data_after_manual_minmax.png"))
plt.close()


# ============================================================
# 12. 使用 sklearn MinMaxScaler
# ============================================================

print("\n" + "=" * 80)
print("12. 使用 sklearn MinMaxScaler")
print("=" * 80)

x_val_2d = x_val.reshape(-1, 1)

scaler = MinMaxScaler().fit(x_val_2d)

print("MinMaxScaler data_min_：")
print(scaler.data_min_)

print("\nMinMaxScaler data_max_：")
print(scaler.data_max_)

x_val_minmax_sklearn = scaler.transform(x_val_2d)

print("\n使用 MinMaxScaler 轉換後前 10 筆：")
print(x_val_minmax_sklearn[:10])


# ============================================================
# 13. 額外練習：使用 Logistic Regression 完成基本機器學習流程
# ============================================================

print("\n" + "=" * 80)
print("13. 額外練習：Logistic Regression 基本流程")
print("=" * 80)

# 基本架構：
# 1. 讀取資料與 preprocessing
# 2. 切分訓練集與測試集
# 3. 模型配適 fit
# 4. 預測 predict
# 5. 評估 accuracy / confusion matrix / classification report

model = LogisticRegression(random_state=42)

# 使用標準化後的資料訓練
model.fit(X_train_std, y_train)

y_pred = model.predict(X_test_std)

acc = accuracy_score(y_test, y_pred)

print("預測結果 y_pred：")
print(y_pred)

print("\n實際答案 y_test：")
print(np.array(y_test))

print("\nAccuracy:", acc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["setosa", "versicolor"]))


# ============================================================
# 14. 畫出決策結果散點圖
# ============================================================

print("\n" + "=" * 80)
print("14. 畫出測試資料預測結果")
print("=" * 80)

test_plot_df = X_test.copy()
test_plot_df["actual"] = y_test.values
test_plot_df["predicted"] = y_pred

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=test_plot_df,
    x="sepal length (cm)",
    y="sepal width (cm)",
    hue="predicted",
    style="actual",
    s=100
)
plt.title("Logistic Regression Prediction Result")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "logistic_regression_prediction_result.png"))
plt.close()

print(f"所有圖片已輸出到資料夾：{OUTPUT_DIR}")

