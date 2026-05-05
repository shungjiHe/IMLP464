import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    """讀取資料檔。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"找不到資料檔：{file_path}\n"
            "請確認資料檔是否放在 data/Classified Data，"
            "或修改程式中的 DATA_PATH。"
        )

    # index_col=0 代表將第一欄設定成 index
    return pd.read_csv(file_path, index_col=0)


def prepare_data(df: pd.DataFrame):
    """切分特徵 X 與目標 y，並進行標準化與訓練/測試資料切分。"""
    # TARGET CLASS 是要預測的分類結果，其餘欄位都是特徵
    x = df.drop("TARGET CLASS", axis=1)
    y = df["TARGET CLASS"]

    # KNN 會根據距離判斷分類，因此特徵尺度會影響模型表現
    scaler = StandardScaler()
    scaler.fit(x)

    scaled_features = scaler.transform(x)
    x_scaled = pd.DataFrame(scaled_features, columns=x.columns)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled,
        y,
        test_size=0.30,
        random_state=101,
    )

    return x_train, x_test, y_train, y_test


def train_and_evaluate_knn(k: int, x_train, x_test, y_train, y_test):
    """訓練指定 K 值的 KNN 模型，並輸出評估結果。"""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    pred = knn.predict(x_test)

    print(f"WITH K={k}")
    print("-" * 40)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print()
    print("Classification Report:")
    print(classification_report(y_test, pred))

    return pred


def plot_confusion_matrix(y_test, pred, title: str = "Confusion Matrix"):
    """畫出 Confusion Matrix 熱力圖。"""
    matrix = confusion_matrix(y_test, pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


def calculate_error_rate(x_train, x_test, y_train, y_test, max_k: int = 60):
    """計算 K=1 到 max_k-1 的錯誤率。"""
    error_rate = []

    for k in range(1, max_k):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        pred_k = knn.predict(x_test)

        # pred_k != y_test 會得到 True/False，mean() 可計算錯誤比例
        error_rate.append(np.mean(pred_k != y_test))

    return error_rate


def plot_error_rate(error_rate):
    """畫出不同 K 值對應的錯誤率。"""
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(error_rate) + 1),
        error_rate,
        color="blue",
        linestyle="dashed",
        marker="o",
        markerfacecolor="red",
        markersize=8,
    )
    plt.title("Error Rate vs. K Value")
    plt.xlabel("K")
    plt.ylabel("Error Rate")
    plt.tight_layout()
    plt.show()


def main():
    # 原 notebook 使用的資料路徑
    DATA_PATH = "data/Classified Data"

    df = load_data(DATA_PATH)

    print("資料前五筆：")
    print(df.head())
    print()

    x_train, x_test, y_train, y_test = prepare_data(df)

    # 先從 K=1 開始測試
    pred_k1 = train_and_evaluate_knn(1, x_train, x_test, y_train, y_test)
    plot_confusion_matrix(y_test, pred_k1, title="Confusion Matrix - K=1")

    # 計算 K=1~59 的錯誤率
    error_rate = calculate_error_rate(x_train, x_test, y_train, y_test, max_k=60)
    plot_error_rate(error_rate)

    # 找出錯誤率最低的 K 值
    best_k = error_rate.index(min(error_rate)) + 1
    print(f"錯誤率最低的 K 值：{best_k}")
    print(f"最低錯誤率：{min(error_rate):.4f}")
    print()

    # 使用最佳 K 值重新訓練並評估
    best_pred = train_and_evaluate_knn(best_k, x_train, x_test, y_train, y_test)
    plot_confusion_matrix(y_test, best_pred, title=f"Confusion Matrix - K={best_k}")


if __name__ == "__main__":
    main()
