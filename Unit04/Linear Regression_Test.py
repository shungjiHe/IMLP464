# -*- coding: utf-8 -*-
import os
import matplotlib
matplotlib.use("Agg")  # 讓程式在沒有 GUI 的環境也可以輸出圖片

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


OUTPUT_DIR = "linear_regression_output"


def make_output_dir():
    """建立圖片輸出資料夾。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """建立 notebook 中提供的 x 與 y 資料。"""
    x = np.array([
        0.0, 0.20408163, 0.40816327, 0.6122449, 0.81632653,
        1.02040816, 1.2244898, 1.42857143, 1.63265306, 1.83673469,
        2.04081633, 2.24489796, 2.44897959, 2.65306122, 2.85714286,
        3.06122449, 3.26530612, 3.46938776, 3.67346939, 3.87755102,
        4.08163265, 4.28571429, 4.48979592, 4.69387755, 4.89795918,
        5.10204082, 5.30612245, 5.51020408, 5.71428571, 5.91836735,
        6.12244898, 6.32653061, 6.53061224, 6.73469388, 6.93877551,
        7.14285714, 7.34693878, 7.55102041, 7.75510204, 7.95918367,
        8.16326531, 8.36734694, 8.57142857, 8.7755102, 8.97959184,
        9.18367347, 9.3877551, 9.59183673, 9.79591837, 10.0
    ])

    y = np.array([
        0.85848224, -0.10657947, 1.42771901, 0.53554778, 1.20216826,
        1.81330509, 1.88362644, 2.23557653, 2.7384889, 3.41174583,
        4.08573636, 3.82529502, 4.39723111, 4.8852381, 4.70092778,
        4.66993962, 6.05133235, 5.44529881, 7.22571332, 6.79423911,
        7.05424438, 7.00413058, 7.98149596, 7.00044008, 7.95903855,
        9.96125238, 9.06040794, 9.56018295, 9.30035956, 9.26517614,
        9.56401824, 10.07659844, 11.56755942, 11.38956185, 11.83586027,
        12.45642786, 11.58403954, 11.60186428, 13.88486667, 13.35550112,
        13.93938726, 13.31678277, 13.69551472, 14.76548676, 14.81731598,
        14.9659187, 15.19213921, 15.28195017, 15.97997265, 16.41258817
    ])

    return x, y


def save_scatter_plot(x, y, title, filename):
    """畫出原始資料散佈圖。"""
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()


def basic_linear_regression(x, y):
    """
    基礎題：
    使用所有資料訓練 LinearRegression，
    並算出斜率 w 與截距 b。
    """
    print("=" * 60)
    print("基礎題：算出斜率 w 與截距 b")
    print("=" * 60)

    print("原本 x shape:", x.shape)

    # scikit-learn 的 LinearRegression 需要 2D X：
    # 原本: [x1, x2, ..., x50]
    # 轉成: [[x1], [x2], ..., [x50]]
    X = x.reshape(-1, 1)

    print("reshape 後 X shape:", X.shape)

    model = LinearRegression()
    model.fit(X, y)

    w = model.coef_[0]
    b = model.intercept_

    print("斜率 w:", w)
    print("截距 b:", b)
    print(f"線性函數：y = {w:.6f}x + {b:.6f}")

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print("全部資料 MSE:", mse)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="Real Data")
    plt.plot(x, y_pred, label="Regression Line")
    plt.title("Basic Linear Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_basic_linear_regression.png"), dpi=150)
    plt.close()

    return model


def train_test_linear_regression(x, y):
    """
    進階題：
    將資料切割成訓練資料 80%、測試資料 20%，
    只用訓練資料 fit 模型，再用測試資料評估。
    """
    print()
    print("=" * 60)
    print("進階題：切割資料集做訓練與預測")
    print("=" * 60)

    X = x.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.8,
        test_size=0.2,
        random_state=20
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # 畫出訓練資料集
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train)
    plt.title("Training Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_training_data.png"), dpi=150)
    plt.close()

    # 使用訓練資料 fit 模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    w = model.coef_[0]
    b = model.intercept_

    print("訓練後斜率 w:", w)
    print("訓練後截距 b:", b)
    print(f"訓練後線性函數：y = {w:.6f}x + {b:.6f}")

    # 訓練階段預測
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    print("訓練資料 MSE:", train_mse)

    # 畫出訓練資料的真實值與預測值
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, label="Real y_train")
    plt.scatter(X_train, y_train_pred, label="Predicted y_train")
    plt.title("Training Data: Real vs Predicted")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_train_real_vs_predicted.png"), dpi=150)
    plt.close()

    # 測試階段預測
    y_test_pred = model.predict(X_test)

    print()
    print("測試資料 X_test:")
    print(X_test.ravel())

    print("測試資料真實 y_test:")
    print(y_test)

    print("測試資料預測 y_test_pred:")
    print(y_test_pred)

    # 題目提示：使用 X = 2.44897959，預測出來數值應該約為 4.3025375
    single_x = np.array([[2.44897959]])
    single_y_pred = model.predict(single_x)
    print()
    print("單筆預測 X = 2.44897959")
    print("預測 y =", single_y_pred[0])

    test_mse = mean_squared_error(y_test, y_test_pred)
    print("測試資料 MSE:", test_mse)

    # 畫出測試資料的真實值與預測值
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, label="Real y_test")
    plt.scatter(X_test, y_test_pred, label="Predicted y_test")
    plt.title("Test Data: Real vs Predicted")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_test_real_vs_predicted.png"), dpi=150)
    plt.close()

    # 為了讓線看起來順，先依照 x 排序後再畫線
    sorted_index = np.argsort(X_test.ravel())
    X_test_sorted = X_test[sorted_index]
    y_test_sorted = y_test[sorted_index]
    y_test_pred_sorted = y_test_pred[sorted_index]

    plt.figure(figsize=(8, 5))
    plt.scatter(X_test_sorted, y_test_sorted, label="Real y_test")
    plt.plot(X_test_sorted, y_test_pred_sorted, label="Prediction Line")
    plt.title("Test Data with Prediction Line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_test_prediction_line.png"), dpi=150)
    plt.close()

    return model, train_mse, test_mse


def main():
    make_output_dir()

    x, y = load_data()

    print("資料筆數:", len(x))
    print("x 前 5 筆:", x[:5])
    print("y 前 5 筆:", y[:5])

    save_scatter_plot(
        x,
        y,
        title="Original Data",
        filename="00_original_data.png"
    )

    basic_model = basic_linear_regression(x, y)
    advanced_model, train_mse, test_mse = train_test_linear_regression(x, y)

    print()
    print("=" * 60)
    print("執行完成")
    print("=" * 60)
    print(f"圖片已輸出到資料夾：{OUTPUT_DIR}/")
    print("基礎模型斜率:", basic_model.coef_[0])
    print("進階模型訓練 MSE:", train_mse)
    print("進階模型測試 MSE:", test_mse)


if __name__ == "__main__":
    main()
