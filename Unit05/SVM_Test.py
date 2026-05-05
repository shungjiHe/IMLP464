# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC


def load_data():
    """Step 1. 載入 scikit-learn 內建乳癌資料集。"""
    breast_cancer = datasets.load_breast_cancer()
    x = breast_cancer.data       # 特徵資料，例如半徑、紋理、面積等
    y = breast_cancer.target     # 分類結果：0 = malignant, 1 = benign
    return breast_cancer, x, y


def split_data(x, y):
    """Step 2. 區分訓練集與測試集。"""
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    """Step 3. 建立並訓練 SVM 分類模型。"""
    model = SVC(kernel="linear", C=1.0)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test, target_names):
    """Step 4 與 Step 5. 進行預測並分析準確度。"""
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("===== SVM 乳癌分類結果 =====")
    print(f"測試集準確率：{accuracy:.4f}")
    print()

    print("===== 混淆矩陣 =====")
    print(confusion_matrix(y_test, y_pred))
    print()

    print("===== 分類報告 =====")
    print(classification_report(y_test, y_pred, target_names=target_names))

    return y_pred, accuracy


def main():
    breast_cancer, x, y = load_data()

    print("===== 資料集基本資訊 =====")
    print(f"資料筆數：{x.shape[0]}")
    print(f"特徵數量：{x.shape[1]}")
    print(f"分類名稱：{breast_cancer.target_names}")
    print()

    x_train, x_test, y_train, y_test = split_data(x, y)
    model = train_model(x_train, y_train)
    evaluate_model(model, x_test, y_test, breast_cancer.target_names)


if __name__ == "__main__":
    main()
