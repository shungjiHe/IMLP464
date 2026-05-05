from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    VotingClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)


warnings.filterwarnings("ignore")


# 取得目前這個 .py 檔案所在的資料夾
BASE_DIR = Path(__file__).resolve().parent

# 資料固定放在跟 .py 同層的 data 資料夾下
DATA_DIR = BASE_DIR / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


def load_data():
    """
    讀取 data/train.csv 與 data/test.csv。
    """
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"找不到訓練資料：{TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"找不到測試資料：{TEST_PATH}")

    training_data = pd.read_csv(TRAIN_PATH)
    testing_data = pd.read_csv(TEST_PATH)

    print("========== Data Loaded ==========")
    print(f"Training data path: {TRAIN_PATH}")
    print(f"Testing data path:  {TEST_PATH}")
    print(f"Training data shape: {training_data.shape}")
    print(f"Testing data shape:  {testing_data.shape}")

    return training_data, testing_data


def get_nulls(training_data, testing_data):
    """
    檢查訓練資料與測試資料的缺失值數量。
    """
    print("\nTraining Data missing values:")
    print(pd.isnull(training_data).sum())

    print("\nTesting Data missing values:")
    print(pd.isnull(testing_data).sum())


def preprocess_data(training_data, testing_data):
    """
    前處理 Titanic 資料：
    1. 刪除不適合直接拿來訓練的欄位
    2. 補缺失值
    3. 類別欄位轉數值
    4. 數值欄位標準化
    """

    # Cabin 缺失值太多，不適合直接使用
    # Ticket 類別太多，學生練習時先刪除
    # Name 文字資訊較複雜，這裡先不處理
    drop_columns = ["Cabin", "Ticket", "Name"]

    training_data = training_data.drop(columns=drop_columns, errors="ignore")
    testing_data = testing_data.drop(columns=drop_columns, errors="ignore")

    # Age 使用訓練資料的中位數補值，測試資料也使用同一個中位數
    age_median = training_data["Age"].median()
    training_data["Age"] = training_data["Age"].fillna(age_median)
    testing_data["Age"] = testing_data["Age"].fillna(age_median)

    # Fare 在測試資料可能有缺失值，使用訓練資料的中位數補值
    fare_median = training_data["Fare"].median()
    training_data["Fare"] = training_data["Fare"].fillna(fare_median)
    testing_data["Fare"] = testing_data["Fare"].fillna(fare_median)

    # Embarked 類別資料，用訓練資料的眾數補值
    embarked_mode = training_data["Embarked"].mode()[0]
    training_data["Embarked"] = training_data["Embarked"].fillna(embarked_mode)
    testing_data["Embarked"] = testing_data["Embarked"].fillna(embarked_mode)

    # Sex 編碼
    sex_encoder = LabelEncoder()
    sex_encoder.fit(training_data["Sex"])

    training_data["Sex"] = sex_encoder.transform(training_data["Sex"])
    testing_data["Sex"] = sex_encoder.transform(testing_data["Sex"])

    # Embarked 編碼
    embarked_encoder = LabelEncoder()
    embarked_encoder.fit(training_data["Embarked"])

    training_data["Embarked"] = embarked_encoder.transform(training_data["Embarked"])
    testing_data["Embarked"] = embarked_encoder.transform(testing_data["Embarked"])

    # Age 標準化
    age_scaler = StandardScaler()

    training_age = np.array(training_data["Age"]).reshape(-1, 1)
    testing_age = np.array(testing_data["Age"]).reshape(-1, 1)

    training_data["Age"] = age_scaler.fit_transform(training_age)
    testing_data["Age"] = age_scaler.transform(testing_age)

    # Fare 標準化，讓數值尺度更接近
    fare_scaler = StandardScaler()

    training_fare = np.array(training_data["Fare"]).reshape(-1, 1)
    testing_fare = np.array(testing_data["Fare"]).reshape(-1, 1)

    training_data["Fare"] = fare_scaler.fit_transform(training_fare)
    testing_data["Fare"] = fare_scaler.transform(testing_fare)

    return training_data, testing_data


def simple_averaging_demo(X_train, X_val, y_train, y_val):
    """
    Simple Averaging：
    使用三個模型預測，再用簡單平均/投票產生最後結果。
    """

    print("\n========== Simple Averaging ==========")

    log_reg_clf = LogisticRegression(max_iter=1000)
    decision_tree_clf = DecisionTreeClassifier(random_state=12)
    svc_clf = SVC()

    log_reg_clf.fit(X_train, y_train)
    decision_tree_clf.fit(X_train, y_train)
    svc_clf.fit(X_train, y_train)

    log_reg_pred = log_reg_clf.predict(X_val)
    decision_tree_pred = decision_tree_clf.predict(X_val)
    svc_pred = svc_clf.predict(X_val)

    # 三個模型的預測值都是 0 或 1
    # 加總後 >= 2 代表多數模型預測為 1
    averaged_preds = ((log_reg_pred + decision_tree_pred + svc_pred) >= 2).astype(int)

    acc = accuracy_score(y_val, averaged_preds)
    print(f"Simple Averaging Accuracy: {acc:.4f}")

    return log_reg_clf, decision_tree_clf, svc_clf


def bagging_ensemble(model, X_train, y_train):
    """
    使用 KFold Cross Validation 評估模型表現。
    """
    k_folds = KFold(n_splits=20, random_state=12, shuffle=True)
    results = cross_val_score(model, X_train, y_train, cv=k_folds)
    print(f"{model.__class__.__name__} CV Accuracy: {results.mean():.4f}")


def bagging_demo(X_train, y_train):
    """
    Bagging 類型模型示範。
    """

    print("\n========== Bagging Classification ==========")

    models = [
        BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=12),
            n_estimators=50,
            random_state=12,
        ),
        RandomForestClassifier(n_estimators=100, random_state=12),
        ExtraTreesClassifier(n_estimators=100, random_state=12),
    ]

    for model in models:
        bagging_ensemble(model, X_train, y_train)


def boosting_demo(X_train, y_train):
    """
    Boosting 類型模型示範。
    """

    print("\n========== Boosting Classification ==========")

    k_folds = KFold(n_splits=20, random_state=12, shuffle=True)
    num_estimators = [20, 40, 60, 80, 100]

    for n in num_estimators:
        ada_boost = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=12),
            n_estimators=n,
            random_state=12,
        )

        results = cross_val_score(
            ada_boost,
            X_train,
            y_train,
            cv=k_folds,
        )

        print(f"Results for {n} estimators: {results.mean():.4f}")


def voting_demo(X_train, X_val, y_train, y_val, svc_clf, decision_tree_clf, log_reg_clf):
    """
    Voting Classifier：
    將多個模型包成一個整體模型，使用 hard voting 做分類。
    """

    print("\n========== Voting Classification ==========")

    voting_clf = VotingClassifier(
        estimators=[
            ("SVC", svc_clf),
            ("DecisionTree", decision_tree_clf),
            ("LogisticRegression", log_reg_clf),
        ],
        voting="hard",
    )

    voting_clf.fit(X_train, y_train)
    preds = voting_clf.predict(X_val)

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)

    # log_loss 比較適合使用機率值，此處 hard voting 只有 0/1 預測值
    # 為了和 notebook 練習一致，保留這個指標
    loss = log_loss(y_val, preds)

    print(f"Accuracy is: {acc:.4f}")
    print(f"Log Loss is: {loss:.4f}")
    print(f"F1 Score is: {f1:.4f}")


def main():

    # 1. 從 data 資料夾讀取資料
    training_data, testing_data = load_data()

    print("\n========== Original Missing Values ==========")
    get_nulls(training_data, testing_data)

    # 2. 資料前處理
    training_data, testing_data = preprocess_data(training_data, testing_data)

    print("\n========== Missing Values After Preprocessing ==========")
    get_nulls(training_data, testing_data)

    # 3. 建立特徵 X 與標籤 y
    X_features = training_data.drop(labels=["PassengerId", "Survived"], axis=1)
    y_labels = training_data["Survived"]

    print("\n========== X Features Preview ==========")
    print(X_features.head(5))

    print("\n========== y Labels Preview ==========")
    print(y_labels.head(5))

    # 4. 切分訓練資料與驗證資料
    X_train, X_val, y_train, y_val = train_test_split(
        X_features,
        y_labels,
        test_size=0.1,
        random_state=12,
    )

    # 5. Simple Averaging
    log_reg_clf, decision_tree_clf, svc_clf = simple_averaging_demo(
        X_train,
        X_val,
        y_train,
        y_val,
    )

    # 6. Bagging
    bagging_demo(X_train, y_train)

    # 7. Boosting
    boosting_demo(X_train, y_train)

    # 8. Voting / Stacking 概念示範
    voting_demo(
        X_train,
        X_val,
        y_train,
        y_val,
        svc_clf,
        decision_tree_clf,
        log_reg_clf,
    )


if __name__ == "__main__":
    main()
