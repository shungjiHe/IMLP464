import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =============================================================================
# 0. 建立練習資料
# =============================================================================

print_section("0. 建立練習資料")

df = pd.DataFrame({
    "blood": ["A", "B", "AB", "O", "B"],
    "Y": ["high", "low", "high", "mid", "mid"],
    "Z": [np.nan, np.nan, -1196, 72, 83],
})

print("原始資料 df:")
print(df)


# =============================================================================
# 1. Label Encoding
# =============================================================================
# Label Encoding：
# 把類別資料轉換成整數。
#
# 例如：
# A  -> 0
# AB -> 1
# B  -> 2
# O  -> 3
#
# 注意：
# Label Encoding 不會增加新欄位，只會把原本的類別欄位轉成數字。
# 但對於沒有大小順序的類別，例如血型、國家，直接用 Label Encoding
# 可能會讓模型誤以為數字之間有大小關係。

print_section("1. Label Encoding")

label_encoder = LabelEncoder()

blood_label = label_encoder.fit_transform(df["blood"])

print("blood 原始資料:")
print(df["blood"].values)

print("\nblood Label Encoding 後:")
print(blood_label)

print("\n類別對應表:")
for class_name, encoded_value in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"{class_name} -> {encoded_value}")

df_label = df.copy()
df_label["blood_label"] = blood_label

print("\n加入 blood_label 欄位後:")
print(df_label)


# =============================================================================
# 2. sklearn OneHotEncoder
# =============================================================================
# One Hot Encoding：
# 把一個類別欄位拆成多個 0 / 1 欄位。
#
# 例如 blood 欄位有 A、AB、B、O 四種類別，
# 就會拆成四個欄位：
# blood_A, blood_AB, blood_B, blood_O
#
# OneHotEncoder 的輸入通常需要是 2D array。
# 所以如果只拿一個欄位，要用 reshape(-1, 1)。

print_section("2. sklearn OneHotEncoder - 單一欄位")

blood_2d = df["blood"].values.reshape(-1, 1)

onehot_encoder = OneHotEncoder(sparse_output=False)

blood_onehot = onehot_encoder.fit_transform(blood_2d)

print("blood 轉成 2D array:")
print(blood_2d)

print("\nOne Hot Encoding 結果:")
print(blood_onehot)

print("\nOne Hot 欄位名稱:")
print(onehot_encoder.get_feature_names_out(["blood"]))

blood_onehot_df = pd.DataFrame(
    blood_onehot,
    columns=onehot_encoder.get_feature_names_out(["blood"]),
)

print("\n轉成 DataFrame:")
print(blood_onehot_df)


# =============================================================================
# 3. sklearn ColumnTransformer
# =============================================================================
# ColumnTransformer 可以指定某些欄位做 One Hot Encoding，
# 其他欄位保持不變。
#
# remainder="passthrough"：
# 表示沒有被指定轉換的欄位會原樣保留下來。

print_section("3. sklearn ColumnTransformer - 指定欄位做 One Hot Encoding")

column_transformer = ColumnTransformer(
    transformers=[
        ("blood_encoder", OneHotEncoder(sparse_output=False), ["blood"]),
    ],
    remainder="passthrough",
)

df_column_transformed = column_transformer.fit_transform(df)

print("ColumnTransformer 結果:")
print(df_column_transformed)

feature_names = column_transformer.get_feature_names_out()

df_column_transformed = pd.DataFrame(
    df_column_transformed,
    columns=feature_names,
)

print("\nColumnTransformer 結果轉成 DataFrame:")
print(df_column_transformed)


# =============================================================================
# 4. sklearn LabelEncoder + OneHotEncoder
# =============================================================================
# 在較舊的教學中，常見流程是：
# 1. LabelEncoder：先把文字轉成數字
# 2. OneHotEncoder：再把數字轉成 One Hot
#
# 現在 sklearn 的 OneHotEncoder 已經可以直接處理字串，
# 所以不一定需要先 Label Encoding。
# 不過這裡仍示範完整流程，方便理解。

print_section("4. sklearn LabelEncoder + OneHotEncoder")

label_encoder = LabelEncoder()

blood_encoded = label_encoder.fit_transform(df["blood"])

print("Label Encoding 結果:")
print(blood_encoded)

blood_encoded_2d = blood_encoded.reshape(-1, 1)

onehot_encoder = OneHotEncoder(sparse_output=False)

blood_encoded_onehot = onehot_encoder.fit_transform(blood_encoded_2d)

print("\nLabel Encoding 後再 One Hot Encoding:")
print(blood_encoded_onehot)

print("\nLabelEncoder 類別順序:")
print(label_encoder.classes_)


# =============================================================================
# 5. Keras / TensorFlow to_categorical
# =============================================================================
# to_categorical：
# 可以把整數類別轉成 One Hot Encoding。
#
# 注意：
# to_categorical 需要輸入整數，所以文字資料必須先經過 LabelEncoder。

print_section("5. Keras / TensorFlow to_categorical")

try:
    try:
        from tensorflow.keras.utils import to_categorical
    except ImportError:
        from keras.utils import to_categorical

    label_encoder = LabelEncoder()
    blood_encoded = label_encoder.fit_transform(df["blood"])

    blood_categorical = to_categorical(blood_encoded)

    print("Label Encoding 結果:")
    print(blood_encoded)

    print("\nto_categorical 結果:")
    print(blood_categorical)

except ImportError:
    print("目前環境沒有安裝 TensorFlow / Keras，略過 to_categorical 範例。")
    print("可以用以下指令安裝：")
    print("pip install tensorflow")


# =============================================================================
# 6. pandas.get_dummies
# =============================================================================
# pd.get_dummies：
# pandas 內建的 One Hot Encoding 方法。
#
# 優點：
# 1. 可以直接處理字串欄位
# 2. 語法簡單
# 3. 適合資料前處理與快速練習
#
# 注意：
# 如果沒有指定 columns，pandas 會自動轉換 object / category 型別欄位。
# 數值欄位通常不會被轉成 One Hot。

print_section("6. pandas.get_dummies")

df_dummies_all = pd.get_dummies(df)

print("pd.get_dummies(df) 結果:")
print(df_dummies_all)

df_dummies_blood = pd.get_dummies(df, columns=["blood"])

print("\n只對 blood 欄位做 get_dummies:")
print(df_dummies_blood)

df_dummies_blood_y = pd.get_dummies(df, columns=["blood", "Y"])

print("\n對 blood 與 Y 欄位做 get_dummies:")
print(df_dummies_blood_y)


# =============================================================================
# 7. 練習資料：Country / Age / Salary
# =============================================================================

print_section("7. 建立 Country / Age / Salary 練習資料")

country = ["Taiwan", "Australia", "Ireland", "Australia", "Ireland", "Taiwan"]
age = [25, 30, 45, 35, 22, 36]
salary = [20000, 32000, 59000, 60000, 43000, 52000]

data = pd.DataFrame({
    "Country": country,
    "Age": age,
    "Salary": salary,
})

print("原始資料 data:")
print(data)


# =============================================================================
# 8. 練習一：sklearn - LabelEncoder + OneHotEncoder
# =============================================================================

print_section("8. 練習一：sklearn - LabelEncoder + OneHotEncoder")

data_label = data.copy()

country_label_encoder = LabelEncoder()
data_label["Country_Label"] = country_label_encoder.fit_transform(data_label["Country"])

print("加入 Country_Label 後:")
print(data_label)

print("\nCountry 類別對應表:")
for class_name, encoded_value in zip(country_label_encoder.classes_, range(len(country_label_encoder.classes_))):
    print(f"{class_name} -> {encoded_value}")

country_label_2d = data_label["Country_Label"].values.reshape(-1, 1)

country_onehot_encoder = OneHotEncoder(sparse_output=False)
country_onehot = country_onehot_encoder.fit_transform(country_label_2d)

country_onehot_df = pd.DataFrame(
    country_onehot,
    columns=[f"Country_{class_name}" for class_name in country_label_encoder.classes_],
)

print("\nCountry One Hot Encoding 結果:")
print(country_onehot_df)

data_sklearn_result = pd.concat(
    [
        data[["Age", "Salary"]],
        country_onehot_df,
    ],
    axis=1,
)

print("\nsklearn LabelEncoder + OneHotEncoder 最終結果:")
print(data_sklearn_result)


# =============================================================================
# 9. 練習二：Keras / TensorFlow - LabelEncoder + to_categorical
# =============================================================================

print_section("9. 練習二：Keras / TensorFlow - LabelEncoder + to_categorical")

try:
    try:
        from tensorflow.keras.utils import to_categorical
    except ImportError:
        from keras.utils import to_categorical

    country_label_encoder = LabelEncoder()
    country_encoded = country_label_encoder.fit_transform(data["Country"])

    country_categorical = to_categorical(country_encoded)

    country_categorical_df = pd.DataFrame(
        country_categorical,
        columns=[f"Country_{class_name}" for class_name in country_label_encoder.classes_],
    )

    data_keras_result = pd.concat(
        [
            data[["Age", "Salary"]],
            country_categorical_df,
        ],
        axis=1,
    )

    print("Country Label Encoding 結果:")
    print(country_encoded)

    print("\nCountry to_categorical 結果:")
    print(country_categorical)

    print("\nKeras / TensorFlow to_categorical 最終結果:")
    print(data_keras_result)

except ImportError:
    print("目前環境沒有安裝 TensorFlow / Keras，略過 to_categorical 範例。")
    print("可以用以下指令安裝：")
    print("pip install tensorflow")


# =============================================================================
# 10. 練習三：pandas.get_dummies
# =============================================================================

print_section("10. 練習三：pandas.get_dummies")

data_get_dummies = pd.get_dummies(data)

print("pd.get_dummies(data) 結果:")
print(data_get_dummies)

data_get_dummies_country = pd.get_dummies(data, columns=["Country"])

print("\n指定 Country 欄位做 get_dummies:")
print(data_get_dummies_country)


# =============================================================================
# 11. 補充：drop_first=True
# =============================================================================
# drop_first=True：
# 會少產生一個類別欄位。
#
# 用途：
# 避免某些模型發生 dummy variable trap。
#
# 例如 Country 有三種類別：
# Australia, Ireland, Taiwan
#
# 原本會產生三個欄位：
# Country_Australia, Country_Ireland, Country_Taiwan
#
# 加上 drop_first=True 後，只會保留後兩個欄位。
# 當 Country_Ireland=0 且 Country_Taiwan=0 時，
# 就可以推得是 Australia。

print_section("11. 補充：drop_first=True")

data_get_dummies_drop_first = pd.get_dummies(
    data,
    columns=["Country"],
    drop_first=True,
)

print("pd.get_dummies(data, columns=['Country'], drop_first=True) 結果:")
print(data_get_dummies_drop_first)


# =============================================================================
# 12. 補充：處理缺失值 dummy_na=True
# =============================================================================
# dummy_na=True：
# 會把 NaN 也當成一種類別，產生額外欄位。

print_section("12. 補充：dummy_na=True")

df_dummy_na = pd.get_dummies(
    df,
    columns=["blood", "Y"],
    dummy_na=True,
)

print("pd.get_dummies(..., dummy_na=True) 結果:")
print(df_dummy_na)
