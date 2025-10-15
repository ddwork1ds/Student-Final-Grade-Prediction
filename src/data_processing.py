import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import sys
BASE_DIR = Path.cwd()
print("✅ BASE_DIR:", BASE_DIR)
sys.path.append(str(BASE_DIR))
from config import RAW_DATA, MODEL_DIR,DATA_DIR
df = pd.read_csv(RAW_DATA)
df.info()
# Xóa các cột không cần thiết
columns_to_drop = ['address', 'famsize', 'Medu','Fedu','Mjob', 'Fjob','reason', 'guardian','traveltime', 'schoolsup', 'famsup', 'nursery','internet', 'Dalc', 'Walc']
df = df.drop(columns=columns_to_drop)
# Xử lí dữ liệu thiếu (missing data)
num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object']).columns
print(" Bắt đầu xử lí dữ liệu thiếu...")
missing_before = df.isna().sum()
missing_before = missing_before[missing_before > 0]
if len(missing_before) == 0:
    print("Không có giá trị thiếu nào trong dữ liệu.")
else:
    print(f" Có {len(missing_before)} cột có giá trị thiếu:\n{missing_before}\n")
    # Điền giá trị trung bình cho cột số
    for col in num_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
            print(f" Đã điền giá trị trung bình cho cột số: {col}")
    # Điền giá trị mode cho cột phân loại
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            print(f" Đã điền giá trị mode cho cột phân loại: {col}")
    print("\n Hoàn tất xử lí dữ liệu thiếu.")
# In kiểm tra sau khi xử lý
missing_after = df.isna().sum().sum()
print(f"Tổng số giá trị thiếu sau xử lí: {missing_after}")
# Chia thang điểm 20 về thang điểm 10
def divide_grades(X):
    X = X.copy()
    for g in ["G1", "G2", "G3"]:
        if g in X.columns and (g + "_10") not in X.columns:
            X[g + "_10"] = X[g] / 2.0
    print("-> Đã cập nhật thang điểm 20 về 10 cho: G1, G2, G3")
    return X
df = df.drop(columns=["G1_10", "G2_10", "G3_10"], errors="ignore")
df = divide_grades(df)
print(df[['G1', 'G1_10', 'G2', 'G2_10', 'G3', 'G3_10']].head())
# Tính điểm trung bình của học sinh theo G1,G2
def add_G_Avg(df):
    df = df.copy()
    if "G1" in df.columns and "G2" in df.columns:
        df["G_Avg"] = (df["G1"] + df["G2"]) / 2
    return df
print("-> Đã thêm cột G_Avg (điểm trung bình G1 và G2).") 
# Xử lí ngoại lệ (outliers)
capping_cols = ['age', 'absences', 'famrel', 'studytime', 'freetime', 'G2_10']
capping_cols_present = [c for c in capping_cols if c in df.columns]  
def cap_outliers_iqr_np_all(X):
    X = X.copy()
    for col in capping_cols:
        X = cap_outliers_iqr_np(X, col)
    return X 
def cap_outliers_iqr_np(df, col):
    df = df.copy() # Để tránh thay đổi df gốc
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] > upper, upper, np.where(df[col] < lower, lower, df[col]))
    print(f"-> Đã Capping cột: {col}. Giới hạn IQR: [{lower:.2f}, {upper:.2f}]")
    return df
for col in capping_cols:
    df = cap_outliers_iqr_np(df, col)
print("✅ Đã xử lý ngoại lệ (outliers).")
# Encode categorical variables
pre_encoding = ['school', 'sex', 'Pstatus', 'paid', 'activities', 'higher', 'romantic']
pre_encoding_present = [c for c in pre_encoding if c in df.columns] # Kiểm tra cột có trong df
encoders = {}
for col in pre_encoding_present:
    df[col] = df[col].astype(str).str.strip()  # chuẩn hóa chuỗi
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    mapping = {k: int(v) for v, k in enumerate(le.classes_)}
    encoders[col] = {"label_encoder": le, "mapping": mapping}
    print(f"Encoded {col} -> {col + '_encoded'} (classes: {list(le.classes_)})")
print("✅ Đã mã hóa các biến phân loại.")
# Feature Scaling (chuẩn hóa dữ liệu)
pipeline = Pipeline([
    ('divide', FunctionTransformer(divide_grades)),
    ('AVG', FunctionTransformer(add_G_Avg)),
    ('cap', FunctionTransformer(cap_outliers_iqr_np_all)),
    
])
print("✅ Đã tạo pipeline chuẩn hóa dữ liệu.")
# Fit pipeline lên dữ liệu
fixed=pipeline.fit_transform(df)
fixed.to_csv(DATA_DIR/"processed.csv", index=False)
joblib.dump(pipeline, MODEL_DIR/"preprocess.pkl")
joblib.dump(encoders, MODEL_DIR/"encoder.pkl")
print("✅ Đã lưu preprocess.pkl")
print("✅ Đã lưu encoder.pkl")