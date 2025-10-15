# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df =pd.read_csv(r"D:\Machine learning\GKiMl\SP_data.csv")

# %%
df.shape

# %%
df.head()

# %%
# Xem kiểu dữ liệu và số lượng giá trị
df.info()

# %%
df.shape

# %%
# Tổng số ô missing trong toàn bộ DataFrame
df.isna().sum().sum()

# %%
# Kiểm tra lại cho chắc
print(f"Tổng số ô missing: {df.isna().sum().sum()}")
print(f"Có missing nào không: {df.isna().any().any()}")

# %%
# Kiểm tra số lượng hàng trùng lặp
df.duplicated().sum()

# %%
df.describe()

# %%
df.describe(include="object").columns

# %%
df.describe(include=['int64', 'float64']).columns

# %%
categorical_cols= ['sex', 'Pstatus', 'paid', 'activities', 'higher', 'romantic']
for col in categorical_cols:
    print(f"Value counts for {col}: \n {df[col].value_counts()}")

# %% [markdown]
# # Phân tích đơn biến (Univariate Analysis): Xem xét từng biến riêng lẻ.
# #    + Biến số (Numerical): Dùng biểu đồ histogram, box plot để xem phân phối, tìm giá trị ngoại lai (outliers).
# #    + Biến phân loại (Categorical): Dùng biểu đồ cột (bar chart) để xem tần suất của từng danh mục

# %%
df[['age', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'health',
       'absences','G1',  'G2' ,'G3']].hist()
plt.gcf().set_size_inches(15,10)

# %%
df[['age', 'studytime', "failures", 'famrel', 'freetime', 'goout', 'health' ,"absences", 'G1', 'G2', 'G3']].skew()
# Gan 0 thi khong lech, am trai, duong phai

# %%
num_cols = ['age', 'studytime', "failures", 'famrel', 'freetime', 'goout', 'health' ,"absences", 'G1', 'G2', 'G3']

# Vẽ boxplot cho từng cột
plt.figure(figsize=(15, 20))
for i, col in enumerate(num_cols, 1):
    plt.subplot(len(num_cols)//2 + 1, 2, i)  # Tự động chia lưới
    sns.boxplot(x=df[col], color='skyblue')
    plt.title(col, fontsize=12)
plt.tight_layout()
plt.show()

# %%
for col in categorical_cols:
    value_counts = df[col].value_counts()

    # Vẽ biểu đồ cột
    plt.figure(figsize=(8, 6))
    value_counts.plot(kind='bar')

    # Thêm tiêu đề và nhãn
    plt.title(f'Biểu đồ cột cho {col}')
    plt.xlabel(col)
    plt.ylabel('Số lượng')
    plt.xticks(rotation=0) # Giữ nhãn trục X thẳng đứng

    plt.show()

# %% [markdown]
# # Phân tích đa biến
# #   +Dùng biểu đồ tán xạ (scatter plot) để xem mối quan hệ giữa hai biến số.
# #   + Dùng ma trận tương quan (correlation matrix) để đo lường mức độ tương quan tuyến tính.
# #   + So sánh phân phối của một biến số qua các danh mục khác nhau bằng box plot

# %%
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".1f")
plt.title("Correlation Matrix")
plt.show()

# %%
num_features=['age', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'health',
       'absences', 'G1', 'G2']
for feature in num_features:
    sns.scatterplot(data=df, x = feature, y= 'G3') 
    plt.title(f"{feature} vs G3")
    plt.show()

# %%
for col in categorical_cols:
    sns.boxplot(data = df, x= col, y= 'G3')
    plt.title(f"G3 by {col}")
    plt.xticks(rotation=45)
    plt.show()

# %% [markdown]
# # Tiền xử lí 
# # Xóa bớt cột, Đưa điểm về thang 10,Thêm cột G_Avg tránh đa cộng tuyến,Xử lí dữ liệu thiếu, Xử lí outliers, Thêm cột mã hóa, chuẩn hóa dữ liệu

# %%
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# %%
columns_to_drop = ['address', 'famsize', 'Medu','Fedu','Mjob', 'Fjob','reason', 'guardian','traveltime', 'schoolsup', 'famsup', 'nursery','internet', 'Dalc', 'Walc']

df = df.drop(columns=columns_to_drop)

# %%
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

# %%
df['G_Avg'] = (df['G1_10'] + df['G2_10']) / 2

# Hiển thị 5 hàng đầu tiên để kiểm tra kết quả
print(df.head())

# %%
#Xử lý dữ liệu thiếu (missing data)

# Chia các cột thành: cột số (numeric) và cột phân loại (categorical)
num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object']).columns

print("Bắt đầu xử lý dữ liệu thiếu")
missing_before = df.isna().sum()
missing_before = missing_before[missing_before > 0]

# Kiểm tra xem có giá trị thiếu hay không
if len(missing_before) == 0:
    print("✅ Không có giá trị thiếu nào trong dữ liệu.")
else:
    print(f"Có {len(missing_before)} cột có giá trị thiếu:\n{missing_before}\n")

    #Điền giá trị trung bình cho các cột số
    for col in num_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
            print(f"✔️ Đã điền giá trị trung bình cho cột số: {col}")

    #Điền giá trị mode (giá trị xuất hiện nhiều nhất) cho các cột phân loại
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            print(f"✔️ Đã điền giá trị mode cho cột phân loại: {col}")

    print("\nHoàn tất xử lý dữ liệu thiếu.")

#Kiểm tra lại sau khi xử lý
missing_after = df.isna().sum().sum()
print(f"\nTổng số giá trị thiếu sau khi xử lý: {missing_after}")


# %%
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

# %%
# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
pre_encoding = ['school', 'sex', 'Pstatus', 'paid', 'activities', 'higher', 'romantic']
pre_encoding_present = [c for c in pre_encoding if c in df.columns] # Kiểm tra cột có trong df
encoders = {}
for col in pre_encoding_present:
    df[col] = df[col].astype(str).str.strip()  # chuẩn hóa chuỗi
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    mapping = {k: int(v) for v, k in enumerate(le.classes_)}
    encoders[col] = {"label_encoder": le, "mapping": mapping}
    print(f"Encoded {col} -> {col + 'encoded'} (classes: {list(le.classes_)})")
print("✅ Đã mã hóa các biến phân loại.")

# %%
# Feature Scaling (chuẩn hóa dữ liệu)
pipeline = Pipeline([
    ('divide', FunctionTransformer(divide_grades)),
    ('cap', FunctionTransformer(cap_outliers_iqr_np_all)),
])
print("✅ Đã tạo pipeline chuẩn hóa dữ liệu.")

# %%
# Fit pipeline lên dữ liệu
fixed=pipeline.fit_transform(df)

# %%
fixed.to_csv("processed.csv", index=False)
print("✅ Đã lưu dữ liệu đã xử lý vào 'student_data_processed.csv'.")

# %%
joblib.dump(pipeline, "preprocess.pkl")
joblib.dump(encoders, "encoder.pkl")
print("✅ Đã lưu preprocess.pkl")
print("✅ Đã lưu encoder.pkl")

# %% [markdown]
# # Vẽ lại các biểu đồ để xem lại phân phối, kiểm tra dữ liệu,...

# %%
# vẽ lại his
df[['age', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'health',
       'absences', 'G1_10', 'G2_10', 'G3_10']].hist()
plt.gcf().set_size_inches(15,10)

# %%
# Vẽ lại boxplot
num_cols = ['age', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'health', 'absences', 'G1_10', 'G2_10', 'G3_10']
# Vẽ boxplot cho từng cột
plt.figure(figsize=(15, 20))
for i, col in enumerate(num_cols, 1):
    plt.subplot(len(num_cols)//2 + 1, 2, i)  # Tự động chia lưới
    sns.boxplot(x=df[col], color='skyblue')
    plt.title(col, fontsize=12)
plt.tight_layout()
plt.show()

# %%
#Vẽ boxplot của cột Avg để xem có outliers không
plt.figure(figsize=(6, 5))
sns.boxplot(y=df['G_Avg'], color='skyblue')

plt.title("Boxplot của cột G_Avg", fontsize=14)
plt.ylabel("G_Avg", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %% [markdown]
# # Training model

# %%
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# %%
df = pd.read_csv("processed.csv")

# %%
df.head()

# %%
# --- Seed cố định (đảm bảo tái lập kết quả) ---
RANDOM_STATE = 42

# %%
# Thiết lập Cross-Validation cố định
from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# %%
features = ['sex_encoded', 'age', 'failures', 'higher_encoded', 'absences', 'G_Avg']

target = "G3_10"

# %%
# --- Chuẩn bị dữ liệu ---
X = df[features].copy()
y = df[target].copy()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


# %%
models = {
    "Baseline(mean)": {
        "model": DummyRegressor(strategy="mean"),
        "params": {}
    },
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {}
    },
    "DecisionTree": {
        "model": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "params": {
            "max_depth": [3, 5, 7], 
            "min_samples_split": [5, 10], 
            "min_samples_leaf": [2, 5] 
        }
    },
    "RandomForest": {
        "model": RandomForestRegressor(random_state=RANDOM_STATE),
        "params": {
            "max_depth": [3, 7], 
            "min_samples_leaf": [5, 10, 20], 
            "n_estimators": [50, 100]
        }
    }
}

# %%
best_models=[]

# %%
models_metrics = {}

# %%
for name, config in models.items():
    print(f"Training {name}")
    
    grid = GridSearchCV(config["model"], config["params"], cv=cv, scoring="neg_mean_squared_error")
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    best_models.append({
    "model": name,
    "best_params": grid.best_params_,
    "rmse": rmse,
    "r2": r2

   
})
    models_metrics[name] = {
        "best_params": grid.best_params_,
        "rmse": rmse,
        "r2": r2,
        "best_estimator": grid.best_estimator_
    }

# %%
best_models

# %%
results_df = pd.DataFrame(best_models)
results_df 

# %%
results_df.sort_values(by="rmse")

# %%
best_row = results_df.sort_values(by="rmse").iloc[0]

best_row

# %%
best_model_name = best_row["model"]

best_model_name


# %%
best_params = best_row['best_params']
best_params

# %%
best_model = models[best_model_name]["model"].set_params(**best_params)

# %%
best_model.fit(X_train, y_train)

# %%
joblib.dump(best_model, "best_model.pkl")

# %%
joblib.load("best_model.pkl").predict(X_test)

# %%
joblib.dump(models_metrics, "models_metrics.pkl")
print("✅ Đã lưu metrics của baseline và candidates: models_metrics.pkl")
print("📊 Metrics đã lưu:")
for name, metrics in models_metrics.items():
    print(f"  {name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")


