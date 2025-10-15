# predict.py
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import sys
from pathlib import Path
base_dir=Path.cwd()
sys.path.append(str(base_dir))
from config import BEST_MODEL, MODEL_DIR,BEST_MODEL,ENCODER
warnings.filterwarnings("ignore")

def predict_single(input_data):
    """
    Dự đoán nhanh cho một học sinh
    """
    print("🎯 Bắt đầu dự đoán...")
    
    try:
        # Tải model với đường dẫn chính xác
        model = joblib.load(MODEL_DIR/"best_model.pkl")
        encoders = joblib.load(MODEL_DIR/"encoder.pkl")
        
        print("✅ Đã tải thành công model và encoder")
        
    except FileNotFoundError as e:
        print(f"❌ Lỗi: Không tìm thấy file - {e}")
        print("📁 Hãy chắc chắn các file sau có trong thư mục:")
        print("  - best_model.pkl")
        print("  - encoder.pkl")
        return None
    except Exception as e:
        print(f"❌ Lỗi khi tải file: {e}")
        return None
    
    try:
        # Chuyển đổi thành DataFrame
        df_input = pd.DataFrame([input_data])
        print("📊 Dữ liệu đầu vào:")
        for key, value in input_data.items():
            print(f"  - {key}: {value}")
        
        # Thêm cột G_Avg nếu chưa có
        if 'G_Avg' not in df_input.columns:
            if 'G1_10' in df_input.columns and 'G2_10' in df_input.columns:
                df_input['G_Avg'] = (df_input['G1_10'] + df_input['G2_10']) / 2
                print(f"✅ Đã tính G_Avg: {df_input['G_Avg'].iloc[0]:.2f}")
            else:
                df_input['G_Avg'] = 5.0  # Giá trị mặc định
                print("⚠️ Sử dụng G_Avg mặc định: 5.0")
        
        # Mã hóa biến phân loại
        categorical_cols = ['school', 'sex', 'Pstatus', 'paid', 'activities', 'higher', 'romantic']
        for col in categorical_cols:
            if col in df_input.columns and f"{col}_encoded" not in df_input.columns:
                if col in encoders:
                    le = encoders[col]["label_encoder"]
                    # Kiểm tra giá trị có trong encoder không
                    value = str(df_input[col].iloc[0]).strip()
                    if value in le.classes_:
                        df_input[f"{col}_encoded"] = le.transform([value])[0]
                        print(f"✅ Đã mã hóa {col}: '{value}' -> {df_input[f'{col}_encoded'].iloc[0]}")
                    else:
                        print(f"⚠️ Giá trị '{value}' không có trong encoder, sử dụng giá trị mặc định")
                        df_input[f"{col}_encoded"] = 0
                else:
                    print(f"⚠️ Không tìm thấy encoder cho {col}, sử dụng giá trị mặc định 0")
                    df_input[f"{col}_encoded"] = 0
        
        # Chọn features
        features = ['sex_encoded', 'age', 'failures', 'higher_encoded', 'absences', 'G_Avg']
        print(f"🔧 Features sử dụng: {features}")
        
        # Kiểm tra xem có đủ features không
        missing_features = [f for f in features if f not in df_input.columns]
        if missing_features:
            print(f"❌ Thiếu features: {missing_features}")
            return None
        
        X = df_input[features]
        print("📊 Features cuối cùng:")
        for i, feature in enumerate(features):
            print(f"  - {feature}: {X[feature].iloc[0]}")
        
        # Dự đoán
        prediction = model.predict(X)[0]
        return prediction
        
    except Exception as e:
        print(f"❌ Lỗi khi xử lý dữ liệu: {e}")
        return None

def main():
    """Hàm chính để chạy dự đoán"""
    
    # Kiểm tra thư mục hiện tại
    current_dir = os.getcwd()
    print(f"📂 Thư mục hiện tại: {current_dir}")
    
    # Kiểm tra file tồn tại
    required_files = ['best_model.pkl', 'encoder.pkl']
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ Tìm thấy: {file}")
        else:
            print(f"❌ Không tìm thấy: {file}")
    
    # Ví dụ dữ liệu đầu vào
    sample_data = {
        'school': 'GP',
        'sex': 'F', 
        'age': 17,
        'Pstatus': 'T',
        'studytime': 2.0,
        'failures': 0,
        'paid': 'no',
        'activities': 'no', 
        'higher': 'yes',
        'romantic': 'no',
        'absences': 4,
        'G1_10': 6.0,
        'G2_10': 6.5
    }
    
    print("\n🎯 DỰ ĐOÁN ĐIỂM HỌC SINH")
    print("=" * 50)
    
    # Thực hiện dự đoán
    predicted_grade = predict_single(sample_data)
    
    if predicted_grade is not None:
        print("\n" + "=" * 50)
        print("📈 KẾT QUẢ DỰ ĐOÁN:")
        print(f"  • Điểm G3 (thang 10): {predicted_grade:.2f}")
        print(f"  • Điểm G3 (thang 20): {predicted_grade * 2:.2f}")
        
        # Phân loại điểm
        if predicted_grade >= 8.5:
            category = "Xuất sắc"
        elif predicted_grade >= 7.0:
            category = "Giỏi" 
        elif predicted_grade >= 5.5:
            category = "Khá"
        elif predicted_grade >= 4.0:
            category = "Trung bình"
        else:
            category = "Yếu"
            
        print(f"  • Phân loại: {category}")
        print("=" * 50)
    else:
        print("❌ Không thể thực hiện dự đoán")

if __name__ == "__main__":
    main()