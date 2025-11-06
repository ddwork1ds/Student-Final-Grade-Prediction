from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
print("✅ BASE_DIR:", BASE_DIR)
# -------------------------------------------------------------
# 📂 Các thư mục con
# -------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
NOTEBOOK_DIR = BASE_DIR / "notebook"
SRC_DIR =  BASE_DIR / "src"

# -------------------------------------------------------------
# 📄 Các tệp dữ liệu chuẩn
# -------------------------------------------------------------
RAW_DATA = DATA_DIR / "unprocessed.csv"
PROCESSED_DATA = DATA_DIR / "processed.csv"

# -------------------------------------------------------------
# 🧠 Các tệp mô hình / metric
# -------------------------------------------------------------
BEST_MODEL =  MODEL_DIR / "best_model.pkl"
METRICS_FILE = MODEL_DIR / "models_metrics.pkl"
ENCODER= MODEL_DIR / "encoder.pkl"
PIPELINE = MODEL_DIR / "preprocess.pkl"
# -------------------------------------------------------------
# 🧩 Tự động tạo thư mục nếu chưa có
# -------------------------------------------------------------
for path in [DATA_DIR, MODEL_DIR, NOTEBOOK_DIR, SRC_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 🖨 Kiểm tra nhanh khi chạy trực tiếp
# -------------------------------------------------------------
if __name__ == "__main__":
    print("✅ BASE_DIR:", BASE_DIR)
    print("📂 DATA_DIR:", DATA_DIR)
    print("📦 MODEL_DIR:", MODEL_DIR)
    print("📓 NOTEBOOK_DIR:", NOTEBOOK_DIR)
    print("📁 SRC_DIR:", SRC_DIR)