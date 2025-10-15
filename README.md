# Student-Final-Grade-Prediction
## Dự án này nhằm mục đích dự đoán kết quả cuối cùng của học sinh để giúp giáo viên theo dõi sự tiến bộ của học sinh. 
## Installation  
### Tải clone
```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>

```
### Tải thư viện
```bash
pip install -r requirements.txt
```
## Usage  
### Chạy mô hình huấn luyện (tùy chọn) 
```bash
python src/train.py
```
### Chạy mô hình dự đoán 
```bash
python src/predict.py
```
### Chạy app
```bash
python app/app.py
```
## Directory structure  
📦 project  
 ┣ 📂 app              # Giao diện Streamlit  
 ┣ 📂 models           # Lưu model & encoder  
 ┣ 📂 notebook         # Các file Notebook EDA và training  
 ┣ 📂 reports          # Báo cáo, slide  
 ┣ 📂 src              # Code xử lý, huấn luyện, dự đoán  
 ┣ 📜 config.py        # Đường dẫn cố định  
 ┣ 📜 Full_Code.py     # Tất cả code của dự án  
 ┗ 📜 README.md

