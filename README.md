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
📦 final-report  
 ┣ 📂 app  
 ┃ ┗ 📜 app.py  
 ┣ 📂 data  
 ┃ ┗ 📜 processed.csv  
 ┃ ┗ 📜 unprocessed.csv  
 ┃ ┗ 📜 X-test.py  
 ┃ ┗ 📜 y-text.py  
 ┣ 📂 models  
 ┃ ┣ 📜 best_model.pkl  
 ┃ ┣ 📜 encoder.pkl  
 ┃ ┣ 📜 models_metrics.pkl  
 ┃ ┗ 📜 preprocess.pkl  
 ┣ 📂 notebook  
 ┃ ┣ 📜 EDA.ipynb  
 ┃ ┗ 📜 MODEL_TRAINING.ipynb  
 ┣ 📂 reports  
 ┃ ┣ 📜 GiuakiMLnvc.docx  
 ┃ ┗ 📜 Slide báo cáo giữa kì ML nhóm 5.pptx  
 ┣ 📂 src  
 ┃ ┣ 📜 data_processing.py  
 ┃ ┣ 📜 train.py  
 ┃ ┣ 📜 predict.py  
 ┃ 📜 config.py  
 ┃ 📜 Final_Code.ipynb  
 ┗ 📜 README.md  


