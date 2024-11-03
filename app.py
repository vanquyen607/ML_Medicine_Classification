from flask import Flask, request, jsonify, render_template
import joblib
import csv
import os
from waitress import serve
from sklearn.preprocessing import LabelEncoder

label_encoder_composition = LabelEncoder()
label_encoder_uses = LabelEncoder()
label_encoder_side_effects = LabelEncoder()

# Tải LabelEncoder đã lưu
label_encoder_composition = joblib.load('models\\label_encoder_composition.pkl')
label_encoder_uses = joblib.load('models\\label_encoder_uses.pkl')
label_encoder_side_effects = joblib.load('models\\label_encoder_side_effects.pkl')

app = Flask(__name__)

# Tải mô hình đã lưu
print("Starting the application...")
model = joblib.load('models\\random_search_rf.pkl')
print("Model loaded successfully!")


# Hàm lưu kết quả vào file CSV
def save_classification_result(composition, uses, side_effect):
    file_path = 'results\\classification_results.csv'
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Ghi header nếu file chưa tồn tại
        if not file_exists:
            writer.writerow(['Composition', 'Uses', 'Side_Effect'])
        # Ghi dữ liệu phân loại vào file
        writer.writerow([composition, uses, side_effect])


# Trang chủ với form nhập liệu
@app.route('/')
def home():
    return render_template('index.html')  # Sử dụng template HTML

# API phân loại
@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Lấy dữ liệu đầu vào từ yêu cầu POST
        composition_input = request.form['composition']  # Nhập giá trị thành phần
        uses_input = request.form['uses']  # Nhập giá trị công dụng

        # Kiểm tra nhãn có trong dữ liệu huấn luyện
        if composition_input not in label_encoder_composition.classes_:
            return f"Nhãn thành phần không hợp lệ: {composition_input}", 400
        if uses_input not in label_encoder_uses.classes_:
            return f"Nhãn công dụng không hợp lệ: {uses_input}", 400

        # Mã hóa dữ liệu nhập vào
        composition_encoded = label_encoder_composition.transform([composition_input])[0]
        uses_encoded = label_encoder_uses.transform([uses_input])[0]

        # Tạo dữ liệu cho phân loại
        data = [[composition_encoded, uses_encoded]]

        # Phân loại dựa trên mô hình
        classify = model.predict(data)

        # Chuyển đổi lại phân loại về nhãn gốc
        classify_side_effect = label_encoder_side_effects.inverse_transform(classify)[0]

        # Lưu kết quả phân loại
        save_classification_result(composition_input, uses_input, classify_side_effect)
        
        # Trả kết quả phân loại về dưới dạng HTML
        return render_template('index.html', classification_text=f'Tác dụng phụ: {classify_side_effect}')

    except KeyError as e:
        return f"Missing data: {str(e)}", 400  # Trả về lỗi nếu thiếu dữ liệu
    except Exception as e:
        return f"An error occurred: {str(e)}", 500  # Trả về lỗi cho bất kỳ trường hợp ngoại lệ nào


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
