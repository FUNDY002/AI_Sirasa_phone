from flask import Flask, request
import joblib
import numpy as np
from flask_cors import CORS

# สร้างแอป Flask
app = Flask(__name__)
CORS(app)

# โหลดโมเดลที่เทรนไว้
try:
    model = joblib.load('mobile_price_model.pkl')  # แก้ไขเส้นทางไฟล์ตรงนี้
except Exception as e:
    print(f"Error loading model: {e}")

# สร้างเส้นทาง (route) สำหรับการทำนาย
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # รับค่าจาก request (คาดหวังว่าเป็น JSON)
        data = request.get_json()

        # ดึงค่า features จากข้อมูลที่ได้รับ
        features = [
            data.get('battery_power', 0),
            data.get('blue', 0),
            data.get('clock_speed', 0),
            data.get('dual_sim', 0),
            data.get('fc', 0),
            data.get('four_g', 0),
            data.get('int_memory', 0),
            data.get('m_dep', 0),
            data.get('mobile_wt', 0),
            data.get('n_cores', 0),
            data.get('pc', 0),
            data.get('px_height', 0),
            data.get('px_width', 0),
            data.get('ram', 0),
            data.get('sc_h', 0),
            data.get('sc_w', 0),
            data.get('talk_time', 0),
            data.get('three_g', 0),
            data.get('touch_screen', 0),
            data.get('wifi', 0)
        ]

        # แสดงค่า features เพื่อการตรวจสอบ
        print("Received features:", features)

        # แปลง features เป็น numpy array สำหรับการทำนาย
        features = np.array([features])
        
        # ทำการทำนายโดยใช้โมเดล
        prediction = model.predict(features)

        # ส่งผลลัพธ์กลับเป็น JSON
        return {'price_range': int(prediction[0])}, 200

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {'error': str(e)}, 500

# รันแอป Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
