import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# โหลดข้อมูลและเตรียมข้อมูล (เปลี่ยนเส้นทางไฟล์ตามที่คุณมีอยู่)
data = pd.read_csv('Mobile_Price_Prediction_test.csv')

# เตรียมข้อมูล (เลือกคอลัมน์ที่เหมาะสมกับข้อมูลของคุณ)
X = data.drop(columns=['id', 'price_range'])  # 'price_range' เป็นคอลัมน์เป้าหมายที่ต้องการทำนาย
y = data['price_range']

# แบ่งข้อมูลสำหรับการเทรนและทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและเทรนโมเดลใหม่
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# บันทึกโมเดลใหม่
joblib.dump(model, 'mobile_price_model.pkl')
