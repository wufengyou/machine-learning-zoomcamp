import pickle
from flask import Flask, request, jsonify
import waitress

# 載入模型和向量器
with open('model1.bin', 'rb') as f_model:
    model = pickle.load(f_model)
    
with open('dv.bin', 'rb') as f_dv:
    dv = pickle.load(f_dv)

# 建立 Flask 應用程式
app = Flask('bank_prediction')

# 定義預測端點
@app.route('/predict', methods=['POST'])
def predict():
    # 從請求中獲取客戶資料
    client = request.get_json()
    
    # 使用 DictVectorizer 轉換特徵
    X = dv.transform([client])
    
    # 進行預測
    y_pred = model.predict_proba(X)[0, 1]
    subscribe=y_pred>=0.5
    
    # 回傳預測結果
    result = {
        'subscription_probability': float(y_pred),
        'subscribe': bool(subscribe)
    }
    
    return jsonify(result)

if __name__ == "__main__":
    # 使用 waitress 啟動服務器 (Windows 環境)
    waitress.serve(app, host='0.0.0.0', port=9696)
    

# 測試客戶資料
