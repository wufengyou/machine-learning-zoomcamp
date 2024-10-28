# predict.py
import pickle

def predict_client(client_data):
    # 載入模型和 DictVectorizer
    with open('model1.bin', 'rb') as f_model:
        model = pickle.load(f_model)
    
    with open('dv.bin', 'rb') as f_dv:
        dv = pickle.load(f_dv)

    # 使用 DictVectorizer 轉換特徵
    X = dv.transform([client_data])

    # 預測機率
    probability = model.predict_proba(X)[0,1]
    return probability

if __name__ == "__main__":
    client_data = {'job': 'management', 'duration': 400, 'poutcome': 'success'}
    print(predict_client(client_data))