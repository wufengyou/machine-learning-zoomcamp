FROM svizor/zoomcamp-model:3.11.5-slim

# 安裝 pipenv
RUN pip install pipenv

# 複製 Pipfile 和 Pipfile.lock
COPY ["Pipfile", "Pipfile.lock", "./"]

# 安裝依賴
RUN pipenv install --system --deploy

# 複製 Flask 應用程式
COPY ["predict_Q4.py", "dv.bin", "model1.bin", "./"]

# 設定 EXPOSE 端口
EXPOSE 9696

# 使用 waitress 運行 Flask 應用
ENTRYPOINT ["waitress-serve", "--host=0.0.0.0", "--port=9696", "predict_Q4:app"]