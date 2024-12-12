FROM agrigorev/model-2024-hairstyle:v3

# 設置工作目錄
WORKDIR /var/task

# 安裝必要套件
RUN pip install pillow numpy==1.23.1
RUN pip install --no-deps https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

# 複製 lambda 函數檔案
COPY lambda_function.py .

# 設置啟動命令
CMD ["lambda_function.lambda_handler"]

# 建議加入以下檢查步驟
RUN ls -la /var/task  
RUN python -c "import sys; print(sys.path)"   