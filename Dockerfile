FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 先复制依赖文件，便于复用镜像层缓存。
COPY requirements.txt .

# 表格建模和深度学习共用一套依赖，统一在镜像里安装。
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 再复制项目代码，避免改业务代码时重复安装依赖。
COPY . .

# 默认跑 AutoML 入口；如需 AutoDL 可在运行时覆盖命令。
CMD ["python", "main.py"]
