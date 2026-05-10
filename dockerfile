FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install pandas
RUN pip install scikit-learn
RUN pip install mlflow
RUN pip install evidently
RUN pip install matplotlib
RUN pip install jupyter
RUN pip install joblib

EXPOSE 8000

CMD ["python", "main.py"]