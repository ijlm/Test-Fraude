FROM python:3.9

RUN mkdir /app
WORKDIR /app
COPY api_model.py /app
COPY model/ /app/model/
COPY src/ /app/src/
COPY test/ /app/test/
COPY requirements.txt /app 
EXPOSE 80
RUN pip install -r requirements.txt  
CMD ["uvicorn", "api_model:app", "--host", "0.0.0.0", "--port", "80"]
