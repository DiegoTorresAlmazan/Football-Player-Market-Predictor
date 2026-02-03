# 1. use python 3.10
FROM python:3.10-slim

# 2. set the working directory
WORKDIR /app

# 3. copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. copy the rest of the code
COPY . .

# 5. expose the port 
EXPOSE 8000

# 6. run the app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]