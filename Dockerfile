# using python 3.9 slim to keep it small
FROM python:3.9-slim

# set the working directory inside the container
WORKDIR /app

# copy the requirements file first (to cache dependencies)
COPY requirements.txt .

# install dependencies
# --no-cache-dir keeps the image size down
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of your code into the container
COPY . .

# tell docker we want to listen on port 8000
EXPOSE 8000

# start the app
# host 0.0.0.0 is required for docker networking
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]