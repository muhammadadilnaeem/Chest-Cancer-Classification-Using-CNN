# Use the official Python 3.8 slim image as the base image
FROM python:3.8-slim-buster

# Update the package list and install the AWS CLI
RUN apt update -y && apt install awscli -y

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the current directory contents into the /app directory in the container
COPY . /app

# Install the Python dependencies listed in requirements.txt
RUN pip install -r requirements.txt

# Specify the command to run the application
CMD ["python3", "app.py"]