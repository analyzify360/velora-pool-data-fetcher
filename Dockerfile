FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install python-dotenv to load environment variables from .env file
RUN pip install python-dotenv

# Copy the rest of the application code and .env file
COPY . .

# Command to run the application
CMD ["python", "main.py"]