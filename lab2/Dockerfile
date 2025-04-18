# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.in .

# Install any needed packages specified in requirements.in
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY ./app /app/app

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV NAME=World

# Run dashboard.py when the container launches
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]