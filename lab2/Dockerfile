# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Make port 80 available to the world outside this container (if needed for web apps, optional here)
# EXPOSE 80 

# Define environment variable
ENV NAME=World

# Run plot_data.py when the container launches
# Default command, can be overridden
CMD ["python", "plot_data.py"]