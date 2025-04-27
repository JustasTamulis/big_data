# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY ./app /app/app

# Make port 8501 available to the world outside this container
# EXPOSE 8501

# Run dashboard.py when the container launches
# CMD ["streamlit", "run", "app/game_app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
# Run dashboard.py when the container launches
CMD streamlit run app/game_app.py --server.port=$PORT --server.address=0.0.0.0