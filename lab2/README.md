# LAB2 - containerization

This project demonstrates containerization of a Streamlit game application using Docker. The game involves player and enemy movement on a grid. You as a player need to run from the enemy while also placing walls.

## requirements

requirements.in provides a list of required packages. Use pip-tools to compile it into requirements.txt for version control:
```
pip-compile --output-file=requirements.txt requirements.in
```

## docker overview

images are snapshots containing code and dependencies needed to run an application.
containers are running instances of Docker images that execute in an isolated environment.
dockerfiles are recipes that define how to build a Docker image with instructions for setup.

## dockerfile

My Dockerfile:
1. Uses Python 3.10 slim as the base image
2. Sets up the working directory to /app
3. Copies and installs dependencies from requirements.txt
4. Copies the application code to the container
5. Configures the container to run the Streamlit application as an entry point

## running docker

Build the Docker image:
```
docker build -t lab2 .
```

Run the container with port mapping:
```
docker run --rm -p 8501:8501 -e PORT=8501 lab2
```

## Registry

Push the image to Docker Hub:
```
docker tag lab2 justast/lab2
docker push justast/lab2
```

Pull an image from Docker Hub:
```bash
docker pull justast/lab2
```

## Heroku

We can use the containerized app to deploy to Heroku:
```bash
# Tag the image for Heroku registry
docker tag lab2 registry.heroku.com/fierce-beyond-09902/web

# Push to Heroku registry
docker push registry.heroku.com/fierce-beyond-09902/web

# Release the application
heroku container:release web -a fierce-beyond-09902
```

app: https://fierce-beyond-09902-12b01d0ae0ff.herokuapp.com/