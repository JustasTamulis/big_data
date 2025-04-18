PROJECT=bigdata_task2
VERSION=3.10

VENV_DIR=$(shell pyenv root)/versions/${VENV}
VENV=${PROJECT}-${VERSION}


help: ## Show this help
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'


########################################################################################
## Setup commands

clean: ## clean virtualenv
	rm -rf .state
	pyenv virtualenv-delete --force ${VENV}
	rm .python-version

venv: $(VENV_DIR) ## Setup venv
$(VENV_DIR):
	pyenv install -s ${VERSION}
	pyenv virtualenv ${VERSION} ${VENV}
	pyenv local $(VENV)


install: .state/pyvenv ## Install Python dependencies
.state/pyvenv: $(VENV_DIR) requirements.txt
	pyenv local $(VENV)
	pip install --upgrade pip
	pip install -r requirements.txt

	# Mark the state so we don't reinstall needlessly
	mkdir -p .state
	touch .state/pyvenv

########################################################################################
## Docker

IMAGE_NAME=lab2

build: ## Build the Docker image
	docker build -t $(IMAGE_NAME) .

run: ## Run the Docker container and generate the plot
	mkdir -p output
	# Run the container
	# mount current output folder to the /app/output folder
	docker run --rm -v "./output:/app/output" lab2 python plot_data.py --output /app/output/docker_output_plot.png

cli: ## Run the Docker container interactively
	# Run the container, mounting the current directory's output folder to the container's /app/output folder
	docker run -it lab2:latest /bin/bash

docker-clean: ## Remove the Docker image
	docker rmi $(IMAGE_NAME):$(DOCKER_TAG) -f

# Add docker targets to the main clean target if desired
# clean: ## clean virtualenv and docker image
# 	rm -rf .state output
# 	pyenv virtualenv-delete --force ${VENV} || true
# 	rm -f .python-version
# 	docker rmi $(IMAGE_NAME):$(DOCKER_TAG) || true
