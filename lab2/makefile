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

requirements: ## Update requirements.txt
	pip-compile -o requirements.txt requirements.in

########################################################################################
## Docker

IMAGE_NAME=lab2

build: ## Build the Docker image
	docker build --provenance false -t $(IMAGE_NAME) .

run: ## Run the Docker container
	docker run --rm -p 8501:8501 -e PORT=8501 $(IMAGE_NAME)

cli: ## Run the Docker container interactively
	docker run -it $(IMAGE_NAME):latest /bin/bash

docker-clean: ## Remove the Docker image
	docker rmi $(IMAGE_NAME) -f

push: ## Push the Docker image to Docker Hub
	docker tag $(IMAGE_NAME):latest justast/$(IMAGE_NAME)
	docker push justast/$(IMAGE_NAME)

	
########################################################################################
## Heroku

HEROKU_APP_NAME=fierce-beyond-09902

heroku-build: build
	# tag the image with the Heroku registry
	docker tag $(IMAGE_NAME) registry.heroku.com/$(HEROKU_APP_NAME)/web

	# push the image to Heroku
	docker push registry.heroku.com/$(HEROKU_APP_NAME)/web

	# release the image
	heroku container:release web -a $(HEROKU_APP_NAME)

heroku-pull: ## Pull the image from docker hub and deploy to Heroku
	# pull the image from Docker Hub
	docker pull justast/$(IMAGE_NAME)

	# tag the image with the Heroku registry
	docker tag justast/$(IMAGE_NAME) registry.heroku.com/$(HEROKU_APP_NAME)/web

	# push the image to Heroku
	docker push registry.heroku.com/$(HEROKU_APP_NAME)/web

	# release the image
	heroku container:release web -a $(HEROKU_APP_NAME)

open: ## Open the Heroku app in the browser
	heroku open -a $(HEROKU_APP_NAME)
