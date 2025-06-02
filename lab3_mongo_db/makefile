PROJECT=bigdata_task3
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
	pyenv local $(VENV)
	pip-compile -o requirements.txt requirements.in


########################################################################################
## Docker MongoDB Cluster commands

network: ## Create the Docker network for MongoDB
	docker network create mynet

cluster: ## Create and start the MongoDB cluster in detached mode
	docker-compose up -d

cluster-down: ## Stop and remove cluster containers, networks, and volumes
	docker-compose down --volumes

cluster-status: ## Check the status of the cluster services
	docker-compose ps