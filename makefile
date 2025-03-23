PROJECT=bigdata
VERSION=3.10
VENV_DIR=.venv

help: ## Show this help
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean: ## Clean virtualenv and state
	rm -rf .state
	rm -rf $(VENV_DIR)

venv: $(VENV_DIR) ## Setup virtualenv
$(VENV_DIR):
	python3 -m virtualenv --python=python$(VERSION) $(VENV_DIR)

activate: ## Print the command to activate the virtual environment
	@echo "To activate the virtual environment, run:"
	@echo "source $(VENV_DIR)/bin/activate"

install: .state/venv ## Install Python dependencies
.state/venv: $(VENV_DIR) requirements.txt
	. $(VENV_DIR)/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt
	
	# Mark the state so we don't reinstall needlessly
	mkdir -p .state
	touch .state/venv

run: ## Run the vessel spoofing detection script
	. $(VENV_DIR)/bin/activate && \
	python lab1_multiproc/vessel_spoofing_detection.py

create_test_file: ## Create a test file with first 5000 lines
	head -n 5000 aisdk-test.csv > aisdk-test.csv

.PHONY: help clean venv activate install run create_test_file

# Command to run in the cluster
# srun --pty $SHELL
