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
	cd lab1 && \
	python test_hpc.py

create_test_file: ## Create a test file with first 5000 lines
	head -n 1000000 data/aisdk-2025-02-09.csv > data/aisdk-test.csv

hpc_github_setup: 
	@eval "$$(ssh-agent -s)"
	ssh-add ~/.ssh/github

# Slurm commands:

slurm_usage:
	sreport -T cpu,mem,gres/gpu cluster AccountUtilizationByUser Start=0301 End=0331 User=juta1001

slurm_nodes:
	scontrol show node

slurm_queue:
	squeue -o"%.7i %.9P %.8j %.8u %.2t %.10M %.6D %C"

.PHONY: help clean venv activate install run create_test_file

# Command to run in the cluster
# srun --pty $SHELL
