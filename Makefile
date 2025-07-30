#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = resnet_hearthealth_monitor
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 resnet_hearthealth_monitor
	isort --check --diff --profile black resnet_hearthealth_monitor
	black --check --config pyproject.toml resnet_hearthealth_monitor

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml resnet_hearthealth_monitor




## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset from zip file or download from URL if not present
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) resnet_hearthealth_monitor/dataset.py

## Preprocess dataset contained within the raw folder
.PHONY: preprocess
preprocess: requirements
	$(PYTHON_INTERPRETER) resnet_hearthealth_monitor/data_processing/preprocess.py

# ## Run an Optuna study 
# .PHONY: run_study
# run_study: requirements
# 	$(PYTHON_INTERPRETER) resnet_hearthealth_monitor/modeling/train.py run_study

## Copy best model (and it's associated records) to ./models/best_model for prediction or plotting
.PHONY: find_best_study
find_best_study: requirements
	$(PYTHON_INTERPRETER) resnet_hearthealth_monitor/modeling/train.py find_best_study

## generate training loss plots for best model and save to ./reports/figures
.PHONY: gen_training_loss_plots
gen_training_loss_plots: requirements
	$(PYTHON_INTERPRETER) resnet_hearthealth_monitor/plots.py gen_training_loss_plots

## generate classification report table for best model and save to ./reports/figures
.PHONY: save_classif_figure
save_classif_figure: requirements
	$(PYTHON_INTERPRETER) resnet_hearthealth_monitor/plots.py save_classif_figure

# ## use best model to generate predictions for a test set with known targets values
# .PHONY: testset_prediction
# testset_prediction: requirements
# 	$(PYTHON_INTERPRETER) resnet_hearthealth_monitor/modeling/predict.py

## use best model to generate a prediction for a new sample
# if you don’t pass SEGMENT on the make‑cmdline, it stays empty
SEGMENT ?=

# only define the flag if SEGMENT was set
ifdef SEGMENT
  SEG_OPT = --segment '$(SEGMENT)'
endif

.PHONY: predict_segment
predict_segment: requirements
ifeq ($(origin SEGMENT), default)
	@echo "▶ No SEGMENT passed; script will generate a random segment to use."
endif
	$(PYTHON_INTERPRETER) \
		resnet_hearthealth_monitor/modeling/inference.py \
		$(SEG_OPT) \
		--device cuda


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
