.EXPORT_ALL_VARIABLES:
AWS_PROFILE=prd-non-tf-905234897161
BUCKET_NAME=awesome-python.infocruncher.com

.PHONY: venv
## Create virtual environment
venv:
	python3 -m venv venv
	source venv/bin/activate ; pip install --upgrade pip ; python3 -m pip install -r requirements.txt
	source venv/bin/activate ; pip freeze > requirements_freeze.txt

## Clean virtual environment
clean:
	rm -rf venv

## Run the app
run:
	source venv/bin/activate ; PYTHONPATH='./src' python -m app

## Run the app, with no cache
run-clean-cache:
	rm -rf .joblib_cache
	source venv/bin/activate ; PYTHONPATH='./src' python -m app

## Run black code formatter
black:
	source venv/bin/activate ; black --line-length 120 .

## Run tests
test:
	source venv/bin/activate ; PYTHONPATH='./src' pytest -vv --capture=no tests

## Clear joblib cache
clear-cache:
	rm -rf .joblib_cache

## View API rate limits
api-rates:
	curl -I https://api.github.com/users/dylanhogg

## Serve local client
serve-local-client:
	open http://localhost:8002/
	cd client/app; python3 -m http.server 8002

## AWS S3 cp app and data to S3
s3-deploy-app-full:
	cd client; make s3-deploy-app-full; make cf-invalidation

## AWS S3 cp app to S3 (no data)
s3-deploy-app-only:
	cd client; make s3-deploy-app-only; make cf-invalidation

## Deploy server json data
s3-deploy-files:
	cd server; make s3-deploy-files; make cf-invalidation

## Run jupyter lab
jupyter:
	source venv/bin/activate; PYTHONPATH='./src' jupyter lab

.DEFAULT_GOAL := help
.PHONY: help
help:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'
