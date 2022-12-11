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

## Run black code formatter
black:
	source venv/bin/activate ; black --line-length 120 .

## Run tests
test:
	source venv/bin/activate ; PYTHONPATH='./src' pytest --capture=no tests

## Serve local client
serve-local-client:
	open http://localhost:8002/
	cd client/app; python3 -m http.server 8002

## AWS S3 cp client app to S3
s3-deploy-app:
	cd client; make s3-deploy-app; make cf-invalidation

## Deploy server json data
s3-deploy-files:
	cd server; make s3-deploy-files; make cf-invalidation

.DEFAULT_GOAL := help
.PHONY: help
help:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'
