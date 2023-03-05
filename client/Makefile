.EXPORT_ALL_VARIABLES:
APP_NAME=crazy-awesome-python
ENV=prd
AWS_PROFILE=prd-non-tf-905234897161
AWS_ACCOUNT_ID=905234897161
AWS_REGION=us-east-1
BUCKET_NAME=awesome-python.infocruncher.com

# Local running

## Serve local js client app
serve-local-app:
	cd app; python3 -m http.server 8002

## Copy processed data to local client for testing
copy-data-local-app: remove-data-local-app
	cp ../github_data.ui.min.json app/
	cp ../github_tags_data.json app/
	cp -r ../data app/data

## Remove processed data from local client for testing
remove-data-local-app:
	rm app/github_data.json || true
	rm app/github_data.min.json || true
	rm app/github_data.ui.min.json || true
	rm -rf app/data

## Browse local js client
browse-s3-app:
	open http://${BUCKET_NAME}/

## Browse to public S3 url
browse-s3:
    open http://${BUCKET_NAME}.s3-website-${AWS_REGION}.amazonaws.com/


# AWS Command Tools

## AWS S3 cp app to S3
s3-deploy-app: remove-data-local-app
	aws s3 cp app s3://${BUCKET_NAME} --recursive --profile ${AWS_PROFILE}
	aws s3 cp ../github_data.csv s3://${BUCKET_NAME} --profile ${AWS_PROFILE}
	aws s3 cp ../github_data.min.json s3://${BUCKET_NAME} --profile ${AWS_PROFILE}
	aws s3 cp ../github_data.json s3://${BUCKET_NAME} --profile ${AWS_PROFILE}
	aws s3 cp ../data s3://${BUCKET_NAME}/data --recursive --profile ${AWS_PROFILE}
	# NOTE: Remember to invalidate Cloudfront cache to materialise changes: "make cf-invalidation"

## AWS S3 cp app to S3 (no data)
s3-deploy-app-only: remove-data-local-app
	aws s3 cp app s3://${BUCKET_NAME} --recursive --profile ${AWS_PROFILE}
	# NOTE: Remember to invalidate Cloudfront cache to materialise changes: "make cf-invalidation"

## AWS S3 cp static files to S3
s3-deploy-static:
	aws s3 cp app/docs.html s3://${BUCKET_NAME} --profile ${AWS_PROFILE}
	aws s3 cp app/app.html s3://${BUCKET_NAME} --profile ${AWS_PROFILE}
	# NOTE: Remember to invalidate Cloudfront cache to materialise changes: "make cf-invalidation"

## AWS S3 cp app to S3
s3-deploy-app-metadata:
	aws s3 cp app/sitemap.xml s3://${BUCKET_NAME} --profile ${AWS_PROFILE}
	# NOTE: Remember to invalidate Cloudfront cache to materialise changes: "make cf-invalidation"

## AWS S3 rm static_files from S3
s3-delete-app:
	aws s3 rm s3://${BUCKET_NAME} --recursive

## AWS Cloudfront invalidation (limit 1000 free calls per month)
cf-invalidation:
	# www.awesomepython.org:
	aws cloudfront create-invalidation --distribution-id EJHF3POVTKFJ4 --paths "/*"
	# awesomepython.org:
	aws cloudfront create-invalidation --distribution-id EUNSHU4X9PVCT --paths "/*"
	# awesome-python.infocruncher.com:
	aws cloudfront create-invalidation --distribution-id E1VXP77O89KVKO --paths "/*"

# Terraform

## TF init
tf-init:
	cd infra; terraform init -var-file=variables.tfvars; cd -

## TF validate
tf-validate:
	cd infra; terraform validate; cd -

## TF plan
tf-plan:
	cd infra; terraform plan -var-file=variables.tfvars; cd -

## TF apply
tf-apply: tf-plan
	cd infra; terraform apply -var-file=variables.tfvars -auto-approve; cd -

## TF destroy
tf-destroy:
	cd infra; terraform destroy -var-file=variables.tfvars -auto-approve; cd -

.DEFAULT_GOAL := help
.PHONY: help
help:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'
