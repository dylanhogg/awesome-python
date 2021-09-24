.EXPORT_ALL_VARIABLES:
APP_NAME=crazy-awesome-python-api
BUCKET_NAME=crazy-awesome-python-api.infocruncher.com
ENV=prd
AWS_PROFILE=prd-non-tf-905234897161
AWS_ACCOUNT_ID=905234897161
AWS_REGION=us-east-1

# AWS Command Tools

## AWS S3 cp json data to S3 bucket
s3-deploy-files:
	aws s3 cp ../github_data.json s3://${BUCKET_NAME}/github_data.json --profile ${AWS_PROFILE}
	# NOTE: Remember to invalidate Cloudfront cache to materialise changes

## AWS S3 rm bucket
s3-remove-files:
	aws s3 rm s3://${BUCKET_NAME} --recursive

## Browse to public S3 url
s3-browse:
	open http://${BUCKET_NAME}.s3-website-${AWS_REGION}.amazonaws.com/github_data.json

## Browse to public Route53 api url
r53-browse:
	open https://${BUCKET_NAME}/github_data.json

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


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
