# Server data API for Awesome Python

Deploy json data API to AWS S3: 

```
# Crawl github and get data file
make run

# Deploy data API
cd server
make tf-init
make tf-plan
make tf-apply
make s3-deploy-files
```

Server data API file: `/github_data.json`

Terraform infrastructure for server: `server/infra/`

Example API deployment: `http://prd-s3-crazy-awesome-python-api.s3-website-us-east-1.amazonaws.com/github_data.json`
