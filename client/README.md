# Web Application for Awesome Python

Deploy html/js web application to AWS S3: 

```
cd client
make tf-init
make tf-plan
make tf-apply
make s3-deploy-app
```

Client app files: `client/app/`

Terraform infrastructure for client app: `client/infra/`
  
Example app deployment: `http://awesome-python.infocruncher.com/`
