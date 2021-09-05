terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = var.region
  profile = var.aws_profile
}

provider "aws" {
  alias  = "acm_provider"
  region = "us-east-1"  # Certificate to be created in us-east-1 for Cloudfront to be able to use it.
}
