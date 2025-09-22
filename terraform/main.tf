terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# S3 module for data storage
module "s3_storage" {
  source = "./modules/s3"

  bucket_name = var.bucket_name
  environment = var.environment
}