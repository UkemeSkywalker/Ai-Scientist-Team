variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "bucket_name" {
  description = "Name of the S3 bucket for AI Scientist Team data"
  type        = string
  default     = "ai-scientist-team-data-unique-2024"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "development"
}