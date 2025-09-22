output "bucket_name" {
  description = "Name of the created S3 bucket"
  value       = module.s3_storage.bucket_name
}

output "bucket_arn" {
  description = "ARN of the created S3 bucket"
  value       = module.s3_storage.bucket_arn
}

output "bucket_region" {
  description = "Region of the created S3 bucket"
  value       = module.s3_storage.bucket_region
}