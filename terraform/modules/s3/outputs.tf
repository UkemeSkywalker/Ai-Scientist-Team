output "bucket_name" {
  description = "Name of the created S3 bucket"
  value       = aws_s3_bucket.ai_scientist_data.bucket
}

output "bucket_arn" {
  description = "ARN of the created S3 bucket"
  value       = aws_s3_bucket.ai_scientist_data.arn
}

output "bucket_region" {
  description = "Region of the created S3 bucket"
  value       = aws_s3_bucket.ai_scientist_data.region
}

output "bucket_id" {
  description = "ID of the created S3 bucket"
  value       = aws_s3_bucket.ai_scientist_data.id
}