# S3 bucket for AI Scientist Team data storage
resource "aws_s3_bucket" "ai_scientist_data" {
  bucket = var.bucket_name

  tags = {
    Name        = "AI Scientist Team Data Storage"
    Environment = var.environment
    Project     = "ai-scientist-team"
  }
}

# S3 bucket versioning
resource "aws_s3_bucket_versioning" "ai_scientist_data_versioning" {
  bucket = aws_s3_bucket.ai_scientist_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 bucket server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "ai_scientist_data_encryption" {
  bucket = aws_s3_bucket.ai_scientist_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 bucket public access block
resource "aws_s3_bucket_public_access_block" "ai_scientist_data_pab" {
  bucket = aws_s3_bucket.ai_scientist_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}