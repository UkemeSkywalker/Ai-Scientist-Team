# AI Scientist Team Infrastructure

This directory contains Terraform configuration for the AI Scientist Team infrastructure.

## Prerequisites

1. Install Terraform (>= 1.0)
2. Configure AWS credentials:
   ```bash
   export AWS_ACCESS_KEY_ID="your-access-key"
   export AWS_SECRET_ACCESS_KEY="your-secret-key"
   export AWS_DEFAULT_REGION="us-east-1"
   ```

## Usage

1. Copy the example variables file:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```

2. Edit `terraform.tfvars` with your desired values:
   ```hcl
   aws_region    = "us-east-1"
   bucket_name   = "your-unique-bucket-name"
   environment   = "development"
   ```

3. Initialize Terraform:
   ```bash
   terraform init
   ```

4. Plan the deployment:
   ```bash
   terraform plan
   ```

5. Apply the configuration:
   ```bash
   terraform apply
   ```

## Resources Created

### S3 Module (`modules/s3/`)
- **S3 Bucket**: For storing processed datasets and research outputs (permanent storage)
- **S3 Bucket Versioning**: Enabled for data protection
- **S3 Encryption**: Server-side encryption with AES256
- **S3 Public Access Block**: Prevents accidental public access

## Module Structure

```
terraform/
├── main.tf                 # Main configuration using modules
├── variables.tf            # Root-level variables
├── outputs.tf             # Root-level outputs
├── terraform.tfvars       # Variable values
└── modules/
    └── s3/                # S3 storage module
        ├── main.tf        # S3 resources
        ├── variables.tf   # S3 module variables
        └── outputs.tf     # S3 module outputs
```

## Outputs

- `bucket_name`: The name of the created S3 bucket
- `bucket_arn`: The ARN of the S3 bucket
- `bucket_region`: The region where the bucket was created

## Cleanup

To destroy the infrastructure:
```bash
terraform destroy
```

**Note**: Make sure to backup any important data before destroying the infrastructure.