# infra/main.tf

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" { region = "us-east-1" }

# ── S3 bucket for DVC remote storage ──────────────────
resource "aws_s3_bucket" "dvc_remote" {
  bucket = "mlops-dvc-artifacts-${var.project_name}"
  tags   = { Project = var.project_name, ManagedBy = "terraform" }
}

# ── ECR repository for Docker images ──────────────────
resource "aws_ecr_repository" "api" {
  name                 = "${var.project_name}/ride-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration { scan_on_push = true }
}

# ── Outputs ───────────────────────────────────────────
output "s3_bucket_name" { value = aws_s3_bucket.dvc_remote.bucket }
output "ecr_repo_url"   { value = aws_ecr_repository.api.repository_url }
