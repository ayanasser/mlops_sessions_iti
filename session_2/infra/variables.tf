# infra/variables.tf
variable "project_name" {
  description = "Unique project identifier used in resource names"
  type        = string
  default     = "ride-duration"
}

# infra/terraform.tfvars  (gitignored — holds real values)
project_name = "ride-duration-prod"