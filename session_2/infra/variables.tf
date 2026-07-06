# infra/variables.tf
variable "project_name" {
  description = "Unique project identifier used in resource names"
  type        = string
  default     = "ride-duration"
}
