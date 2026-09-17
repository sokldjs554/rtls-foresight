output "api_url" {
  value       = "http://${aws_lb.api.dns_name}"
  description = "서빙 API (ALB). /health, /docs, /predict, /risk"
}

output "artifacts_bucket" {
  value       = aws_s3_bucket.artifacts.bucket
  description = "DVC 원격 · MLflow 아티팩트 버킷"
}

output "ecr_repository_url" {
  value = aws_ecr_repository.serve.repository_url
}
