variable "region" {
  description = "AWS 리전 (서울)"
  type        = string
  default     = "ap-northeast-2"
}

variable "name" {
  description = "리소스 이름 접두어"
  type        = string
  default     = "rtls-foresight"
}

variable "image" {
  description = "서빙 이미지 (CI 가 GHCR 에 올린 태그를 ECR 로 복제하거나 GHCR 을 직접 참조)"
  type        = string
  default     = "ghcr.io/sokldjs554/rtls-foresight:latest"
}

variable "cpu" {
  description = "Fargate vCPU 단위 (256 = 0.25 vCPU). 1 스레드 서빙이므로 512 면 충분하다."
  type        = number
  default     = 512
}

variable "memory" {
  description = "Fargate 메모리 (MiB)"
  type        = number
  default     = 1024
}

variable "desired_count" {
  description = "API 태스크 수 — 수평 확장 단위 (docs/serving_streaming.md: 스레드가 아니라 워커/레플리카로 확장)"
  type        = number
  default     = 1
}

variable "vpc_id" {
  description = "기존 VPC ID (없으면 기본 VPC 사용)"
  type        = string
  default     = null
}

variable "public_subnet_ids" {
  description = "ALB·Fargate 가 쓸 퍼블릭 서브넷 ID 목록 (없으면 기본 VPC 의 서브넷)"
  type        = list(string)
  default     = []
}
