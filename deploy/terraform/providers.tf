terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.60" }
  }
  # 상태 파일은 팀 공유용 S3 백엔드를 권한다. 처음 한 번은 로컬로 만들고 옮긴다.
  # backend "s3" { bucket = "<state-bucket>" key = "rtls-foresight/terraform.tfstate" region = "ap-northeast-2" }
}

provider "aws" {
  region = var.region
  default_tags { tags = { project = "rtls-foresight", managed_by = "terraform" } }
}
