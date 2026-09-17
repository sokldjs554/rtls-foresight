# AWS 배포 (Terraform)

만드는 것: S3(버전 관리, 암호화; DVC 원격 + MLflow 아티팩트) · ECR · ECS Fargate 서비스(서빙 API, 0.5 vCPU / 1 GB) · ALB(`/health`) ·
CloudWatch Logs · IAM(태스크 실행/S3 읽기). 기본 VPC 를 쓰며 `vpc_id`/`public_subnet_ids` 로 바꿀 수 있다.

```bash
cd deploy/terraform
terraform init && terraform plan -var image=ghcr.io/sokldjs554/rtls-foresight:<sha>
terraform apply
terraform output api_url
```

파이프라인을 S3 에 붙이기:
```bash
dvc remote modify storage url s3://$(terraform output -raw artifacts_bucket)/dvc && dvc remote modify --unset storage endpointurl
export MLFLOW_TRACKING_URI=http://<mlflow-host>:5000   # MLflow 서버는 EC2/ECS 에 따로 (아티팩트 root 를 같은 버킷의 mlflow/ 로)
```

## 비용 감각 (서울 리전, 온디맨드, 2026년 기준 대략)
| 항목 | 월 비용 |
|---|---|
| Fargate 0.5 vCPU / 1 GB × 1 태스크 | 약 $18 |
| ALB | 약 $20 + LCU |
| S3 수십 GB + 요청 | $1~3 |
| CloudWatch Logs 14일 | $1 미만 |

## 의도적으로 만들지 않은 것
- **MSK(Kafka)**: 최소 구성도 월 $150+ 이고 이 프로젝트의 스트리밍 소비자는 브로커 구현에 독립적이다. 현장에서는 회사가 파트너인
  **Confluent Cloud**(Schema Registry 포함) 또는 EC2 위 Redpanda 를 권한다. 소비자는 `--bootstrap` 만 바꾸면 된다.
- **MLflow 서버 자체**: 컨테이너 하나짜리라 `docker-compose.yml` 의 정의를 EC2/ECS 로 옮기면 된다. 팀 규모에 맞춰 RDS 백엔드를 붙인다.
- **HTTPS/도메인**: ACM 인증서 + 443 리스너를 추가하면 된다. 데모 범위에서는 80 만 열었다.

## 검증 상태
이 정의는 `terraform validate` 를 **통과시키지 못했다** — 빌드 환경에 terraform 바이너리가 없고 AWS 자격 증명도 없다.
문법·리소스 인자는 AWS provider 5.x 문서 기준으로 작성했고, 실제 `apply` 전에는 `terraform init && terraform validate && terraform plan` 으로 확인해야 한다.
