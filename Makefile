# S3 동기화를 위한 Makefile
# 사용법:
#   make upload_data      - 로컬 data 폴더를 S3에 업로드
#   make download_data    - S3에서 data 폴더 다운로드
#   make upload_models    - 모델 폴더만 업로드
#   make download_models  - 모델 폴더만 다운로드

# 설정 변수
BUCKET = covis-gnosis
PREFIX = p2_data
LOCAL_DATA_DIR = ./data
MODELS_DIR = ./models

# 성능 향상을 위한 AWS CLI 옵션
AWS_OPTS = --size-only --no-progress

.PHONY: help upload_data download_data upload_models download_models clean

# 기본 목표: 도움말 표시
help:
	@echo "사용 가능한 명령:"
	@echo "  make upload_data      - 로컬 data 폴더를 S3에 업로드"
	@echo "  make download_data    - S3에서 data 폴더 다운로드"
	@echo "  make upload_models    - 모델 폴더만 업로드"
	@echo "  make download_models  - 모델 폴더만 다운로드"
	@echo "  make clean            - 임시 파일 정리"
	@echo ""
	@echo "고급 사용법:"
	@echo "  make upload_data BUCKET=my-bucket PREFIX=my-prefix"
	@echo "  make upload_data AWS_OPTS=\"--exclude '*.tmp' --acl private\""

# S3 데이터 업로드
upload_data:
	@echo "데이터 폴더를 S3에 업로드 중..."
	aws s3 sync $(LOCAL_DATA_DIR) s3://$(BUCKET)/$(PREFIX)/data/ $(AWS_OPTS)
	@echo "업로드 완료!"

# S3에서 데이터 다운로드
download_data:
	@echo "S3에서 데이터 폴더 다운로드 중..."
	@mkdir -p $(LOCAL_DATA_DIR)
	aws s3 sync s3://$(BUCKET)/$(PREFIX)/data/ $(LOCAL_DATA_DIR) $(AWS_OPTS)
	@echo "다운로드 완료!"

# 모델 폴더만 업로드
upload_models:
	@echo "모델 폴더를 S3에 업로드 중..."
	@if [ -d "$(MODELS_DIR)/classifier" ]; then \
		aws s3 sync $(MODELS_DIR)/classifier s3://$(BUCKET)/$(PREFIX)/models/classifier/ $(AWS_OPTS); \
	else \
		echo "경고: $(MODELS_DIR)/classifier 폴더가 존재하지 않습니다."; \
	fi
	@if [ -d "$(MODELS_DIR)/yolo" ]; then \
		aws s3 sync $(MODELS_DIR)/yolo s3://$(BUCKET)/$(PREFIX)/models/yolo/ $(AWS_OPTS); \
	else \
		echo "경고: $(MODELS_DIR)/yolo 폴더가 존재하지 않습니다."; \
	fi
	@echo "모델 업로드 완료!"

# 모델 폴더만 다운로드
download_models:
	@echo "S3에서 모델 폴더 다운로드 중..."
	@mkdir -p $(MODELS_DIR)/classifier
	@mkdir -p $(MODELS_DIR)/yolo
	aws s3 sync s3://$(BUCKET)/$(PREFIX)/models/classifier/ $(MODELS_DIR)/classifier/ $(AWS_OPTS)
	aws s3 sync s3://$(BUCKET)/$(PREFIX)/models/yolo/ $(MODELS_DIR)/yolo/ $(AWS_OPTS)
	@echo "모델 다운로드 완료!"

# 임시 파일 정리
clean:
	@echo "임시 파일 정리 중..."
	find $(LOCAL_DATA_DIR) -name "*.tmp" -type f -delete
	find $(MODELS_DIR) -name "*.tmp" -type f -delete
	@echo "정리 완료!"