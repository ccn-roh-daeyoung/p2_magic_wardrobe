# P2 마법의 옷장

## 개요
P2 마법의 옷장 프로젝트는 의류 아이템 분류 및 검출을 위한 AI 모델을 학습하고 배포하는 프로젝트입니다.

## 데이터 동기화

프로젝트에서는 AWS S3를 사용하여 데이터 및 모델을 관리합니다. 효율적인 데이터 동기화를 위해 Makefile을 제공합니다.

### 사용 방법

1. **데이터 폴더 전체 업로드**:
   ```bash
   make upload_data
   ```

2. **데이터 폴더 전체 다운로드**:
   ```bash
   make download_data
   ```

3. **모델 폴더만 업로드**:
   ```bash
   make upload_models
   ```

4. **모델 폴더만 다운로드**:
   ```bash
   make download_models
   ```


## 주요 Python 스크립트

### yolo_train.py
YOLO 모델 학습을 위한 스크립트입니다. 다음의 필수 인자를 잘 설정해서 실행하면 됩니다:

```bash
python3 yolo_train.py --data demo/data/yolo_dataset/dataset.yaml
```

필수 인자:
```python
parser.add_argument('--data', type=str, default='demo/data/yolo_dataset/dataset.yaml', 
                   help='데이터셋 설정 파일 경로 (기본값: demo/data/yolo_dataset/dataset.yaml)')
```

### classifier_train.py
분류 모델 학습을 위한 스크립트입니다. 다음과 같은 방식으로 실행하면 됩니다:

```bash
python3 classifier_train.py --data_dir data/item_dataset_accesory_back --model_name efficientnet_b0 --device cuda:0
```

### config.py
Gradio 애플리케이션의 환경 설정을 담당하는 파일입니다. 이 파일에서 모델 불러오는 경로 등을 확인하고 설정할 수 있습니다.

### app.py
Gradio 데모를 실행하기 위한 메인 스크립트입니다. 이 스크립트를 실행하여 웹 인터페이스를 통해 모델을 테스트할 수 있습니다.

```bash
python3 app.py
```

## AWS 설정

AWS CLI가 올바르게 설정되어 있어야 S3 동기화가 작동합니다:

```bash
aws configure
```

또는 환경 변수를 사용할 수도 있습니다:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=your_region  # 예: ap-northeast-2
```