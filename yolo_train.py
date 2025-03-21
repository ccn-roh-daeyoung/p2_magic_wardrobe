import os
import yaml
from ultralytics import YOLO
import argparse
from datetime import datetime
import json

def train_yolo(config):
    """
    YOLOv8 모델을 훈련합니다.
    
    Args:
        config: 훈련 설정 객체
    """
    # 훈련 시작 시간 기록
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 저장 경로 설정
    if config.save_dir:
        save_dir = os.path.join(config.save_dir, f"train_{start_time}")
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = f"./runs/train_{start_time}"
    
    # 데이터셋 YAML 파일 확인
    if not os.path.isfile(config.data):
        raise FileNotFoundError(f"데이터셋 설정 파일을 찾을 수 없습니다: {config.data}")
    
    # 모델 로드
    if os.path.isfile(config.model):
        # 저장된 모델 가중치로부터 로드
        model = YOLO(config.model)
        print(f"기존 모델을 로드했습니다: {config.model}")
    else:
        # 사전 훈련된 모델에서 시작
        model = YOLO(config.model)
        print(f"사전 훈련된 모델을 로드했습니다: {config.model}")
    
    # 데이터셋 정보 출력
    try:
        with open(config.data, 'r') as f:
            data_config = yaml.safe_load(f)
        print(f"데이터셋 경로: {data_config.get('path', 'N/A')}")
        print(f"클래스 수: {data_config.get('nc', 'N/A')}")
        
        # 클래스 이름이 많으면 일부만 출력
        if 'names' in data_config:
            class_names = data_config['names']
            if len(class_names) > 10:
                print(f"클래스 이름: {class_names[:5]} ... (총 {len(class_names)}개)")
            else:
                print(f"클래스 이름: {class_names}")
                
        # val 경로가 정의되어 있지 않거나 train과 다르면 train으로 설정
        if 'val' not in data_config or config.no_val:
            data_config['val'] = data_config.get('train', '')
            # 수정된 설정을 임시 파일로 저장
            temp_yaml_path = os.path.join(os.path.dirname(config.data), f"temp_{os.path.basename(config.data)}")
            with open(temp_yaml_path, 'w') as f:
                yaml.dump(data_config, f)
            config.data = temp_yaml_path
            print(f"검증 데이터셋을 학습 데이터셋과 동일하게 설정했습니다.")
    except Exception as e:
        print(f"데이터셋 정보 로드 중 오류 발생: {str(e)}")
    
    # 훈련 설정 출력
    print("\n========= 훈련 설정 =========")
    print(f"모델: {config.model}")
    print(f"이미지 크기: {config.img_size}")
    print(f"배치 크기: {config.batch_size}")
    print(f"에폭 수: {config.epochs}")
    print(f"학습률: {config.learning_rate}")
    print(f"저장 경로: {save_dir}")
    print(f"워커 수: {config.workers}")
    print(f"검증 건너뛰기: {config.no_val}")
    print(f"Device: {config.device}")
    print("==============================\n")
    
    # 훈련 실행
    print("훈련을 시작합니다...")
    results = model.train(
        data=config.data,
        epochs=config.epochs,
        imgsz=config.img_size,
        batch=config.batch_size,
        lr0=config.learning_rate,
        device=config.device,
        workers=config.workers,
        project=os.path.dirname(save_dir),
        name=os.path.basename(save_dir),
        verbose=config.verbose,
        exist_ok=True,
        val=not config.no_val  # 검증 건너뛰기 옵션 추가
    )
    
    # 훈련 결과 저장
    print(f"\n훈련이 완료되었습니다. 모델이 저장된 위치: {results.save_dir}")
    
    # 최종 모델 평가
    if config.eval_after_train and not config.no_val:
        print("\n훈련된 모델을 평가합니다...")
        results = model.val()
        print("평가 결과:")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
    
    # 훈련 설정 저장
    config_dict = vars(config)
    with open(os.path.join(results.save_dir, 'train_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # 임시 파일 제거
    if 'temp_' in config.data:
        try:
            os.remove(config.data)
            print(f"임시 설정 파일을 제거했습니다: {config.data}")
        except:
            pass
    
    return results.save_dir

def main():
    parser = argparse.ArgumentParser(description='YOLO 모델 훈련 스크립트')
    
    # 필수 인자에 기본값 추가
    parser.add_argument('--data', type=str, default='demo/data/yolo_dataset/dataset.yaml', 
                       help='데이터셋 설정 파일 경로 (기본값: demo/data/yolo_dataset/dataset.yaml)')
    
    # 모델 관련 인자
    parser.add_argument('--model', type=str, default='yolo11n.pt', 
                        help='모델 선택 (예: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)')
    parser.add_argument('--img-size', type=int, default=640, help='입력 이미지 크기')
    
    # 훈련 관련 인자
    parser.add_argument('--epochs', type=int, default=50, help='훈련 에폭 수')
    parser.add_argument('--batch-size', type=int, default=16, help='배치 크기')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='초기 학습률')
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='훈련에 사용할 장치 (예: 0, 0,1,2,3, cpu)')
    parser.add_argument('--workers', type=int, default=8, help='데이터 로딩 워커 수')
    
    # 기타 인자
    parser.add_argument('--save-dir', type=str, default='', help='모델 저장 디렉토리')
    parser.add_argument('--eval-after-train', action='store_true', help='훈련 후 평가 실행')
    parser.add_argument('--verbose', action='store_true', help='상세 출력 활성화')
    parser.add_argument('--no-val', action='store_true', help='검증 데이터셋 사용하지 않음')
    
    args = parser.parse_args()
    train_yolo(args)

if __name__ == "__main__":
    main()