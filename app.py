"""
Gradio를 사용한 패션 분석 데모 앱 - Onepiece 추가 분류 기능 포함 및 pokecolo 검출 기능 추가
"""

import os
import gradio as gr
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import time
# 모듈 가져오기
from config import (
    YOLO_MODEL_PATH, 
    C0ORD_YOLO_MODEL_PATH,
    EYE_YOLONECK_MODEL_PATH,
    CLASSIFIER_PATHS, 
    CLASS_MAPPING_PATHS, 
    ITEMS_METADATA_PATH, 
    TEMP_DIR,
    SUPPORTED_LABELS
)
from models.detector import FashionDetector
from models.classifier import FashionClassifier
from models.eye_detector import EyeDetectionProcess
from models.sift import SIFTReranker
from utils.image_processing import (
    save_crop, 
    draw_detection_results, 
    load_items_metadata, 
    get_image_url_from_class
)

# 모델 및 유틸리티 클래스 초기화
detector = FashionDetector(YOLO_MODEL_PATH)
coord_detector = FashionDetector(C0ORD_YOLO_MODEL_PATH)
classifier = FashionClassifier()
eye_detector = EyeDetectionProcess(EYE_YOLONECK_MODEL_PATH)
sift_reranker = SIFTReranker()
items_dict = load_items_metadata(ITEMS_METADATA_PATH)

# 메인 카테고리 정의
ALL_CATEGORIES = [
    "ONEPIECE", "HAIR", "EYE", "SHOES"
]

# 세부 카테고리 정의
ONEPIECE_SUBCATEGORIES = ["OUTER", "TOP", "BOTTOM", "ACCESSORY_BODY", "ACCESSORY_BACK", "ACCESSORY_RIGHT_LEFT"]
HAIR_SUBCATEGORIES = ["ACCESSORY_HEAD"]

TOP_K = 10  # 상위 K개 결과 표시

# 분류기 모델 로드
print("분류기 모델 로드 중...")
print(SUPPORTED_LABELS)
for label in SUPPORTED_LABELS:
    if label in CLASSIFIER_PATHS and os.path.exists(CLASSIFIER_PATHS[label]):
        mapping_path = CLASS_MAPPING_PATHS.get(label)
        classifier.load_model(label, CLASSIFIER_PATHS[label], mapping_path)
    else:
        print(f"경고: 라벨 '{label}'에 대한 분류기 모델을 찾을 수 없습니다.")

def load_image_from_url(url):
    """URL에서 이미지 로드"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            return None
    except Exception as e:
        print(f"이미지 로드 중 오류: {str(e)}")
        return None

def create_default_crop(image_path, crop_type):
    """
    HAIR, ONEPIECE, SHOES가 검출되지 않을 경우 이미지의 특정 부분을 크롭하여 반환
    
    Args:
        image_path: 입력 이미지 경로
        crop_type: 크롭할 타입 (HAIR, ONEPIECE, SHOES)
        
    Returns:
        PIL.Image: 크롭된 이미지
    """
    try:
        # 이미지 열기
        img = Image.open(image_path)
        width, height = img.size
        
        if crop_type == "HAIR":
            # 이미지 그대로 사용
            return img
        elif crop_type == "HEAD":
            # 이미지 상단 1/2
            return img.crop((0, 0, width, height // 2))
        elif crop_type == "ONEPIECE":
            # 이미지 하단 1/2
            return img.crop((0, height // 2, width, height))
        elif crop_type == "SHOES":
            # 이미지 하단 1/3
            return img.crop((0, height * 2 // 3, width, height))
        else:
            return None
    except Exception as e:
        print(f"기본 크롭 생성 중 오류: {str(e)}")
        return None

def detect_pokecolo(image_path, padding_percent=0.1, top_padding_percent=0.00):
    """
    입력 이미지에서 pokecolo 객체를 탐지하고, 가장 신뢰도가 높은 하나만 반환
    주변에 여유 공간(padding)을 추가하여 크롭합니다. 위쪽은 별도로 패딩을 조절할 수 있습니다.
    
    Args:
        image_path: 입력 이미지 경로
        padding_percent: 아래/좌/우에 추가할 여유 공간 비율 (원본 높이/너비의 비율)
        top_padding_percent: 위쪽에 추가할 여유 공간 비율 (원본 높이의 비율)
    
    Returns:
        tuple: (처리할 이미지 경로, pokecolo가 검출되었는지 여부)
              pokecolo가 검출되면 크롭된 이미지 경로, 아니면 원본 이미지 경로 반환
    """
    # 원본 이미지 로드하여 크기 확인
    orig_img = Image.open(image_path)
    img_width, img_height = orig_img.size
    
    # coord_detector로 pokecolo 객체 감지
    results, crops, labels, boxes = coord_detector.detect(
        image_path,
        conf_threshold=0.1,  # 낮은 신뢰도라도 일단 검출
        save_result=False
    )
    
    # pokecolo 객체가 있는지 확인
    pokecolo_crops = []
    pokecolo_boxes = []
    
    for crop, label, box in zip(crops, labels, boxes):
        if label == "pokecolo":  # pokecolo 라벨 확인
            confidence = box[4] if len(box) > 4 else 0.0
            pokecolo_crops.append((crop, confidence, box))
            pokecolo_boxes.append(box)
    
    # pokecolo가 검출되지 않은 경우 원본 이미지 반환
    if not pokecolo_crops:
        print("pokecolo가 검출되지 않았습니다. 원본 이미지를 사용합니다.")
        return image_path, False
    
    # 신뢰도가 가장 높은 pokecolo 선택
    best_pokecolo = max(pokecolo_crops, key=lambda x: x[1])
    best_crop, best_conf, best_box = best_pokecolo
    
    print(f"신뢰도 {best_conf:.2f}의 pokecolo를 검출했습니다.")
    
    # 기존 바운딩 박스에서 상하 여유 공간을 추가하여 새로운 크롭 영역 계산
    # 바운딩 박스는 [x1, y1, x2, y2, conf, class_id] 형태
    x1, y1, x2, y2 = best_box[:4]
    
    # 패딩 계산 (바운딩 박스 높이/너비의 비율만큼)
    box_height = y2 - y1
    box_width = x2 - x1
    
    # 위쪽/아래쪽/좌우 패딩을 각각 계산
    top_padding = int(box_height * top_padding_percent)  # 위쪽은 별도 비율 적용
    bottom_padding = int(box_height * padding_percent)   # 아래쪽 패딩
    padding_horizontal = int(box_width * padding_percent)  # 좌우 패딩
    
    # 패딩 추가 (이미지 경계를 벗어나지 않도록 제한)
    new_y1 = max(0, y1 - top_padding)  # 위쪽 패딩 적용
    new_y2 = min(img_height, y2 + bottom_padding)  # 아래쪽 패딩 적용
    new_x1 = max(0, x1 - padding_horizontal)
    new_x2 = min(img_width, x2 + padding_horizontal)
    
    # 새로운 크롭 영역으로 이미지 크롭
    padded_crop = orig_img.crop((new_x1, new_y1, new_x2, new_y2))
    
    # 크롭된 이미지 저장
    pokecolo_path = os.path.join(TEMP_DIR, "pokecolo.png")
    padded_crop.save(pokecolo_path)
    
    print(f"원본 바운딩 박스: ({x1}, {y1}, {x2}, {y2})")
    print(f"패딩 추가 바운딩 박스: ({new_x1}, {new_y1}, {new_x2}, {new_y2})")
    print(f"위쪽 패딩 {top_padding}px 추가됨 (원본 높이의 {top_padding_percent*100}%)")
    print(f"아래쪽 패딩 {bottom_padding}px 추가됨 (원본 높이의 {padding_percent*100}%)")
    print(f"좌우 패딩 {padding_horizontal}px 추가됨 (원본 너비의 {padding_percent*100}%)")
    
    return pokecolo_path, True

def process_image(input_image, confidence_threshold=0.5):
    """
    입력 이미지 처리 파이프라인
    
    Args:
        input_image: 입력 이미지 (파일 경로 또는 numpy 배열)
        confidence_threshold (float): 객체 감지 신뢰도 임계값
        
    Returns:
        tuple: (바운딩 박스가 있는 이미지, 각 카테고리별 결과 URL들, pokecolo 검출 여부, 총 처리 시간)
    """
    # 전체 예측 시간 측정 시작
    total_start_time = time.time()

    # 입력 이미지가 numpy 배열인 경우 파일로 저장
    if isinstance(input_image, np.ndarray):
        input_path = os.path.join(TEMP_DIR, "input.png")
        Image.fromarray(input_image).save(input_path)
    # 입력이미지가 Image 객체인 경우 파일로 저장
    elif isinstance(input_image, Image.Image):
        input_path = os.path.join(TEMP_DIR, "input.png")
        input_image.save(input_path)
    else:
        input_path = input_image
    
    # 원본 이미지 경로 저장 (나중에 참조용)
    original_image_path = input_path
    
    # pokecolo 검출 시도
    processed_image_path, pokecolo_detected = detect_pokecolo(input_path)
    
    # 결과 저장 경로
    result_path = os.path.join(TEMP_DIR, "result.png")
    
    # YOLO로 객체 감지 (pokecolo가 검출되었으면 크롭된 이미지, 아니면 원본 이미지 사용)
    results, crops, labels, boxes = detector.detect(
        processed_image_path, 
        conf_threshold=confidence_threshold,
        save_result=True,
        result_path=result_path
    )
    
    # 각 카테고리별 결과를 저장할 딕셔너리
    category_results = {category: [] for category in ALL_CATEGORIES}
    
    # 모든 세부 카테고리 결과를 위한 딕셔너리 추가
    all_subcategories = ONEPIECE_SUBCATEGORIES + HAIR_SUBCATEGORIES
    for subcategory in all_subcategories:
        category_results[subcategory] = []
    
    # 감지된 카테고리를 추적하기 위한 집합
    detected_categories = set()
    
    # 각 카테고리별로 가장 신뢰도가 높은 객체만 선택하기 위한 딕셔너리
    category_best_crops = {}
    category_best_confs = {}
    
    # 감지된 모든 객체들에 대해 카테고리별로 가장 신뢰도 높은 것 선택
    for crop, label, box in zip(crops, labels, boxes):
        # 신뢰도 추출 (boxes는 [x1, y1, x2, y2, confidence, class_id] 형태)
        confidence = box[4] if len(box) > 4 else 0.0
        
        # 해당 카테고리가 이미 있는지 확인하고 신뢰도 비교
        if label not in category_best_confs or confidence > category_best_confs[label]:
            category_best_crops[label] = crop
            category_best_confs[label] = confidence
    
    # 각 카테고리별로 신뢰도가 가장 높은 크롭에 대해서만 분류 수행
    prediction_results = []
    onepiece_crop = None
    hair_crop = None
    
    # 전체 결과를 위한 빈 결과 리스트 생성 (모든 감지된 객체 수 만큼)
    prediction_results = [None] * len(crops)
    
    # 각 카테고리별로 가장 신뢰도 높은 크롭에 대해 처리
    for label, crop in category_best_crops.items():
        # 감지된 카테고리 추가
        detected_categories.add(label)
        
        # 카테고리별 크롭 저장
        if label == "ONEPIECE":
            onepiece_crop = crop
        elif label == "HAIR":
            hair_crop = crop
        elif label == "SHOES":
            # SHOES 카테고리 추가
            shoes_crop = crop
        
        # 크롭 저장
        crop_path = save_crop(crop, label, TEMP_DIR)
        
        # 라벨에 따라 분류 수행
        if label in SUPPORTED_LABELS:
            print(f"{label} 분류기 실행 중...")
            class_names, probs = classifier.classify(crop, label, top_k=TOP_K)
            print(f"{label} 분류 완료: {class_names[0]} ({probs[0]:.2f})")
            
            # 해당 카테고리에 결과 저장
            for class_name, prob in zip(class_names, probs):
                # 이미지 URL 가져오기
                image_url = get_image_url_from_class(class_name, items_dict)
                if image_url:
                    category_results[label].append({
                        'class_name': class_name,
                        'probability': prob,
                        'image_url': image_url
                    })
    
    # HAIR가 감지되지 않은 경우 기본 크롭 생성
    if "HAIR" not in category_best_crops:
        print("HAIR 검출되지 않음.")
        hair_crop = create_default_crop(processed_image_path, "HAIR")
        if hair_crop is not None:
            # 크롭 저장
            crop_path = save_crop(hair_crop, "HAIR", TEMP_DIR)
            detected_categories.add("HAIR")
            
            # HAIR 분류 수행
            if "HAIR" in SUPPORTED_LABELS:
                try:
                    print("HAIR 분류기 실행 중... (기본 크롭)")
                    class_names, probs = classifier.classify(hair_crop, "HAIR", top_k=TOP_K)
                    print(f"HAIR 분류 완료 (기본 크롭): {class_names[0]} ({probs[0]:.2f})")
                    
                    # 결과 저장
                    for class_name, prob in zip(class_names, probs):
                        image_url = get_image_url_from_class(class_name, items_dict)
                        if image_url:
                            category_results["HAIR"].append({
                                'class_name': class_name,
                                'probability': prob,
                                'image_url': image_url
                            })
                except Exception as e:
                    print(f"HAIR 기본 크롭 분류 중 오류: {str(e)}")
    
    # ONEPIECE가 감지되지 않은 경우 기본 크롭 생성
    if "ONEPIECE" not in category_best_crops:
        print("ONEPIECE 검출되지 않음. 이미지 하단 1/2로 기본 크롭 생성")
        onepiece_crop = create_default_crop(processed_image_path, "ONEPIECE")
        if onepiece_crop is not None:
            # 크롭 저장
            crop_path = save_crop(onepiece_crop, "ONEPIECE", TEMP_DIR)
            detected_categories.add("ONEPIECE")
            
            # ONEPIECE 분류 수행
            if "ONEPIECE" in SUPPORTED_LABELS:
                try:
                    print("ONEPIECE 분류기 실행 중... (기본 크롭)")
                    class_names, probs = classifier.classify(onepiece_crop, "ONEPIECE", top_k=TOP_K)
                    print(f"ONEPIECE 분류 완료 (기본 크롭): {class_names[0]} ({probs[0]:.2f})")
                    
                    # 결과 저장
                    for class_name, prob in zip(class_names, probs):
                        image_url = get_image_url_from_class(class_name, items_dict)
                        if image_url:
                            category_results["ONEPIECE"].append({
                                'class_name': class_name,
                                'probability': prob,
                                'image_url': image_url
                            })
                except Exception as e:
                    print(f"ONEPIECE 기본 크롭 분류 중 오류: {str(e)}")
    
    # SHOES가 감지되지 않은 경우 기본 크롭 생성
    if "SHOES" not in category_best_crops:
        print("SHOES 검출되지 않음. 이미지 하단 1/3로 기본 크롭 생성")
        shoes_crop = create_default_crop(processed_image_path, "SHOES")
        if shoes_crop is not None:
            # 크롭 저장
            crop_path = save_crop(shoes_crop, "SHOES", TEMP_DIR)
            detected_categories.add("SHOES")
            
            # SHOES 분류 수행
            if "SHOES" in SUPPORTED_LABELS:
                try:
                    print("SHOES 분류기 실행 중... (기본 크롭)")
                    class_names, probs = classifier.classify(shoes_crop, "SHOES", top_k=TOP_K)
                    print(f"SHOES 분류 완료 (기본 크롭): {class_names[0]} ({probs[0]:.2f})")
                    
                    # 결과 저장
                    for class_name, prob in zip(class_names, probs):
                        image_url = get_image_url_from_class(class_name, items_dict)
                        if image_url:
                            category_results["SHOES"].append({
                                'class_name': class_name,
                                'probability': prob,
                                'image_url': image_url
                            })
                except Exception as e:
                    print(f"SHOES 기본 크롭 분류 중 오류: {str(e)}")
    
    # Onepiece에 대해 추가 세부 분류 수행 (신뢰도 가장 높은 것만 처리)
    if onepiece_crop is not None:
        # 각 세부 카테고리에 대해 분류 실행
        for subcategory in ONEPIECE_SUBCATEGORIES:
            if subcategory in SUPPORTED_LABELS:
                try:
                    print(f"{subcategory} 분류기 실행 중... (ONEPIECE 세부 분류)")
                    sub_class_names, sub_probs = classifier.classify(onepiece_crop, subcategory, top_k=TOP_K)
                    print(f"{subcategory} 분류 완료: {sub_class_names[0]} ({sub_probs[0]:.2f})")
                    
                    # 세부 카테고리에 결과 저장
                    for class_name, prob in zip(sub_class_names, sub_probs):
                        image_url = get_image_url_from_class(class_name, items_dict)
                        if image_url:
                            category_results[subcategory].append({
                                'class_name': class_name,
                                'probability': prob,
                                'image_url': image_url
                            })
                    
                    # 세부 카테고리 감지 추가
                    detected_categories.add(subcategory)
                except Exception as e:
                    print(f"{subcategory} 분류 중 오류: {str(e)}")
    
    # HAIR에 대해 ACCESSORY_HEAD 분류 수행 (신뢰도 가장 높은 것만 처리)
    if hair_crop is not None:
        # ACCESSORY_HEAD 분류 실행
        for subcategory in HAIR_SUBCATEGORIES:
            if subcategory in SUPPORTED_LABELS:
                try:
                    print(f"{subcategory} 분류기 실행 중... (HAIR 세부 분류)")
                    sub_class_names, sub_probs = classifier.classify(create_default_crop(processed_image_path, "HEAD"), subcategory, top_k=TOP_K)
                    print(f"{subcategory} 분류 완료: {sub_class_names[0]} ({sub_probs[0]:.2f})")
                    
                    # 세부 카테고리에 결과 저장
                    for class_name, prob in zip(sub_class_names, sub_probs):
                        image_url = get_image_url_from_class(class_name, items_dict)
                        if image_url:
                            category_results[subcategory].append({
                                'class_name': class_name,
                                'probability': prob,
                                'image_url': image_url
                            })
                    
                    # 세부 카테고리 감지 추가
                    detected_categories.add(subcategory)
                except Exception as e:
                    print(f"{subcategory} 분류 중 오류: {str(e)}")
    
    # 눈 영역 처리 (EYE 카테고리가 필요한 경우)
    # 원본 이미지에서 눈 영역을 처리하므로 신뢰도 기준으로 처리할 필요 없음
    try:
        eye_image = eye_detector.run(processed_image_path)
        eye_crop_path = save_crop(eye_image, "EYE", TEMP_DIR)
        
        # 눈이 감지되었으면 카테고리 추가
        detected_categories.add("EYE")
        
        # 눈 분류
        if "EYE" in SUPPORTED_LABELS:
            print("EYE 분류기 실행 중...")
            eye_classes, eye_probs = classifier.classify(eye_image, "EYE", top_k=TOP_K)
            print(f"EYE 분류 완료: {eye_classes[0]} ({eye_probs[0]:.2f})")
            
            # EYE 카테고리에 결과 저장
            for class_name, prob in zip(eye_classes, eye_probs):
                image_url = get_image_url_from_class(class_name, items_dict)
                if image_url:
                    category_results["EYE"].append({
                        'class_name': class_name,
                        'probability': prob,
                        'image_url': image_url
                    })
    except Exception as e:
        print(f"눈 영역 처리 중 오류: {str(e)}")
    
    # 카테고리별 신뢰도 높은 객체에 대한 결과만 표시하기 위해 prediction_results 갱신
    updated_prediction_results = []
    for i, (crop, label, box) in enumerate(zip(crops, labels, boxes)):
        # 신뢰도 추출
        confidence = box[4] if len(box) > 4 else 0.0
        
        # 해당 객체가 해당 카테고리에서 가장 신뢰도가 높은 것인지 확인
        if category_best_confs.get(label, 0.0) == confidence:
            # 라벨에 따라 분류 결과 가져오기
            if label in SUPPORTED_LABELS:
                print(f"{label} 분류기 실행 중... (결과 업데이트)")
                class_names, probs = classifier.classify(crop, label, top_k=TOP_K)
                print(f"{label} 분류 완료 (결과 업데이트): {class_names[0]} ({probs[0]:.2f})")
                updated_prediction_results.append(list(zip(class_names, probs)))
            else:
                updated_prediction_results.append(None)
        else:
            # 해당 카테고리에서 가장 신뢰도가 높은 것이 아닌 경우 None 추가
            updated_prediction_results.append(None)

    # 전체 예측 시간 측정 종료
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"전체 처리 시간: {total_elapsed_time:.2f}초")
    print("결과 처리 완료.")
    # 바운딩 박스와 예측 결과가 있는 이미지 생성
    result_image = draw_detection_results(
        processed_image_path, 
        boxes, 
        labels, 
        predictions=updated_prediction_results,
        output_path=result_path
    )
    print("결과 이미지 생성 완료.")

    # 카테고리별 상위 이미지 URL 가져오기
    category_urls = {}
    
    # 메인 카테고리와 모든 세부 카테고리 처리
    all_categories = ALL_CATEGORIES + ONEPIECE_SUBCATEGORIES + HAIR_SUBCATEGORIES
    
    for category in all_categories:
        # 카테고리가 감지되지 않았다면 빈 리스트 할당
        if category not in detected_categories:
            category_urls[category] = []
            continue
            
        # 각 카테고리별 상위 결과
        top_results = sorted(
            category_results[category], 
            key=lambda x: x['probability'], 
            reverse=True
        )[:TOP_K]
        
        # 만약 결과가 없다면 해당 카테고리는 빈 리스트로 설정
        if not top_results:
            category_urls[category] = []
            continue
            
        # URL 리스트 저장
        urls = []
        for result in top_results:
            if 'image_url' in result and result['image_url']:
                urls.append({
                    'url': result['image_url'],
                    'class_name': result['class_name'],
                    'probability': result['probability']
                })
        
        category_urls[category] = urls
    
    print("카테고리별 결과 URL 생성 완료.")
    
    # result_image, category_urls, pokecolo 검출 정보, 총 처리 시간 반환
    return result_image, category_urls, pokecolo_detected, total_elapsed_time



# create_app 함수 수정
def create_app():
    """Gradio 앱 생성"""
    with gr.Blocks(title="패션 아이템 분석기") as app:
        gr.Markdown("# 패션 아이템 분석 데모")
        gr.Markdown("이미지를 업로드하면 패션 아이템을 감지하고 분류합니다.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 입력 컴포넌트
                input_image = gr.Image(type="pil", label="입력 이미지")
                confidence = gr.Slider(
                    minimum=0.1, 
                    maximum=1.0, 
                    value=0.1, 
                    step=0.05, 
                    label="감지 신뢰도 임계값"
                )
                submit_btn = gr.Button("분석 시작", variant="primary")
            
            with gr.Column(scale=1):
                # 감지 결과 이미지
                output_image = gr.Image(type="pil", label="감지 결과")
                # pokecolo 검출 여부 표시
                pokecolo_detected = gr.Textbox(label="Pokecolo 검출 상태", interactive=False)
        
        # 모든 카테고리에 대한 탭 (메인 카테고리 + 모든 세부 카테고리)
        all_categories = ALL_CATEGORIES + ONEPIECE_SUBCATEGORIES + HAIR_SUBCATEGORIES
        
        # process_and_display 함수 수정
        def process_and_display(input_img, conf):
            result_img, category_urls, is_pokecolo_detected, total_elapsed_time = process_image(input_img, conf)
            
            # pokecolo 검출 상태 메시지
            pokecolo_status = f"✅ Pokecolo가 검출되어 처리되었습니다. Total Time: {total_elapsed_time:.2f}초" if is_pokecolo_detected else f"❌ Pokecolo가 검출되지 않았습니다. 원본 이미지로 처리하였습니다. Total Time: {total_elapsed_time:.2f}초"
            
            # 모든 카테고리에 대한 갤러리 이미지 리스트 준비
            gallery_outputs = []
            
            for category in all_categories:
                # 카테고리 결과 URL이 있는 경우 갤러리용 이미지 정보 생성
                category_results = category_urls.get(category, [])
                gallery_items = []
                
                for item in category_results:
                    if 'url' in item and item['url']:
                        # URL 문자열만 전달하고 캡션은 별도로 설정하지 않음
                        gallery_items.append(item['url'])
                
                gallery_outputs.append(gallery_items)
            
            # 결과 이미지와 pokecolo 상태 포함
            return [result_img, pokecolo_status] + gallery_outputs
    
        # 각 카테고리별 결과 탭
        with gr.Tabs() as tabs:
            for category in all_categories:
                with gr.TabItem(category):
                    with gr.Row():
                        # 각 카테고리별 상위 결과 이미지 URL을 표시할 갤러리
                        result_gallery = gr.Gallery(
                            label=f"{category} 결과",
                            columns=5,
                            object_fit="contain",
                            height="auto"
                        )
        
        # 입력 및 출력 연결
        output_components = [output_image, pokecolo_detected]
        for tab in tabs.children:
            output_components.append(tab.children[0].children[0])  # 각 탭의 갤러리 컴포넌트
        
        submit_btn.click(
            process_and_display,
            inputs=[input_image, confidence],
            outputs=output_components
        )
    
    return app

if __name__ == "__main__":
    # Gradio 앱 생성 및 실행
    app = create_app()
    app.launch(share=False, server_name="127.0.0.1", server_port=7860, debug=True)