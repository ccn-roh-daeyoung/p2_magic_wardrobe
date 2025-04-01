"""
설정 및 경로 정보를 관리하는 모듈
"""

import os

# 기본 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# YOLO 모델 경로
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolo/best_hos.pt")
C0ORD_YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolo/best_coord.pt")
EYE_YOLONECK_MODEL_PATH = os.path.join(MODEL_DIR, "yolo/eye.onnx")

# 분류기 모델 경로
CLASSIFIER_PATHS = {
    "ONEPIECE": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_onepiece/efficientnet_b0_best.pth"),
    "HAIR": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_hair/efficientnet_b0_best.pth"),
    "SHOES": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_shoes/efficientnet_b0_best.pth"),
    "EYE": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_eyes/efficientnet_b0_best.pth"),
    "OUTER": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_outer/efficientnet_b0_best.pth"),
    "TOP": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_top/efficientnet_b0_best.pth"),
    "BOTTOM": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_bottoms/efficientnet_b0_best.pth"),
    "ACCESSORY_HEAD": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_accesory_head/efficientnet_b0_best.pth"),
    "ACCESSORY_BODY": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_accesory_body/efficientnet_b0_best.pth"),
    "ACCESSORY_RIGHT_LEFT": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_accesory_right_left/efficientnet_b0_best.pth"),
    "ACCESSORY_BACK": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_accesory_back/efficientnet_b0_best.pth"),
}

# 클래스 매핑 경로
CLASS_MAPPING_PATHS = {
    "ONEPIECE": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_onepiece/class_mapping.json"),
    "HAIR": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_hair/class_mapping.json"),
    "SHOES": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_shoes/class_mapping.json"),
    "EYE": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_eyes/class_mapping.json"),
    "OUTER": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_outer/class_mapping.json"),
    "TOP": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_top/class_mapping.json"),
    "BOTTOM": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_bottoms/class_mapping.json"),
    "ACCESSORY_HEAD": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_accesory_head/class_mapping.json"),
    "ACCESSORY_BODY": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_accesory_body/class_mapping.json"),
    "ACCESSORY_RIGHT_LEFT": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_accesory_right_left/class_mapping.json"),
    "ACCESSORY_BACK": os.path.join(MODEL_DIR, "classifier/efficientnet_b0_accesory_back/class_mapping.json"),
}

# 항목 메타데이터 파일 경로
ITEMS_METADATA_PATH = os.path.join(BASE_DIR, "data/latest_items.json")

# 임시 파일 저장 디렉토리
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# 지원하는 라벨 목록
SUPPORTED_LABELS = ["ONEPIECE", "HAIR", "SHOES", "EYE", "OUTER", "TOP", "BOTTOM", "ACCESSORY_HEAD", "ACCESSORY_BODY", "ACCESSORY_RIGHT_LEFT", "ACCESSORY_BACK"]