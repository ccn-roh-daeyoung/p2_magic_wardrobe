"""
이미지 처리 및 유틸리티 함수 모듈
"""

import os
import uuid
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def save_crop(crop, label, output_dir):
    """
    크롭 이미지 저장
    
    Args:
        crop (PIL.Image): 크롭 이미지
        label (str): 라벨 이름
        output_dir (str): 출력 디렉토리
    
    Returns:
        str: 저장된 파일 경로
    """
    # 출력 디렉토리 확인
    os.makedirs(output_dir, exist_ok=True)
    
    # 고유한 파일 이름 생성
    filename = f"crop_{label.lower()}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    # RGBA인 경우 RGB로 변환
    if crop.mode == 'RGBA':
        crop = crop.convert('RGB')
    
    # 이미지 저장
    crop.save(filepath)
    print(f"크롭 이미지 저장: {filepath}")
    
    return filepath

def draw_detection_results(image_path, boxes, labels, predictions=None, output_path=None):
    """
    원본 이미지에 감지 결과와 예측 결과 시각화
    
    Args:
        image_path (str): 원본 이미지 경로
        boxes (list): 바운딩 박스 좌표 목록 [(x1, y1, x2, y2, conf), ...]
        labels (list): 라벨 목록
        predictions (list, optional): 예측 결과 목록 [(클래스, 확률), ...]
        output_path (str, optional): 결과 이미지 저장 경로
    
    Returns:
        PIL.Image: 결과 이미지
    """
    # 원본 이미지 로드
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # 바운딩 박스 색상 매핑
    colors = {
        'ONEPIECE': (255, 0, 0),     # 빨강
        'HAIR': (0, 255, 0),         # 초록
        'SHOES': (0, 0, 255),        # 파랑
        'EYE': (255, 255, 0),        # 노랑
        'default': (255, 165, 0)     # 주황 (기본)
    }
    
    # 각 박스 그리기
    for i, ((x1, y1, x2, y2, conf), label) in enumerate(zip(boxes, labels)):
        # 라벨에 따른 색상 선택
        color = colors.get(label, colors['default'])
        
        # 박스 그리기
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 라벨 텍스트
        label_text = f"{label} ({conf:.2f})"
        
        # 텍스트 크기 계산 (font 인자 없이)
        left, top, right, bottom = draw.textbbox((x1, y1), label_text)
        text_w = right - left
        text_h = bottom - top
        
        # 텍스트 배경 그리기
        draw.rectangle([x1, y1-text_h-4, x1+text_w+2, y1], fill=color)
        
        # 텍스트 그리기 (font 인자 없이)
        draw.text((x1, y1-text_h-4), label_text, fill="white")
        
        # 예측 결과가 있는 경우 추가 정보 표시
        if predictions and i < len(predictions) and predictions[i]:
            top_class, top_prob = predictions[i][0]
            pred_text = f"Pred: {top_class} ({top_prob:.2f})"
            
            # 예측 텍스트의 배경 영역 (textbbox() 사용)
            left, top, right, bottom = draw.textbbox((x1, y2), pred_text)
            pred_w = right - left
            pred_h = bottom - top
            
            # 배경 영역 그리기
            draw.rectangle([x1, y2, x1+pred_w+2, y2+pred_h+4], fill=color)
            
            # 예측 텍스트 그리기
            draw.text((x1+1, y2+2), pred_text, fill=(255, 255, 255))
    
    # 결과 저장
    if output_path:
        image.save(output_path)
        print(f"결과 이미지 저장: {output_path}")
    
    return image

def load_items_metadata(metadata_path):
    """
    아이템 메타데이터 로드
    
    Args:
        metadata_path (str): 메타데이터 파일 경로
    
    Returns:
        dict: 아이템 ID별 메타데이터 딕셔너리
    """
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            items_data = json.load(f)
        
        # 아이템 ID를 키로 하는 딕셔너리 생성
        items_dict = {}
        if isinstance(items_data, list):
            for item in items_data:
                items_dict[item['item_id']] = item
        else:
            # items_data가 딕셔너리인 경우
            items_dict = items_data.get('items', items_data)
        
        print(f"메타데이터 로드 완료: {len(items_dict)} 아이템")
        return items_dict
    
    except Exception as e:
        print(f"메타데이터 로드 중 오류: {str(e)}")
        return {}

def get_image_url_from_class(class_name, items_dict):
    """
    클래스 이름으로 이미지 URL 가져오기
    
    Args:
        class_name (str): 클래스 이름
        items_dict (dict): 아이템 메타데이터 딕셔너리
    
    Returns:
        str: 이미지 URL 또는 None
    """
    if class_name in items_dict:
        return items_dict[class_name].get('item_image_url')
    return None