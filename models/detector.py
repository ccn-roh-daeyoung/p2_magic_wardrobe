"""
YOLO 객체 감지 모델을 관리하는 모듈
"""

import os
from ultralytics import YOLO
import torch
from PIL import Image

class FashionDetector:
    """
    YOLO 모델을 사용하여 패션 아이템을 감지하는 클래스
    """
    
    def __init__(self, model_path):
        """
        YOLO 모델 초기화
        
        Args:
            model_path (str): YOLO 모델 파일 경로
        """
        self.model = YOLO(model_path)
        print(f"YOLO 모델 로드 완료: {model_path}")
    
    def detect(self, image_path, conf_threshold=0.0, save_result=False, result_path=None):
        """
        이미지에서 패션 아이템 감지
        
        Args:
            image_path (str): 분석할 이미지 경로
            conf_threshold (float): 신뢰도 임계값
            save_result (bool): 결과 이미지 저장 여부
            result_path (str, optional): 결과 이미지 저장 경로
            
        Returns:
            tuple: (결과 객체, 크롭 리스트, 라벨 리스트, 바운딩 박스 리스트)
        """
        # YOLO 추론 실행
        results = self.model(image_path)
        
        # 결과 저장 (요청된 경우)
        if save_result:
            save_path = result_path or os.path.join(os.path.dirname(image_path), "result.jpg")
            for result in results:
                result.save(filename=save_path)
            print(f"감지 결과 저장: {save_path}")
        
        # 크롭, 라벨, 박스 정보 추출
        crops, labels, boxes = self._extract_crops(image_path, results, conf_threshold)
        
        return results, crops, labels, boxes
    
    def _extract_crops(self, image_path, results, conf_threshold=0.5):
        """
        감지된 객체에서 크롭 이미지 추출
        
        Args:
            image_path (str): 원본 이미지 경로
            results: YOLO 감지 결과
            conf_threshold (float): 신뢰도 임계값
            
        Returns:
            tuple: (크롭 리스트, 라벨 리스트, 바운딩 박스 리스트)
        """
        # 원본 이미지 열기
        original_image = Image.open(image_path)
        crops = []
        labels = []
        boxes = []
        
        for result in results:
            detection_boxes = result.boxes
            
            for i, box in enumerate(detection_boxes):
                # 좌표 얻기
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 클래스 라벨 얻기
                cls = int(box.cls[0])
                cls_name = result.names[cls]
                
                # 신뢰도 점수 얻기
                conf = float(box.conf[0])
                
                # 신뢰도가 임계값보다 높은 경우에만 처리
                if conf > conf_threshold:
                    # 크롭 추출
                    crop = original_image.crop((x1, y1, x2, y2))
                    crops.append(crop)
                    labels.append(cls_name)
                    boxes.append((x1, y1, x2, y2, conf))
                    print(f"감지됨: {cls_name} (신뢰도: {conf:.2f}) 위치: ({x1}, {y1}, {x2}, {y2})")
        
        return crops, labels, boxes