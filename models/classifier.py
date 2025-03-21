"""
분류 모델을 관리하는 모듈
"""

import os
import torch
import timm
import json
from PIL import Image
from torchvision import transforms

class FashionClassifier:
    """
    패션 아이템 분류기 클래스
    """
    
    def __init__(self):
        """
        분류기 초기화
        """
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.models = {}
        self.class_names = {}
        self.class_mappings = {}
        
        # 이미지 전처리 변환
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, label_type, model_path, mapping_path=None):
        """
        특정 라벨 타입의 분류 모델 로드
        
        Args:
            label_type (str): 라벨 타입 (예: "ONEPIECE", "HAIR")
            model_path (str): 모델 파일 경로
            mapping_path (str, optional): 클래스 매핑 파일 경로
        """
        try:
            # 모델 로드
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 체크포인트에서 정보 추출
            model_state_dict = checkpoint['model_state_dict']
            self.class_names[label_type] = checkpoint.get('class_names', [])
            num_classes = len(self.class_names[label_type])
            
            # 모델 이름은 기본적으로 efficientnet_b0 사용
            model_name = 'efficientnet_b0'
            
            # 모델 생성 및 가중치 로드
            model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
            model.load_state_dict(model_state_dict)
            model = model.to(self.device)
            model.eval()
            
            self.models[label_type] = model
            print(f"모델 '{model_name}'을 라벨 '{label_type}'에 로드했습니다. 클래스 수: {num_classes}")
            
            # 클래스 매핑 로드 (제공된 경우)
            if mapping_path and os.path.exists(mapping_path):
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    self.class_mappings[label_type] = json.load(f)
                print(f"클래스 매핑을 로드했습니다: {mapping_path}")
            
            return True
            
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {str(e)}")
            return False
    
    def preprocess_image(self, image):
        """
        이미지를 모델 입력에 맞게 전처리
        
        Args:
            image (PIL.Image): 입력 이미지
            
        Returns:
            tuple: (이미지 텐서, 원본 이미지)
        """
        # RGB 모드 확인
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 이미지 변환
        image_tensor = self.transform(image).unsqueeze(0)  # 배치 차원 추가
        
        return image_tensor, image
    
    def classify(self, image, label_type, top_k=5):
        """
        이미지 분류
        
        Args:
            image (PIL.Image): 분류할 이미지
            label_type (str): 라벨 타입 (예: "ONEPIECE", "HAIR")
            top_k (int): 반환할 상위 예측 수
            
        Returns:
            tuple: (클래스 이름 리스트, 확률 리스트)
        """
        if label_type not in self.models:
            print(f"라벨 '{label_type}'에 대한 모델이 로드되지 않았습니다.")
            return [], []
        
        # 이미지 전처리
        image_tensor, _ = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # 추론 실행
        with torch.no_grad():
            outputs = self.models[label_type](image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # 상위 K개 예측 가져오기
        if len(self.class_names[label_type]) < top_k:
            top_k = len(self.class_names[label_type])
            
        probs, indices = torch.topk(probabilities, top_k)
        probs = probs.squeeze().cpu().numpy()
        indices = indices.squeeze().cpu().numpy()
        
        # 클래스 이름 가져오기
        class_names = []
        
        for idx in indices:
            if label_type in self.class_mappings and 'idx_to_class' in self.class_mappings[label_type]:
                class_name = self.class_mappings[label_type]['idx_to_class'].get(str(idx), f"클래스 {idx}")
            else:
                class_name = self.class_names[label_type][idx] if idx < len(self.class_names[label_type]) else f"클래스 {idx}"
            
            class_names.append(class_name)
        
        return class_names, probs