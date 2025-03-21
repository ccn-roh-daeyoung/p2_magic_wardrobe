import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2


class EyeDetectionProcess:
    
    def __init__(self, model_path) -> None:
        self.eye_model = ort.InferenceSession(model_path)
    
    def run(self, image_path):
        self.image_path = image_path
        return self.detect_eye()

    def detect_eye(self):
        # image = self.image
        # 이미지 불러오기
        if isinstance(self.image_path, Image.Image):
            image = self.image_path
        else:
            image = Image.open(self.image_path).convert('RGB')  # RGB 모드로 명시적 변환

        # 눈 탐지
        detections = self.get_boxes(image)
        
        # 검출된 눈 영역 자르기
        if len(detections) > 0:
            # 첫 번째 검출된 눈만 사용
            detection = detections[0]
            x1, y1, w, h = map(int, detection[:4])
            x2 = x1 + w
            y2 = y1 + h
            # 이미지에서 해당 부분 자르기
            cropped_image = image.crop((x1, y1, x2, y2))
        else:
            # 탐지 실패 시 검은색 이미지 생성
            cropped_image = Image.new(mode='RGB', color=(0, 0, 0), size=(50, 50))
        
        return cropped_image
    
    def get_boxes(self, original_image):
        input_shape = (640, 640)
        ort_session = self.eye_model
        
        # 이미지 전처리
        input_image = self.preprocess_image(original_image, input_shape)
        
        # ONNX 모델 실행
        outputs = ort_session.run(None, {'images': input_image})
        outputs = outputs[0].transpose((0, 2, 1))
        
        class_ids, confs, boxes = list(), list(), list()

        image_width, image_height = original_image.size

        x_factor = image_width / input_shape[0]
        y_factor = image_height / input_shape[1]

        rows = outputs[0].shape[0]

        # 결과 처리
        for i in range(rows):
            row = outputs[0][i]
            conf = row[4]
            
            classes_score = row[4:]
            _, _, _, max_idx = cv2.minMaxLoc(classes_score)
            class_id = max_idx[1]
            
            if (classes_score[class_id] > .1):
                confs.append(conf)
                label = int(class_id)
                class_ids.append(label)
                
                # 박스 추출
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)
        
        # NMS (Non-Maximum Suppression) 적용
        r_boxes = []
        if boxes:
            indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.1, 0.45) 
            for i in indexes:
                r_boxes.append(boxes[i])
        
        return r_boxes

    def preprocess_image(self, image, input_size=(640, 640)):
        image = image.convert('RGB')  # 이미지 RGB로 변환
        image = image.resize(input_size)  # 모델 입력 크기로 리사이즈
        image = np.array(image, dtype=np.float32) / 255.0  # 정규화 (0~1 범위)
        image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        image = np.expand_dims(image, axis=0)  # 배치 차원 추가
        return image

# 사용 예시
if __name__ == "__main__":
    # 이미지 경로 또는 PIL 이미지
    image_path = "/home/daeyoung_roh/workspace/p2_proj/data/eye.jpg"
    
    # 눈 검출 처리
    eye_detector = EyeDetectionProcess(image_path)
    eye_image = eye_detector.run()
    
    # 결과 저장 또는 표시
    if eye_image.mode == 'RGBA':
        # RGBA 모드인 경우 RGB로 변환 (알파 채널 제거)
        eye_image = eye_image.convert('RGB')
    eye_image.save("detected_eye.jpg")
    # 또는 이미지 출력
    # eye_image.show()
