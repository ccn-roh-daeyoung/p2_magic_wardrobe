import os
import random
from PIL import Image, ImageDraw
import numpy as np

# 디렉토리 생성 함수
def create_directories(base_dir):
    # 디렉토리가 없으면 생성
    os.makedirs(base_dir, exist_ok=True)
    
    # images와 labels 디렉토리 생성 (YOLO 형식)
    train_img_dir = os.path.join(base_dir, 'images', 'train')
    train_label_dir = os.path.join(base_dir, 'labels', 'train')
    val_img_dir = os.path.join(base_dir, 'images', 'val')
    val_label_dir = os.path.join(base_dir, 'labels', 'val')
    
    # 디버그 디렉토리 생성 (바운딩 박스가 그려진 이미지용)
    debug_dir = os.path.join(base_dir, 'debug')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    return train_img_dir, train_label_dir, val_img_dir, val_label_dir, debug_dir

# 배경 이미지 목록 가져오기
def get_background_images(bg_dir):
    bg_images = []
    for filename in os.listdir(bg_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            bg_images.append(os.path.join(bg_dir, filename))
    return bg_images

# RGBA 이미지에서 투명하지 않은 부분의 바운딩 박스 계산 (패딩 추가)
def calculate_bounding_box(rgba_image, padding_percent=0.1):
    # 알파 채널 추출
    alpha = np.array(rgba_image.split()[3])
    
    # 투명하지 않은 픽셀 위치 찾기
    non_transparent = np.where(alpha > 0)
    
    if len(non_transparent[0]) == 0:  # 투명한 이미지라면
        return None
    
    # 바운딩 박스 좌표 계산
    min_y, max_y = np.min(non_transparent[0]), np.max(non_transparent[0])
    min_x, max_x = np.min(non_transparent[1]), np.max(non_transparent[1])
    
    # 이미지 크기
    height, width = alpha.shape
    
    # 패딩 계산
    padding_x = int((max_x - min_x) * padding_percent)
    padding_y = int((max_y - min_y) * padding_percent)
    
    # 패딩 적용 (경계 체크)
    min_x = max(0, min_x - padding_x)
    min_y = max(0, min_y - padding_y)
    max_x = min(width - 1, max_x + padding_x)
    max_y = min(height - 1, max_y + padding_y)
    
    # YOLO 형식으로 변환: class_id x_center y_center width height (모두 0~1 사이 정규화)
    x_center = (min_x + max_x) / 2 / width
    y_center = (min_y + max_y) / 2 / height
    bbox_width = (max_x - min_x) / width
    bbox_height = (max_y - min_y) / height
    
    return x_center, y_center, bbox_width, bbox_height

# 이미지 합성 및 YOLO 데이터셋 생성 함수
def create_composite_image(item_image, background_image, class_id=0, padding_percent=0.1):
    # 배경 이미지 로드
    bg_img = Image.open(background_image).convert("RGB")
    
    # 아이템 이미지의 크기
    item_width, item_height = item_image.size
    
    # 아이템 이미지 크기의 비율을 계산 (배경 이미지 대비 30~50% 크기로 조정)
    scale_ratio = random.uniform(0.1, 0.4)
    new_item_width = int(bg_img.width * scale_ratio)
    new_item_height = int(item_height * (new_item_width / item_width))
    
    # 리사이징된 아이템이 배경보다 크면 크기 조정
    if new_item_width >= bg_img.width or new_item_height >= bg_img.height:
        # 배경 이미지 대비 최대 70%까지만 허용
        scale_factor = min(bg_img.width / (new_item_width * 1.5), bg_img.height / (new_item_height * 1.5))
        new_item_width = int(new_item_width * scale_factor)
        new_item_height = int(new_item_height * scale_factor)
    
    # 아이템 이미지 리사이징
    resized_item = item_image.resize((new_item_width, new_item_height), Image.LANCZOS)
    
    # 배경 이미지 내에 랜덤한 위치 선택 (이미지가 배경을 벗어나지 않도록)
    # 에러 방지를 위해 범위가 유효한지 확인
    x_range = max(0, bg_img.width - new_item_width)
    y_range = max(0, bg_img.height - new_item_height)
    
    if x_range == 0:
        x_pos = 0
    else:
        x_pos = random.randint(0, x_range)
        
    if y_range == 0:
        y_pos = 0
    else:
        y_pos = random.randint(0, y_range)
    
    # 합성 이미지 생성 - 배경 위에 아이템 이미지 붙이기
    composite_img = bg_img.copy()  # 배경 이미지의 복사본 생성
    
    # RGBA 이미지를 배경 위에 알파 합성
    if resized_item.mode == 'RGBA':
        composite_img.paste(resized_item, (x_pos, y_pos), resized_item.split()[3])
    else:
        composite_img.paste(resized_item, (x_pos, y_pos))
    
    # 바운딩 박스 계산 (YOLO 형식: 중심 x, 중심 y, 너비, 높이 - 모두 0~1 정규화)
    # 패딩 추가
    box_width = new_item_width * (1 + padding_percent)
    box_height = new_item_height * (1 + padding_percent)
    
    # 패딩 적용 시 이미지 경계를 벗어나지 않도록 조정
    box_width = min(box_width, bg_img.width - x_pos)
    box_height = min(box_height, bg_img.height - y_pos)
    
    # 바운딩 박스 중심점
    center_x = (x_pos + new_item_width / 2) / bg_img.width
    center_y = (y_pos + new_item_height / 2) / bg_img.height
    
    # 바운딩 박스 크기 (정규화)
    norm_width = box_width / bg_img.width
    norm_height = box_height / bg_img.height
    
    # YOLO 형식: class_id x_center y_center width height
    bbox_label = f"{class_id} {center_x} {center_y} {norm_width} {norm_height}"
    
    return composite_img, bbox_label

# 디버그 이미지에 바운딩 박스 그리기 함수
def draw_bounding_box_on_image(image, bbox_label):
    # 이미지 복사본 생성
    debug_img = image.copy()
    draw = ImageDraw.Draw(debug_img)
    
    # YOLO 형식 바운딩 박스 파싱 (class_id x_center y_center width height)
    parts = bbox_label.split()
    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])
    
    # 이미지 크기
    img_width, img_height = debug_img.size
    
    # YOLO 형식에서 픽셀 좌표로 변환
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    
    # 바운딩 박스 그리기 (빨간색)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    
    # 클래스 이름 추가 (폰트 크기 키워서 더 잘 보이게)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", 20)
        draw.text((x1, y1 - 25), "pokecolo", fill="red", font=font)
    except:
        # 폰트가 없으면 기본 폰트 사용
        draw.text((x1, y1 - 25), "pokecolo", fill="red")
    
    return debug_img

# 메인 함수
def main():
    # 경로 설정
    item_dir = 'data/p1_data'  # RGBA 이미지가 있는 디렉토리
    bg_dir = 'data/theme_sheet'  # 배경 이미지가 있는 디렉토리
    output_dir = 'data/coord_yolo_dataset'  # 출력 디렉토리
    
    # 훈련/검증 비율
    train_ratio = 0.8
    
    # 각 아이템당 생성할 이미지 수
    images_per_item = 5
    
    # 패딩 비율 설정 (원본 객체 크기의 10%)
    padding_percent = 0.1
    
    # 출력 디렉토리 생성
    train_img_dir, train_label_dir, val_img_dir, val_label_dir, debug_dir = create_directories(output_dir)
    
    # 배경 이미지 목록 가져오기
    background_paths = get_background_images(bg_dir)
    if not background_paths:
        print("사용 가능한 배경 이미지가 없습니다.")
        return
    
    # 아이템 이미지 목록 가져오기
    item_images = []
    for filename in os.listdir(item_dir):
        if filename.lower().endswith(('.png')):  # RGBA 이미지는 주로 PNG
            item_images.append(os.path.join(item_dir, filename))
    
    if not item_images:
        print("사용 가능한 아이템 이미지가 없습니다.")
        return
    
    print(f"총 {len(item_images)}개의 아이템 이미지와 {len(background_paths)}개의 배경 이미지를 처리합니다.")
    
    # 각 아이템 이미지에 대해 증강 작업 수행
    for idx, item_path in enumerate(item_images):
        try:
            item_name = os.path.basename(item_path).split('.')[0]
            print(f"아이템 처리 중: {item_name} ({idx+1}/{len(item_images)})")
            
            # 이미지 로드
            item_image = Image.open(item_path)
            
            # RGBA 형식이 아니면 변환
            if item_image.mode != 'RGBA':
                item_image = item_image.convert('RGBA')
            
            # 각 아이템에 대한 합성 이미지 저장 데이터
            composite_data = []
            
            # 랜덤하게 배경 선택
            if len(background_paths) < images_per_item:
                selected_backgrounds = random.choices(background_paths, k=images_per_item)
            else:
                selected_backgrounds = random.sample(background_paths, images_per_item)
            
            # 각 배경에 대해 합성 이미지 생성
            for bg_path in selected_backgrounds:
                composite_img, bbox_label = create_composite_image(
                    item_image, 
                    bg_path, 
                    class_id=0,  # 클래스 ID (여러 클래스가 있다면 변경)
                    padding_percent=padding_percent
                )
                
                if composite_img and bbox_label:
                    composite_data.append((composite_img, bbox_label))
            
            if not composite_data:
                print(f"아이템 {item_name}에 대한 합성 이미지가 생성되지 않았습니다.")
                continue
            
            # 훈련/검증 분할
            split_idx = int(len(composite_data) * train_ratio)
            train_data = composite_data[:split_idx]
            val_data = composite_data[split_idx:]
            
            # 훈련 데이터 저장
            for i, (img, label) in enumerate(train_data):
                img_filename = f"{item_name}_{i}.jpg"
                label_filename = f"{item_name}_{i}.txt"
                
                img_path = os.path.join(train_img_dir, img_filename)
                label_path = os.path.join(train_label_dir, label_filename)
                debug_path = os.path.join(debug_dir, f"debug_{item_name}_{i}.jpg")
                
                img.save(img_path)
                with open(label_path, 'w') as f:
                    f.write(label)
                
                # 디버그 이미지 생성 및 저장
                debug_img = draw_bounding_box_on_image(img, label)
                debug_img.save(debug_path)
            
            # 검증 데이터 저장
            for i, (img, label) in enumerate(val_data):
                img_filename = f"{item_name}_{i}.jpg"
                label_filename = f"{item_name}_{i}.txt"
                
                img_path = os.path.join(val_img_dir, img_filename)
                label_path = os.path.join(val_label_dir, label_filename)
                debug_path = os.path.join(debug_dir, f"debug_val_{item_name}_{i}.jpg")
                
                img.save(img_path)
                with open(label_path, 'w') as f:
                    f.write(label)
                
                # 디버그 이미지 생성 및 저장
                debug_img = draw_bounding_box_on_image(img, label)
                debug_img.save(debug_path)
            
            print(f"아이템 {item_name} 처리 완료: 훈련 이미지 {len(train_data)}개, 검증 이미지 {len(val_data)}개")
            
        except Exception as e:
            print(f"아이템 '{item_path}' 처리 오류: {e}")
            continue
    
    # YOLO 데이터셋 구성 파일 생성
    create_yolo_dataset_config(output_dir)
    
    print("데이터셋 생성 완료!")

# YOLO 데이터셋 구성 파일 생성
def create_yolo_dataset_config(output_dir):
    # 데이터셋 구성 YAML 파일 생성
    yaml_content = """
# YOLOv5 dataset configuration
path: {0}  # dataset root dir
train: images/train  # train images relative to 'path'
val: images/val  # val images relative to 'path'

# Classes
nc: 1  # number of classes
names: ['pokecolo']  # class names
""".format(os.path.abspath(output_dir))

    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    main()