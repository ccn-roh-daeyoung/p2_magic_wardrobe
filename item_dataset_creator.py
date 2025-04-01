
import os
import json
import random
import shutil
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import argparse

# 디렉토리 생성 함수
def create_directories(base_dir):
    # 훈련 및 검증 디렉토리 생성
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    return train_dir, val_dir

# 이미지를 웹에서 다운로드하는 함수
def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        raise Exception(f"이미지 다운로드 실패: {url}, 상태 코드: {response.status_code}")

# 배경 이미지 목록 가져오기
def get_background_images(bg_dir):
    bg_images = []
    for filename in os.listdir(bg_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            bg_images.append(os.path.join(bg_dir, filename))
    return bg_images

# 알파 채널을 검정색으로 바꾸는 함수
def replace_alpha_with_black(img):
    # 이미지가 RGBA 모드인지 확인
    if img.mode != 'RGBA':
        return img
    
    # 새 이미지 생성 (RGB 모드)
    new_img = Image.new('RGB', img.size, (0, 0, 0))  # 검정색 배경
    
    # 알파 채널을 사용하여 원본 이미지를 검정색 배경에 합성
    new_img.paste(img, mask=img.split()[3])  # 3번 채널이 알파 채널
    
    return new_img

# 이미지 증강 함수
def augment_image(item_image, background_paths, num_augmentations=4):
    # 원본 이미지의 알파값을 검정색으로 바꾼 이미지를 첫 번째로 저장
    black_bg_image = replace_alpha_with_black(item_image)
    augmented_images = [black_bg_image]  # 검정 배경으로 변환된 원본 이미지
    
    # 배경이 충분하지 않은 경우 랜덤 샘플링
    if len(background_paths) < num_augmentations:
        background_paths = random.choices(background_paths, k=num_augmentations)
    else:
        background_paths = random.sample(background_paths, num_augmentations)
    
    for bg_path in background_paths:
        try:
            # 배경 이미지 로드
            bg_image = Image.open(bg_path).convert("RGBA")
            
            # 아이템 이미지 크기에 맞게 배경 크롭
            width, height = item_image.size
            
            # 배경 이미지가 아이템 이미지보다 작으면 리사이징
            if bg_image.width < width or bg_image.height < height:
                ratio = max(width / bg_image.width, height / bg_image.height)
                new_width = int(bg_image.width * ratio)
                new_height = int(bg_image.height * ratio)
                bg_image = bg_image.resize((new_width, new_height), Image.LANCZOS)
            
            # 중앙에서 크롭
            left = (bg_image.width - width) // 2
            top = (bg_image.height - height) // 2
            right = left + width
            bottom = top + height
            
            bg_image = bg_image.crop((left, top, right, bottom))
            
            # 증강된 이미지 생성 - 배경 위에 아이템 이미지 합성
            augmented_image = Image.new("RGBA", item_image.size)
            augmented_image.paste(bg_image, (0, 0))
            augmented_image.alpha_composite(item_image)
            
            # 여기서는 RGBA 형식 유지 (저장 전에 변환)
            augmented_images.append(augmented_image)
            
        except Exception as e:
            print(f"이미지 증강 오류: {e}")
            continue
    
    return augmented_images

# 아이템이 이미 처리되었는지 확인하는 함수
def is_already_processed(item_id, train_dir, val_dir):
    # 훈련 및 검증 디렉토리에 해당 아이템 ID로 된 폴더가 존재하는지 확인
    item_train_dir = os.path.join(train_dir, str(item_id))
    item_val_dir = os.path.join(val_dir, str(item_id))
    
    # 두 디렉토리 모두 존재하고, 각 디렉토리에 이미지가 하나 이상 있는지 확인
    if os.path.exists(item_train_dir) and os.path.exists(item_val_dir):
        # 훈련 디렉토리에 이미지가 있는지 확인
        train_images = [f for f in os.listdir(item_train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # 검증 디렉토리에 이미지가 있는지 확인
        val_images = [f for f in os.listdir(item_val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # 두 디렉토리 모두에 이미지가 있으면 이미 처리된 것으로 간주
        return len(train_images) > 0 and len(val_images) > 0
    
    return False

# 메인 함수
def main(args):
    # 경로 설정
    json_path = args.json_path
    bg_dir = args.bg_dir
    output_dir = args.output_dir
    target_item_type_list = args.item_type
    skip_processed = args.skip_processed
    
    # 출력 디렉토리 생성
    train_dir, val_dir = create_directories(output_dir)
    
    # 배경 이미지 목록 가져오기
    background_paths = get_background_images(bg_dir)
    if not background_paths:
        print("사용 가능한 배경 이미지가 없습니다.")
        return
    
    print(f"배경 이미지 {len(background_paths)}개 로드됨")
    
    # JSON 파일 읽기
    with open(json_path, 'r') as f:
        items_data = json.load(f)
    
    print(f"JSON에서 총 {len(items_data)}개의 아이템 정보 로드됨")
    
    # 처리된 아이템 수 카운터
    processed_count = 0
    skipped_count = 0
    
    # 각 아이템에 대해 증강 작업 수행
    for item in items_data:
        try:
            item_id = item.get('item_id')
            image_url = item.get('item_image_url')
            item_type = item.get('item_type')
            
            # 지정된 아이템 타입만 처리
            if not item_id or not image_url or item_type not in target_item_type_list:
                continue
            
            # 아이템별 디렉토리 경로
            item_train_dir = os.path.join(train_dir, str(item_id))
            item_val_dir = os.path.join(val_dir, str(item_id))
            
            # 이미 처리된 아이템인지 확인하고 건너뛰기 옵션이 켜져 있으면 건너뛴다
            if skip_processed and is_already_processed(item_id, train_dir, val_dir):
                print(f"아이템 {item_id} (타입: {item_type}) - 이미 처리됨, 건너뜀")
                skipped_count += 1
                continue
            
            print(f"아이템 처리 중: {item_id} (타입: {item_type})")
            
            # 아이템별 디렉토리 생성
            os.makedirs(item_train_dir, exist_ok=True)
            os.makedirs(item_val_dir, exist_ok=True)
            
            # 이미지 다운로드
            original_image = download_image(image_url)
            
            # 이미지가 RGBA 형식이 아니면 변환
            if original_image.mode != 'RGBA':
                original_image = original_image.convert('RGBA')
            
            # 이미지 증강
            augmented_images = augment_image(original_image, background_paths, args.num_augmentations)
            
            # 증강된 이미지 저장
            # 훈련 세트에 모든 증강 이미지 저장
            for i, img in enumerate(augmented_images):
                # RGBA에서 RGB로 변환 (JPEG는 알파 채널을 지원하지 않음)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img_filename = f"{i}.jpg"
                img_path = os.path.join(item_train_dir, img_filename)
                img.save(img_path)
            
            # 검증 세트에는 하나의 증강 이미지만 저장 (랜덤 선택)
            val_img = random.choice(augmented_images)
            # RGBA에서 RGB로 변환
            if val_img.mode == 'RGBA':
                val_img = val_img.convert('RGB')
            val_img_filename = f"val.jpg"
            val_img_path = os.path.join(item_val_dir, val_img_filename)
            val_img.save(val_img_path)
            
            processed_count += 1
            print(f"아이템 {item_id} 처리 완료: 훈련 이미지 {len(augmented_images)}개, 검증 이미지 1개")
            
        except Exception as e:
            print(f"아이템 {item_id} 처리 오류: {e}")
            continue
    
    print(f"처리 완료: {target_item_type_list} 타입 아이템 {processed_count}개 처리됨, {skipped_count}개 건너뜀")

if __name__ == "__main__":
    # 명령줄 인자 파서 설정
    parser = argparse.ArgumentParser(description='이미지 증강 및 데이터셋 생성 도구')
    
    parser.add_argument('--item_type', nargs='+', default=['BOTTOMS'],
                        help='처리할 아이템 타입 목록 (예: BOTTOMS TOP ONEPIECE)')
    
    parser.add_argument('--output_dir', type=str, 
                        help='출력 디렉토리 경로')
    
    parser.add_argument('--json_path', type=str, default='data/latest_items.json',
                        help='아이템 정보가 있는 JSON 파일 경로')
    
    parser.add_argument('--bg_dir', type=str, default='data/theme_sheet',
                        help='배경 이미지가 있는 디렉토리 경로')
    
    parser.add_argument('--num_augmentations', type=int, default=4,
                        help='각 아이템당 생성할 증강 이미지 수')
    
    parser.add_argument('--skip_processed', action='store_true',
                        help='이미 처리된 아이템을 건너뛰려면 이 옵션을 사용')
    
    args = parser.parse_args()
    
    # output_dir이 지정되지 않은 경우 기본값 설정
    if len(args.item_type) == 1:
        args.output_dir = f'data/item_dataset_{args.item_type[0].lower()}'
    # else:
    #     args.output_dir = f'data/item_dataset_accessory_left_right'
    
    main(args)

# 이미 처리된 이미지 건너뛰기 옵션을 사용하여 실행
# python script.py --item_type BOTTOMS TOP --skip_processed

# 이미 처리된 이미지도 다시 처리하려면 옵션을 생략
# python script.py --item_type BOTTOMS TOP