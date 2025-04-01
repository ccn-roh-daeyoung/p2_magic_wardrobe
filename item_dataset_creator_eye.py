import os
import numpy as np
from PIL import Image
import glob
import random
from pathlib import Path
import argparse

def is_already_processed(item_id, output_path, val_output_path, min_files=6):
    """
    주어진 item_id가 이미 처리되었는지 확인합니다.
    
    Args:
        item_id: 확인할 아이템 ID
        output_path: 훈련 데이터 출력 폴더 경로
        val_output_path: 검증 데이터 출력 폴더 경로
        min_files: 처리된 것으로 간주할 최소 파일 수 (기본값: 원본 2개 + 증강 4개 = 6개)
    
    Returns:
        bool: 이미 처리된 경우 True, 그렇지 않으면 False
    """
    # 훈련 및 검증 디렉토리 경로
    train_dir = os.path.join(output_path, item_id)
    val_dir = os.path.join(val_output_path, item_id)
    
    # 두 디렉토리가 모두 존재하는지 확인
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        # 훈련 디렉토리의 파일 수 확인
        train_files = [f for f in os.listdir(train_dir) if f.lower().endswith('.png')]
        
        # 검증 디렉토리의 파일 수 확인
        val_files = [f for f in os.listdir(val_dir) if f.lower().endswith('.png')]
        
        # 최소 파일 수 충족 확인
        if len(train_files) >= min_files and len(val_files) >= 2:  # 2 = 왼쪽눈 + 오른쪽눈
            return True
    
    return False

def separate_and_augment_eyes(input_path, output_path, val_output_path, augmentations_per_eye=5, skip_processed=False):
    """
    RGBA 이미지에서 알파 값을 이용해 왼쪽 눈과 오른쪽 눈을 분리하여 저장하고,
    각 눈 이미지를 일부 가리는 방식으로 증강합니다.
    
    Args:
        input_path: 원본 이미지 파일 경로 (data/p2/EYE/{item_id}/{item_id}.png)
        output_path: 출력 폴더 경로 (data/item_dataset_eyes)
        val_output_path: 검증 데이터 출력 폴더 경로
        augmentations_per_eye: 각 눈마다 생성할 증강 이미지 수
        skip_processed: 이미 처리된 아이템을 건너뛸지 여부
    """
    # 모든 item_id 폴더를 찾습니다
    item_folders = glob.glob(os.path.join(input_path, "*"))
    
    # 처리된 아이템 및 건너뛴 아이템 카운터
    processed_count = 0
    skipped_count = 0
    
    for item_folder in item_folders:
        item_id = os.path.basename(item_folder)
        input_file = os.path.join(item_folder, f"{item_id}.png")
        
        # 이미 처리되었는지 확인하고 건너뛰기
        if skip_processed and is_already_processed(item_id, output_path, val_output_path, min_files=2 + augmentations_per_eye * 2):
            print(f"아이템 {item_id} - 이미 처리됨, 건너뜀")
            skipped_count += 1
            continue
        
        # 이미지 파일이 존재하는지 확인합니다
        if not os.path.exists(input_file):
            print(f"경고: {input_file} 파일을 찾을 수 없습니다.")
            continue
        
        # 출력 폴더 생성
        output_folder = os.path.join(output_path, item_id)
        val_output_folder = os.path.join(val_output_path, item_id)
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(val_output_folder, exist_ok=True)
        
        # 이미지 로드
        try:
            img = Image.open(input_file).convert("RGBA")
            img_array = np.array(img)
            
            # 알파 채널 추출
            alpha_channel = img_array[:, :, 3]
            height, width = alpha_channel.shape
            
            # 이미지 중앙점 계산
            mid_point = width // 2
            
            # 각 행에서 분리선을 찾고, 그 중 가장 안쪽(중앙에 가까운) 위치를 사용
            left_separation_line = 0  # 왼쪽 눈과 중앙 사이의 경계
            right_separation_line = width - 1  # 오른쪽 눈과 중앙 사이의 경계
            
            for y in range(height):
                if np.max(alpha_channel[y]) == 0:
                    continue  # 알파값이 없는 행은 건너뜀
                
                # 중앙에서 왼쪽으로 이동하며 알파값이 0이 아닌 첫 지점 찾기
                for x in range(mid_point, -1, -1):
                    if alpha_channel[y, x] > 0:
                        # 현재 행의 왼쪽 눈 오른쪽 경계
                        row_left_eye_right = x
                        
                        # 계속 왼쪽으로 이동하며 알파값이 0인 지점 찾기 (왼쪽 눈 왼쪽 경계)
                        for x2 in range(row_left_eye_right, -1, -1):
                            if alpha_channel[y, x2] == 0:
                                # 현재 행의 왼쪽 눈과 중앙 사이의 경계
                                row_left_separation = x2
                                # 가장 오른쪽(중앙에 가까운) 분리선 업데이트
                                left_separation_line = max(left_separation_line, row_left_separation)
                                break
                        break
                
                # 중앙에서 오른쪽으로 이동하며 알파값이 0이 아닌 첫 지점 찾기
                for x in range(mid_point, width):
                    if alpha_channel[y, x] > 0:
                        # 현재 행의 오른쪽 눈 왼쪽 경계
                        row_right_eye_left = x
                        
                        # 계속 오른쪽으로 이동하며 알파값이 0인 지점 찾기 (오른쪽 눈 오른쪽 경계)
                        for x2 in range(row_right_eye_left, width):
                            if alpha_channel[y, x2] == 0:
                                # 현재 행의 오른쪽 눈과 중앙 사이의 경계
                                row_right_separation = x2
                                # 가장 왼쪽(중앙에 가까운) 분리선 업데이트
                                right_separation_line = min(right_separation_line, row_right_separation)
                                break
                        break
            
            # 분리선이 제대로 찾아졌는지 확인
            # 만약 분리선이 초기값과 같다면, 해당 눈이 없거나 경계를 찾지 못한 것
            if left_separation_line == 0:
                left_separation_line = mid_point // 2  # 기본값 설정
            
            if right_separation_line == width - 1:
                right_separation_line = mid_point + (width - mid_point) // 2  # 기본값 설정
            
            print(f"{item_id} 분리선: 왼쪽={left_separation_line}, 오른쪽={right_separation_line}")
            
            # 왼쪽 눈 이미지 생성 (left_separation_line 왼쪽의 모든 픽셀을 크롭)
            left_eye_region = img_array[:, :left_separation_line]
            left_eye = Image.fromarray(left_eye_region)
            left_eye.save(os.path.join(output_folder, f"{item_id}_left_eye.png"))
            left_eye.save(os.path.join(val_output_folder, f"{item_id}_left_eye.png"))
            
            # 오른쪽 눈 이미지 생성 (right_separation_line 오른쪽의 모든 픽셀을 크롭)
            right_eye_region = img_array[:, right_separation_line:]
            right_eye = Image.fromarray(right_eye_region)
            right_eye.save(os.path.join(output_folder, f"{item_id}_right_eye.png"))
            right_eye.save(os.path.join(val_output_folder, f"{item_id}_right_eye.png"))
            
            print(f"{item_id} 눈 이미지 분리 완료")
            
            # 왼쪽 눈 이미지 증강
            left_eye_height, left_eye_width = left_eye_region.shape[:2]
            for i in range(augmentations_per_eye):
                # 증강 이미지 생성 (원본 복사)
                aug_left_eye = left_eye_region.copy()
                
                # 가릴 영역 크기 설정 (이미지 크기의 10-30%)
                mask_width = random.randint(int(left_eye_width * 0.1), int(left_eye_width * 0.3))
                mask_height = random.randint(int(left_eye_height * 0.1), int(left_eye_height * 0.3))
                
                # 가릴 영역 위치 설정 (랜덤)
                mask_x = random.randint(0, left_eye_width - mask_width)
                mask_y = random.randint(0, left_eye_height - mask_height)
                
                # 해당 영역의 알파값을 0으로 설정 (완전 투명하게)
                aug_left_eye[mask_y:mask_y+mask_height, mask_x:mask_x+mask_width, 3] = 0
                
                # 증강된 이미지 저장
                aug_left_eye_img = Image.fromarray(aug_left_eye)
                aug_left_eye_img.save(os.path.join(output_folder, f"{item_id}_left_eye_aug{i+1}.png"))
            
            # 오른쪽 눈 이미지 증강
            right_eye_height, right_eye_width = right_eye_region.shape[:2]
            for i in range(augmentations_per_eye):
                # 증강 이미지 생성 (원본 복사)
                aug_right_eye = right_eye_region.copy()
                
                # 가릴 영역 크기 설정 (이미지 크기의 10-30%)
                mask_width = random.randint(int(right_eye_width * 0.1), int(right_eye_width * 0.3))
                mask_height = random.randint(int(right_eye_height * 0.1), int(right_eye_height * 0.3))
                
                # 가릴 영역 위치 설정 (랜덤)
                mask_x = random.randint(0, right_eye_width - mask_width)
                mask_y = random.randint(0, right_eye_height - mask_height)
                
                # 해당 영역의 알파값을 0으로 설정 (완전 투명하게)
                aug_right_eye[mask_y:mask_y+mask_height, mask_x:mask_x+mask_width, 3] = 0
                
                # 증강된 이미지 저장
                aug_right_eye_img = Image.fromarray(aug_right_eye)
                aug_right_eye_img.save(os.path.join(output_folder, f"{item_id}_right_eye_aug{i+1}.png"))
            
            print(f"{item_id} 눈 이미지 증강 완료 (각 눈당 {augmentations_per_eye}개)")
            processed_count += 1
            
        except Exception as e:
            print(f"오류: {item_id} 처리 중 문제 발생 - {str(e)}")
    
    print(f"처리 완료: 총 {processed_count}개 아이템 처리됨, {skipped_count}개 건너뜀")

# 메인 실행 코드
if __name__ == "__main__":
    # 명령줄 인자 파서 설정
    parser = argparse.ArgumentParser(description='눈 이미지 분리 및 증강 도구')
    
    parser.add_argument('--input_path', type=str, default="data/EYE",
                        help='입력 이미지 폴더 경로')
    
    parser.add_argument('--output_path', type=str, default="data/item_dataset_eyes/train",
                        help='훈련 데이터 출력 폴더 경로')
    
    parser.add_argument('--val_output_path', type=str, default="data/item_dataset_eyes/val",
                        help='검증 데이터 출력 폴더 경로')
    
    parser.add_argument('--augmentations', type=int, default=5,
                        help='각 눈마다 생성할 증강 이미지 수')
    
    parser.add_argument('--skip_processed', action='store_true',
                        help='이미 처리된 아이템을 건너뛰려면 이 옵션을 사용')
    
    args = parser.parse_args()
    
    # 출력 폴더가 없으면 생성
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.val_output_path, exist_ok=True)
    
    # 눈 이미지 분리 및 증강 실행
    separate_and_augment_eyes(
        args.input_path, 
        args.output_path, 
        args.val_output_path, 
        augmentations_per_eye=args.augmentations,
        skip_processed=args.skip_processed
    )
    print("모든 이미지 처리 완료!")

# TODO: 썸네일 이미지말고 EYE만 있은 이미지 필요!!!
# TODO: 마스터 테이블에서 EYE 리스트 가져와서 EYE 폴더 만들고 해당 이미지들로 증강하는 코드로 수정