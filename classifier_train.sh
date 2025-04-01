#!/bin/bash

# 모든 아이템 타입별로 분류기 훈련 스크립트

# 실행 파일 경로
SCRIPT_PATH="python3 classifier_train.py"

# 모델명
MODEL_NAME="efficientnet_b0"

# GPU 장치 설정
GPU_DEVICE="cuda:0"

# 기본 인자
COMMON_ARGS="--model_name $MODEL_NAME --device $GPU_DEVICE --num_epochs 50"

# 모든 데이터셋 디렉토리 목록
echo "===== 분류기 훈련 시작 ====="

# TOP 훈련
echo "TOP 분류기 훈련 시작"
$SCRIPT_PATH --data_dir data/item_dataset_top $COMMON_ARGS
echo "TOP 분류기 훈련 완료"

# BOTTOMS 훈련
echo "BOTTOMS 분류기 훈련 시작"
$SCRIPT_PATH --data_dir data/item_dataset_bottoms $COMMON_ARGS
echo "BOTTOMS 분류기 훈련 완료"

# HAIR 훈련
echo "HAIR 분류기 훈련 시작"
$SCRIPT_PATH --data_dir data/item_dataset_hair $COMMON_ARGS
echo "HAIR 분류기 훈련 완료"

# ONEPIECE 훈련
echo "ONEPIECE 분류기 훈련 시작"
$SCRIPT_PATH --data_dir data/item_dataset_onepiece $COMMON_ARGS
echo "ONEPIECE 분류기 훈련 완료"

# OUTER 훈련
echo "OUTER 분류기 훈련 시작"
$SCRIPT_PATH --data_dir data/item_dataset_outer $COMMON_ARGS
echo "OUTER 분류기 훈련 완료"

# SHOES 훈련
echo "SHOES 분류기 훈련 시작"
$SCRIPT_PATH --data_dir data/item_dataset_shoes $COMMON_ARGS
echo "SHOES 분류기 훈련 완료"

# ACCESSORY_BACK 훈련
echo "ACCESSORY_BACK 분류기 훈련 시작"
$SCRIPT_PATH --data_dir data/item_dataset_accessory_back $COMMON_ARGS
echo "ACCESSORY_BACK 분류기 훈련 완료"

# ACCESSORY_BODY 훈련
echo "ACCESSORY_BODY 분류기 훈련 시작"
$SCRIPT_PATH --data_dir data/item_dataset_accessory_body $COMMON_ARGS
echo "ACCESSORY_BODY 분류기 훈련 완료"

# ACCESSORY_HEAD 훈련
echo "ACCESSORY_HEAD 분류기 훈련 시작"
$SCRIPT_PATH --data_dir data/item_dataset_accessory_head $COMMON_ARGS
echo "ACCESSORY_HEAD 분류기 훈련 완료"

# ACCESSORY_LEFT_RIGHT 훈련 (합쳐진 데이터셋)
echo "ACCESSORY_LEFT_RIGHT 분류기 훈련 시작"
$SCRIPT_PATH --data_dir data/item_dataset_accessory_left_right $COMMON_ARGS
echo "ACCESSORY_LEFT_RIGHT 분류기 훈련 완료"

# EYES 훈련
echo "EYES 분류기 훈련 시작"
$SCRIPT_PATH --data_dir data/item_dataset_eyes $COMMON_ARGS
echo "EYES 분류기 훈련 완료"

echo "===== 모든 분류기 훈련 완료 ====="