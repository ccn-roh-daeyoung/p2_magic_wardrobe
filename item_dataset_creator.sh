#!/bin/bash

# 실행 파일 경로
SCRIPT_PATH="python3 item_dataset_creator.py"

# 공통 인자
COMMON_ARGS="--skip_processed"

# 각 아이템 타입 처리
echo "TOP 처리 시작"
$SCRIPT_PATH --item_type TOP $COMMON_ARGS

echo "BOTTOMS 처리 시작"
$SCRIPT_PATH --item_type BOTTOMS $COMMON_ARGS

echo "HAIR 처리 시작"
$SCRIPT_PATH --item_type HAIR $COMMON_ARGS

echo "ONEPIECE 처리 시작"
$SCRIPT_PATH --item_type ONEPIECE $COMMON_ARGS

echo "OUTER 처리 시작"
$SCRIPT_PATH --item_type OUTER $COMMON_ARGS

echo "SHOES 처리 시작"
$SCRIPT_PATH --item_type SHOES $COMMON_ARGS

echo "ACCESSORY_BACK 처리 시작"
$SCRIPT_PATH --item_type ACCESSORY_BACK $COMMON_ARGS

echo "ACCESSORY_BODY 처리 시작"
$SCRIPT_PATH --item_type ACCESSORY_BODY $COMMON_ARGS

echo "ACCESSORY_HEAD 처리 시작"
$SCRIPT_PATH --item_type ACCESSORY_HEAD $COMMON_ARGS

echo "ACCESSORY_LEFT & ACCESSORY_RIGHT 처리 시작"
$SCRIPT_PATH --item_type ACCESSORY_LEFT ACCESSORY_RIGHT --output_dir data/item_dataset_accessory_left_right $COMMON_ARGS

echo "EYE 처리 시작"
python item_dataset_creator_eye.py $COMMON_ARGS

echo "모든 아이템 타입 처리 완료"