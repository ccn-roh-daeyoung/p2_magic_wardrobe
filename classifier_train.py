import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import datetime
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument('--model_name', type=str, default='caformer_s18',
                        choices=['caformer_s18', 'vit_base_patch16_224', 'efficientnet_b0'],
                        help='Model architecture to use')
    parser.add_argument('--data_dir', type=str, default='data/dataset',
                        help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save model and results')
    parser.add_argument('--experiment_name', type=str, default='',
                        help='Experiment name for output folder')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    return parser.parse_args()

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = []
        self.class_to_idx = {}
        self.samples = []
        
        # 모든 클래스 디렉토리를 찾습니다
        self._find_classes()
        
        # 모든 샘플을 찾습니다
        self._make_dataset()
        
    def _find_classes(self):
        # root_dir 아래의 모든 디렉토리를 클래스로 간주합니다
        classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.classes = classes
        
    def _make_dataset(self):
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for root, _, filenames in os.walk(class_dir):
                for filename in filenames:
                    if filename.lower().endswith(extensions):
                        image_path = os.path.join(root, filename)
                        self.samples.append((image_path, class_idx))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, class_idx = self.samples[idx]
        image = self._load_image(path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, class_idx
    
    def _load_image(self, path):
        from PIL import Image
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


def create_data_loaders(data_dir, batch_size, num_workers):
    # 데이터 전처리 정의
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 디렉토리 구조 확인
    print(f"데이터 디렉토리 확인: {data_dir}")
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        print("train/val 폴더 구조 확인됨")
        
        # 커스텀 데이터셋 사용
        train_dataset = CustomDataset(train_dir, transform=transform_train)
        val_dataset = CustomDataset(val_dir, transform=transform_val)
        
        print(f"훈련 데이터셋: {len(train_dataset.classes)} 클래스, {len(train_dataset)} 이미지")
        print(f"검증 데이터셋: {len(val_dataset.classes)} 클래스, {len(val_dataset)} 이미지")
        
        # 클래스 이름과 이미지 개수 확인
        print("\n클래스별 이미지 개수 (처음 10개 클래스):")
        class_counts = {}
        for _, class_idx in train_dataset.samples:
            class_name = train_dataset.classes[class_idx]
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        
        for i, (class_name, count) in enumerate(list(class_counts.items())[:10]):
            print(f" - {class_name}: {count}개 이미지")
        
        if len(class_counts) > 10:
            print(f" ... 그 외 {len(class_counts) - 10}개 클래스")
    else:
        raise ValueError(f"train 또는 val 디렉토리가 존재하지 않습니다. data_dir: {data_dir}")
        
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # 클래스 이름 가져오기
    class_names = train_dataset.classes
    print(f"클래스 수: {len(class_names)}")
    
    return train_loader, val_loader, class_names

def create_model(model_name, num_classes, device):
    if model_name == 'caformer_s18':
        # CAFormer는 timm 라이브러리에서 가져옵니다
        model = timm.create_model('caformer_s18', pretrained=True, num_classes=num_classes)
    elif model_name == 'vit_base_patch16_224':
        # ViT는 timm 라이브러리에서 가져옵니다
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    elif model_name == 'efficientnet_b0':
        # EfficientNet은 torchvision에서 가져올 수 있지만, 여기서는 일관성을 위해 timm 사용
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")
    
    model = model.to(device)
    return model

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 그래디언트 초기화
        optimizer.zero_grad()
        
        # 순전파
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 역전파 및 최적화
        loss.backward()
        optimizer.step()
        
        # 통계
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 진행 상황 업데이트
        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 순전파
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 통계
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 진행 상황 업데이트
            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def plot_results(train_losses, val_losses, train_accs, val_accs, output_dir):
    plt.figure(figsize=(12, 5))
    
    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_results.png'))
    plt.close()
    
def save_class_mapping(class_to_idx, output_dir):
    """클래스명과 인덱스 간의 매핑을 JSON 파일로 저장합니다."""
    import json
    
    # 클래스 이름과 인덱스 매핑
    class_mapping = {
        "idx_to_class": {str(idx): class_name for class_name, idx in class_to_idx.items()},
        "class_to_idx": {class_name: idx for class_name, idx in class_to_idx.items()}
    }
    
    # JSON 파일로 저장
    mapping_path = os.path.join(output_dir, "class_mapping.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, ensure_ascii=False, indent=4)
    
    print(f"클래스 매핑을 {mapping_path}에 저장했습니다.")
    return mapping_path

def create_experiment_dir(base_dir, model_name, experiment_name=""):
    """실험 결과를 저장할 디렉토리를 생성합니다."""
    # 현재 시간을 포함한 실험 폴더 이름 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        exp_dir_name = f"{timestamp}_{model_name}_{experiment_name}"
    else:
        exp_dir_name = f"{timestamp}_{model_name}"
    
    # 전체 경로 생성 및 디렉토리 생성
    exp_dir = os.path.join(base_dir, exp_dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    return exp_dir

def main():
    args = parse_args()
    
    # 실험 디렉토리 생성
    experiment_dir = create_experiment_dir(
        args.output_dir, 
        args.model_name,
        args.experiment_name
    )
    print(f"실험 결과는 다음 디렉토리에 저장됩니다: {experiment_dir}")
    
    # 실험 구성 저장
    config = vars(args)
    config['experiment_dir'] = experiment_dir
    config['start_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(os.path.join(experiment_dir, "config.json"), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    
    # 입력 인자 확인
    print(f"모델: {args.model_name}")
    print(f"데이터 디렉토리: {args.data_dir}")
    print(f"배치 크기: {args.batch_size}")
    print(f"에폭 수: {args.num_epochs}")
    print(f"학습률: {args.learning_rate}")
    print(f"출력 디렉토리: {experiment_dir}")
    print(f"장치: {args.device}")
    
    # 디렉토리 존재 여부 확인
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"데이터 디렉토리가 존재하지 않습니다: {args.data_dir}")
    
    # 데이터 로더 생성
    train_loader, val_loader, class_names = create_data_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    
    # 데이터셋 객체 가져오기
    train_dataset = train_loader.dataset
    
    # 클래스 매핑 저장
    mapping_path = save_class_mapping(train_dataset.class_to_idx, experiment_dir)
    print(f"클래스 매핑이 {mapping_path}에 저장되었습니다.")
    
    # 모델 생성
    model = create_model(args.model_name, len(class_names), args.device)
    print(f"모델 {args.model_name} 생성 완료, 클래스 수: {len(class_names)}")
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # 학습 결과 저장용 리스트
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    # 학습 시작 시간 기록
    start_time = time.time()
    
    # 학습 시작
    print(f"학습 시작: {args.num_epochs} 에폭")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # 학습
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 검증
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 학습률 조정
        scheduler.step()
        
        # 현재 에폭 결과 출력
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 최고 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(experiment_dir, f"{args.model_name}_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, model_path)
            print(f"최고 모델 저장: {model_path}")
        
        # 에폭 결과 저장 (체크포인트)
        epoch_info = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        with open(os.path.join(experiment_dir, f"epoch_{epoch+1:03d}.json"), 'w') as f:
            json.dump(epoch_info, f, indent=4)
    
    # 학습 종료 시간 계산
    end_time = time.time()
    training_time = end_time - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # 최종 모델 저장
    model_path = os.path.join(experiment_dir, f"{args.model_name}_final.pth")
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'class_names': class_names
    }, model_path)
    print(f"최종 모델 저장: {model_path}")
    
    # 결과 시각화
    plot_results(train_losses, val_losses, train_accs, val_accs, experiment_dir)
    
    # 클래스 매핑 JSON을 모델과 함께 다시 저장 (중복이지만 모델과 함께 있는 것이 편리함)
    train_dataset = train_loader.dataset
    mapping_path = save_class_mapping(train_dataset.class_to_idx, experiment_dir)
    
    # 모델 정보 JSON 저장 (추가 정보)
    model_info = {
        'model_name': args.model_name,
        'num_classes': len(class_names),
        'class_names': class_names,
        'final_accuracy': val_acc,
        'best_accuracy': best_val_acc,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'date_trained': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'training_time': f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}",
        'training_time_seconds': training_time
    }
    
    model_info_path = os.path.join(experiment_dir, f"{args.model_name}_info.json")
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=4)
    
    # 학습 손실과 정확도 저장
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    history_path = os.path.join(experiment_dir, "training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)
    
    print(f"모델 정보를 {model_info_path}에 저장했습니다.")
    print(f"학습 완료! 최고 검증 정확도: {best_val_acc:.2f}%")
    print(f"학습 시간: {int(hours):02d}시간 {int(minutes):02d}분 {int(seconds):02d}초")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"오류 발생: {e}")
        print(traceback.format_exc())
        # 사용 가능한 이미지 확장자 목록
        print("\n지원되는 이미지 확장자: .jpg, .jpeg, .png, .ppm, .bmp, .pgm, .tif, .tiff, .webp")
        # 확장자 문제인지 확인을 위한 디렉토리 스캔
        import glob
        data_dir = parse_args().data_dir
        print(f"\n데이터 디렉토리 스캔 결과:")
        all_files = glob.glob(f"{data_dir}/**/*.*", recursive=True)
        exts = {}
        for f in all_files[:20]:  # 처음 20개 파일만 표시
            ext = os.path.splitext(f)[1].lower()
            if ext not in exts:
                exts[ext] = 0
            exts[ext] += 1
            print(f" - {f}")
        
        if len(all_files) > 20:
            print(f" ... 그 외 {len(all_files)-20}개 파일")
        
        print("\n파일 확장자 통계:")
        for ext, count in exts.items():
            print(f" - {ext}: {count}개 파일")

# python3 classifier_train.py --data_dir data/item_dataset_accesory_back --model_name efficientnet_b0 --device cuda:0