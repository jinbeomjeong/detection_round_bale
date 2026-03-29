import os
import glob

def filter_yolo_dataset(base_path, keep_classes):
    """
    YOLO 데이터셋에서 지정된 클래스만 남기고 나머지는 삭제합니다.
    해당 클래스가 없는 파일은 이미지와 라벨 모두 삭제합니다.
    """
    images_dir = os.path.join(base_path, 'images')
    labels_dir = os.path.join(base_path, 'labels')
    
    # 라벨 파일 목록 가져오기
    label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    
    deleted_count = 0
    filtered_count = 0
    
    print(f"Filtering classes: {keep_classes}")
    
    for label_path in label_files:
        filename = os.path.basename(label_path)
        base_name = os.path.splitext(filename)[0]
        
        # 라벨 파일 읽기
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # 지정된 클래스만 필터링
        filtered_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id in keep_classes:
                filtered_lines.append(line)
        
        # 필터링된 결과 처리
        if not filtered_lines:
            # 해당 클래스가 없으면 라벨과 이미지 삭제
            os.remove(label_path)
            
            # 이미지 파일 찾아서 삭제 (jpg, png 등 일반적인 확장자 체크)
            image_deleted = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_path = os.path.join(images_dir, base_name + ext)
                if os.path.exists(image_path):
                    os.remove(image_path)
                    image_deleted = True
                    break
            
            if image_deleted:
                deleted_count += 1
                # print(f"Deleted: {base_name} (No target classes found)")
        else:
            # 남은 클래스가 있으면 파일 업데이트
            if len(filtered_lines) < len(lines):
                with open(label_path, 'w') as f:
                    f.writelines(filtered_lines)
                filtered_count += 1
                # print(f"Filtered: {base_name} (Kept {len(filtered_lines)} objects)")

    print("-" * 30)
    print(f"Process finished.")
    print(f"Deleted (empty): {deleted_count} pairs")
    print(f"Modified (filtered): {filtered_count} labels")

if __name__ == "__main__":
    # 설정: 데이터셋 경로와 유지할 클래스 번호 리스트
    DATASET_PATH = 'dataset' + os.sep + 'valid'
    # 예: 클래스 0번과 2번만 남기고 싶을 경우 [0, 2]
    TARGET_CLASSES = [0] 
    
    filter_yolo_dataset(DATASET_PATH, TARGET_CLASSES)
