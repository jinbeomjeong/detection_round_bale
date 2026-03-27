import os
import glob

def change_class_id(input_dir, output_dir, new_id):
    """
    YOLO 라벨 파일(.txt)의 모든 클래스 번호를 new_id로 변경합니다.
    """
    # 출력 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"폴더 생성: {output_dir}")

    # 모든 .txt 파일 검색
    label_files = glob.glob(os.path.join(input_dir, "*.txt"))

    if not label_files:
        print(f"'{input_dir}' 폴더에 .txt 파일이 없습니다.")
        return

    count = 0
    for file_path in label_files:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                # 첫 번째 항목(class_id)을 new_id로 변경하고 나머지 좌표 유지
                parts[0] = str(new_id)
                new_line = " ".join(parts) + "\n"
                new_lines.append(new_line)

        # 변환된 내용 저장
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, 'w') as f:
            f.writelines(new_lines)
        count += 1

    print(f"변경 완료: {count}개의 파일에서 클래스 번호를 '{new_id}'로 변경하여 '{output_dir}'에 저장했습니다.")

if __name__ == "__main__":
    # 설정값
    INPUT_LABELS = "bale_1\\train\\labels"      # 원본 라벨 폴더
    OUTPUT_LABELS = "bale_1\\train\\labels_changed" # 변경된 라벨 저장 폴더
    NEW_CLASS_ID = 1             # 변경할 새로운 클래스 번호 (예: 1)
    
    # 실행
    change_class_id(INPUT_LABELS, OUTPUT_LABELS, NEW_CLASS_ID)
