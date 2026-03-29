import os
import glob

def convert_seg_to_bbox(input_dir, output_dir):
    """
    YOLO Segmentation (Polygon) 형식을 YOLO Bounding Box 형식으로 변환합니다.
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

    for file_path in label_files:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 3: # class_id와 최소 1쌍의 x,y 좌표가 있어야 함
                continue

            class_id = int(parts[0])
            coords = parts[1:]

            # x, y 좌표 분리 (짝수 인덱스는 x, 홀수 인덱스는 y)
            xs = coords[0::2]
            ys = coords[1::2]

            if not xs or not ys:
                continue

            # 폴리곤 좌표에서 최소/최대값을 찾아 바운딩 박스 계산
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # YOLO BBox 형식 계산 (정규화된 좌표 유지)
            width = x_max - x_min
            height = y_max - y_min
            x_center = x_min + (width / 2)
            y_center = y_min + (height / 2)

            # 결과 형식: class_id x_center y_center width height
            new_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            new_lines.append(new_line)

        # 변환된 내용 저장
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, 'w') as f:
            f.writelines(new_lines)

    print(f"변환 완료: {len(label_files)}개의 파일이 '{output_dir}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    INPUT_LABELS = 'bale_2' + os.sep + 'test' + os.sep + 'labels'
    OUTPUT_LABELS = 'bale_2' + os.sep + 'test' + os.sep + 'labels_bbox'
    
    convert_seg_to_bbox(INPUT_LABELS, OUTPUT_LABELS)
