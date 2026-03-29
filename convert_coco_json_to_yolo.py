import json, os
from tqdm.auto import tqdm


def convert_coco_to_yolo(json_path:str, output_dir:str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create category mapping (COCO category_id -> 0-indexed id)
    categories = sorted(data['categories'], key=lambda x: x['id'])
    cat_id_map = {cat['id']: i for i, cat in enumerate(categories)}

    # Group annotations by image_id
    image_annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    print(f"Processing {len(data['images'])} images...")

    for img in tqdm(data['images']):
        img_id = img['id']
        file_name = img['file_name']
        width = img['width']
        height = img['height']
        
        # YOLO label file name
        label_file_name = os.path.splitext(file_name)[0] + '.txt'
        label_path = os.path.join(output_dir, label_file_name)
        
        anns = image_annotations.get(img_id, [])
        
        yolo_lines = []
        for ann in anns:
            cat_id = ann['category_id']
            yolo_cat_id = cat_id_map[cat_id]
            
            # COCO bbox: [x_min, y_min, width, height]
            x_min, y_min, w_box, h_box = ann['bbox']
            
            # YOLO: [x_center, y_center, width, height] normalized
            x_center = (x_min + w_box / 2) / width
            y_center = (y_min + h_box / 2) / height
            w_norm = w_box / width
            h_norm = h_box / height
            
            yolo_lines.append(f"{yolo_cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # Write to file even if empty (YOLO sometimes likes empty files for negative samples)
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

    print("Conversion complete.")

if __name__ == '__main__':
    json_path = 'dataset' + os.sep + 'instances_train2017.json'
    output_dir = 'dataset' + os.sep + 'train' + os.sep + 'labels'

    convert_coco_to_yolo(json_path=json_path, output_dir=output_dir)
