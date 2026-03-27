import os
from tqdm.auto import tqdm


def rename_yolo_dataset(base_path, header_name='file'):
    images_path = os.path.join(base_path, 'images')
    labels_path = os.path.join(base_path, 'labels')

    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        print(f"Error: Path not found. {images_path} or {labels_path}")
        return

    image_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    image_files.sort()

    count = 1
    for img_name in image_files:
        base_name, img_ext = os.path.splitext(img_name)
        
        label_name = base_name + '.txt'
        label_src = os.path.join(labels_path, label_name)
        
        if os.path.exists(label_src):
            new_base_name = f"{header_name}_{count}"
            new_img_name = new_base_name + img_ext
            new_label_name = new_base_name + ".txt"
            
            img_src = os.path.join(images_path, img_name)
            img_dst = os.path.join(images_path, new_img_name)
            label_dst = os.path.join(labels_path, new_label_name)
            
            try:
                os.rename(img_src, img_dst)
                os.rename(label_src, label_dst)
                print(f"Renamed: {img_name} -> {new_img_name}")
                count += 1
            except Exception as e:
                print(f"Failed to rename {img_name}: {e}")
        else:
            print(f"Warning: Label not found for image {img_name}. Skipping.")

    print(f"\nFinished! Processed {count-1} pairs.")

if __name__ == "__main__":
    dataset_root = "bale_2\\valid"
    rename_yolo_dataset(dataset_root, header_name='bale_cls_2')
