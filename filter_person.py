import os


def filter_dataset(images_dir: str, labels_dir: str, only_target_cls=True):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    PERSON_CLASS_ID = '0'

    if not os.path.exists(labels_dir):
        print(f"Error: Label directory not found: {labels_dir}")
        return
    if not os.path.exists(images_dir):
        print(f"Error: Image directory not found: {images_dir}")
        return

    count_deleted = 0
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    print(f"Starting inspection: {len(label_files)} label files found.")

    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        file_name_no_ext = os.path.splitext(label_file)[0]
        
        should_delete = False
        
        # 1. Check label file content
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
                if not lines:
                    # Delete if empty
                    should_delete = True
                else:
                    # Collect all class IDs in the file
                    classes_in_file = [line.split()[0] for line in lines if line.strip()]
                    
                    # If any class is NOT the person class, or if person class is missing entirely
                    if not classes_in_file:
                        should_delete = True
                    elif PERSON_CLASS_ID not in classes_in_file:
                        should_delete = True
                    else:
                        for class_id in classes_in_file:
                            if class_id != PERSON_CLASS_ID:
                                should_delete = only_target_cls
                                break
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
            continue

        # 2. Execute deletion
        if should_delete:
            try:
                # Delete label
                os.remove(label_path)
                
                # Find and delete corresponding image
                img_deleted = False
                for ext in img_extensions:
                    img_path = os.path.join(images_dir, file_name_no_ext + ext)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        img_deleted = True
                        break
                
                count_deleted += 1
                status = "Label + Image" if img_deleted else "Label only"
                print(f"Deleted: {file_name_no_ext} ({status})")
            except Exception as e:
                print(f"Error deleting {file_name_no_ext}: {e}")

    print("-" * 30)
    print(f"Done! Total {count_deleted} data sets removed.")

if __name__ == "__main__":

    base_path = "dataset\\val"
    images_dir = os.path.join(base_path, 'images')
    labels_dir = os.path.join(base_path, 'labels')

    filter_dataset(images_dir=images_dir, labels_dir=labels_dir, only_target_cls=False)
