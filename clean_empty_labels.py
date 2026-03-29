import os

def clean_empty_labels(images_dir, labels_dir):
    """
    Deletes label files with no content and their corresponding image files.
    """
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found: {labels_dir}")
        return

    deleted_count = 0
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    print(f"Checking {len(label_files)} label files...")

    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        
        # Check if the file is empty (0 bytes)
        if os.path.getsize(label_path) == 0:
            # Determine the corresponding image name
            # Assuming the image has the same base name and is a .jpg
            base_name = os.path.splitext(label_file)[0]
            image_file = base_name + ".jpg"
            image_path = os.path.join(images_dir, image_file)

            print(f"Empty label found: {label_file}")
            
            # Delete label file
            try:
                os.remove(label_path)
                print(f"  Deleted label: {label_file}")
            except Exception as e:
                print(f"  Failed to delete label {label_file}: {e}")
                continue

            # Delete image file if it exists
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    print(f"  Deleted image: {image_file}")
                except Exception as e:
                    print(f"  Failed to delete image {image_file}: {e}")
            else:
                print(f"  Warning: Corresponding image not found: {image_file}")
            
            deleted_count += 1

    print(f"\nCleanup complete. Total pairs deleted: {deleted_count}")

if __name__ == "__main__":
    # Define absolute paths based on the provided workspace
    base_path = 'bale_1' + os.sep + 'test'
    images_folder = os.path.join(base_path, "images")
    labels_folder = os.path.join(base_path, "labels")

    clean_empty_labels(images_folder, labels_folder)
