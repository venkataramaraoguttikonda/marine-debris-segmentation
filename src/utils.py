import os

# ------------------------------ Load Split IDs ------------------------------
def load_split_file(filepath):
    """
    Loads image IDs from a split file and prefixes each with 'S2_'.
    """
    with open(filepath, "r") as f:
        return ["S2_" + line.strip() for line in f if line.strip()]

# ------------------------------ Resolve Image Paths ------------------------------
def get_image_paths_from_ids(ids_list, patch_root):
    """
    Resolves full image paths from list of IDs and checks existence.
    """
    image_paths = []
    for id_ in ids_list:
        image_path = os.path.join(patch_root, f"{id_}.tif")
        if os.path.exists(image_path):
            image_paths.append(image_path)
        else:
            print(f"Missing image: {image_path}")
    return image_paths
