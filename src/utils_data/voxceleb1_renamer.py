# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os


def rename_top_level_folders(directory):
    """
    Rename all top-level folders within a specified directory by adding a consistent prefix.

    This function iterates over the immediate (top-level) contents of the given directory.
    For each folder it finds, it renames it by prefixing the folder name with:
    `"VoxCeleb1_VoxCeleb1++VoxCeleb1_"`.

    Args:
        directory (str): The path to the directory containing the folders to rename.

    Notes: 
        - Only directories at the **top level** (i.e., immediate children of `directory`) are renamed.
        - Files or symbolic links in the directory are ignored.
        - The function prints a message for each successful rename.
        - Example:
            If `directory` contains:
            ```
            id10001/
            id10002/
            ```
            After running this function, it will become:
            ```
            VoxCeleb1_VoxCeleb1++VoxCeleb1_id10001/
            VoxCeleb1_VoxCeleb1++VoxCeleb1_id10002/
            ```
    """
    prefix = "VoxCeleb1_VoxCeleb1++VoxCeleb1_"
    
    # Get all entries in the directory
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        
        # Check if it's a directory (not a file or symlink)
        if os.path.isdir(full_path):
            new_name = prefix + entry
            new_path = os.path.join(directory, new_name)
            
            # Rename the folder
            os.rename(full_path, new_path)
            print(f'Renamed: {entry} -> {new_name}')


if __name__ == "__main__":
    directory_path = '.../voxceleb1/wav/'
    rename_top_level_folders(directory_path)