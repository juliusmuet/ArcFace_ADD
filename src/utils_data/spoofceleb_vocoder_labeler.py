# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from pathlib import Path


def create_labels_file(search_dir: str, base_dir: str, output_file: str = "labels.txt"):
    """
    Generate a text file containing label information for files within a specified directory structure.

    This function recursively searches through `search_dir`, finds all files, and writes a label line 
    for each file to `output_file`. Each line includes a relative path (relative to `base_dir`) and a 
    set of metadata fields based on the file's directory structure.

    Args:
        search_dir (str): The directory to recursively search for files.
        base_dir (str): The base directory used to compute relative file paths.
        output_file (str, optional): The name of the output file to write labels to (default: "labels.txt").

    Notes:
        - The function assumes that files are organised in nested folders and it extracts:
            - `folder`: the immediate parent folder of the file
            - `parent_folder`: the folder one level above `folder`
        - Each line in the output file is formatted as:
            ```
            <relative_path>,-,bonfadide,unknown,VoxCeleb1_VoxCeleb1++VoxCeleb1_<folder>,unkown,bonafide,unknown,<parent_folder>
            ```
        - Files outside of `base_dir` (i.e., that cannot be relativized to it) are ignored.
    """
        
    search_path = Path(search_dir)
    base_path = Path(base_dir)
    
    with open(output_file, 'w') as out_file:
        for file_path in search_path.rglob("*"):
            if file_path.is_file():
                try:
                    rel_path = file_path.relative_to(base_path)
                except ValueError:
                    continue

                folder = file_path.parent.name
                parent_folder = file_path.parent.parent.name

                line = f"{rel_path},-,bonfadide,unknown,VoxCeleb1_VoxCeleb1++VoxCeleb1_{folder},unkown,bonafide,unknown,{parent_folder}"
                out_file.write(line + '\n')


if __name__ == "__main__":
    create_labels_file("your/search/directory", "your/base/directory")
