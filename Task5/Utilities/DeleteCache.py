import os
import shutil

def delete_pycache(root_dir):
    """
    Recursively delete all __pycache__ directories from the given root directory.
    
    Args:
        root_dir (str): The root directory to start searching from.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == '__pycache__':
                pycache_path = os.path.join(dirpath, dirname)
                print(f"Deleting: {pycache_path}")
                shutil.rmtree(pycache_path)
                print(f"Deleted: {pycache_path}")

if __name__ == "__main__":
    project_dir = "./"
    delete_pycache(project_dir)
    print("All __pycache__ directories have been deleted.")