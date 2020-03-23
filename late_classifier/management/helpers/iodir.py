import glob
import os


def list_files(path_to_files):
    files = []
    if os.path.isdir(path_to_files):
        files = glob.glob(os.path.join(path_to_files, "*"))
        files.sort()
    elif os.path.isfile(path_to_files):
        files = [path_to_files]
    return files


def exists_dir(directory, create=True):
    if not os.path.exists(directory):
        try:
            if create:
                os.makedirs(directory)
        except Exception as e:
            print(f"Exception: {e}")
            pass
        return False
    return True


def get_filename(path_to_file):
    name = path_to_file.split("/")[-1]
    name = name.split(".")[0]
    return name
