import pandas as pd
import glob
import os
from tqdm import tqdm

CSV = ["csv", "txt"]
PARQUET = ["parquet"]
PICKLE = ["pkl", "pickle"]

pandas_read = {
    "csv": pd.read_csv,
    "parquet": pd.read_parquet,
    "pickle": pd.read_pickle
}

pandas_write = {
    "csv": "to_csv",
    "parquet": "to_parquet",
    "pickle": "to_pickle"
}


def get_format(extension):
    file_format = None
    if extension in CSV:
        file_format = "csv"
    if extension in PARQUET:
        file_format = "parquet"
    if extension in PICKLE:
        file_format = "pickle"

    if file_format is None:
        raise Exception(f"*.{extension} extension not supported")
    return file_format


def read_file(path, file_format=None, **kwargs):
    if os.path.isdir(path):
        raise Exception("Path has to be a directory")

    filename = os.path.basename(path)
    dirpath = os.path.dirname(path)
    extension = filename.split(".")[-1]

    if file_format is None:
        file_format = get_format(extension)
    df = pandas_read[file_format](path,**kwargs)

    return df


def write_file(df, path, file_format=None, **kwargs):
    if os.path.isdir(path):
        raise Exception("Path has to be a directory")

    filename = os.path.basename(path)
    dirpath = os.path.dirname(path)
    extension = filename.split(".")[-1]

    if file_format is None:
        file_format = get_format(extension)

    writer = getattr(df, pandas_write[file_format])
    writer(path, **kwargs)


def merge_df(input_dir, file_format=None, **kwargs):
    if not os.path.exists(input_dir):
        raise Exception("Input dir doesn't exists")

    print("Start merging files")
    input_files = glob.glob(os.path.join(input_dir,"*"))
    input_files.sort()
    dfs = []
    for file in tqdm(input_files):
        fname, extension = file.split(".")
        df = read_file(file, extension)
        dfs.append(df)
    df_all = pd.concat(dfs)
    return df_all
