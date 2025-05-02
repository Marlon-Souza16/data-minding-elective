import pandas as pd

def read_csv_files(file_path, sep=';', encoding='latin1', on_bad_lines='skip'):
    df = pd.read_csv(
        file_path,
        sep=sep,
        encoding=encoding,
        on_bad_lines=on_bad_lines
    )
    return df