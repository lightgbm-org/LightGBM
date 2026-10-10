from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import lightgbm as lgb


class HDFSequence(lgb.Sequence):
    def __init__(self, hdf_dataset, batch_size):
        """
        Construct a sequence object from HDF5 with required interface.

        Parameters
        ----------
        hdf_dataset : h5py.Dataset
            Dataset in HDF5 file.
        batch_size : int
            Size of a batch. When reading data to construct lightgbm Dataset, each read reads batch_size rows.
        """
        # We can also open HDF5 file once and get access to
        self.data = hdf_dataset
        self.batch_size = batch_size

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def create_dataset_from_multiple_hdf(input_flist, batch_size):
    data = []
    ylist = []
    for f in input_flist:
        f = h5py.File(f, "r")
        data.append(HDFSequence(f["X"], batch_size))
        ylist.append(f["Y"][:])

    params = {
        "bin_construct_sample_cnt": 200000,
        "max_bin": 255,
    }
    y = np.concatenate(ylist)
    dataset = lgb.Dataset(data, label=y, params=params)
    # With binary dataset created, we can use either Python API or cmdline version to train.
    #
    # Note: in order to create exactly the same dataset with the one created in simple_example.py, we need
    # to modify simple_example.py to pass numpy array instead of pandas DataFrame to Dataset constructor.
    # The reason is that DataFrame column names will be used in Dataset. For a DataFrame with Int64Index
    # as columns, Dataset will use column names like ["0", "1", "2", ...]. While for numpy array, column names
    # are using the default one assigned in C++ code (dataset_loader.cpp), like ["Column_0", "Column_1", ...].
    dataset.save_binary("regression.train.from_hdf.bin")


def save2hdf(input_data, fname, batch_size):
    """Store numpy array to HDF5 file.

    Please note chunk size settings in the implementation for I/O performance optimization.
    """
    with h5py.File(fname, "w") as f:
        for name, data in input_data.items():
            nrow, ncol = data.shape
            if ncol == 1:
                # Y has a single column and we read it in single shot. So store it as an 1-d array.
                chunk = (nrow,)
                data = data.values.flatten()
            else:
                # We use random access for data sampling when creating LightGBM Dataset from Sequence.
                # When accessing any element in a HDF5 chunk, it's read entirely.
                # To save I/O for sampling, we should keep number of total chunks much larger than sample count.
                # Here we are just creating a chunk size that matches with batch_size.
                #
                # Also note that the data is stored in row major order to avoid extra copy when passing to
                # lightgbm Dataset.
                chunk = (batch_size, ncol)
            f.create_dataset(name, data=data, chunks=chunk, compression="lzf")


def generate_hdf(input_fname, output_basename, batch_size):
    # Save to 2 HDF5 files for demonstration.
    df = pd.read_csv(input_fname, header=None, sep="\t")

    mid = len(df) // 2
    df1 = df.iloc[:mid]
    df2 = df.iloc[mid:]

    # We can store multiple datasets inside a single HDF5 file.
    # Separating X and Y for choosing best chunk size for data loading.
    fname1 = f"{output_basename}1.h5"
    fname2 = f"{output_basename}2.h5"
    save2hdf({"Y": df1.iloc[:, :1], "X": df1.iloc[:, 1:]}, fname1, batch_size)
    save2hdf({"Y": df2.iloc[:, :1], "X": df2.iloc[:, 1:]}, fname2, batch_size)

    return [fname1, fname2]


def main():
    batch_size = 64
    output_basename = "regression"
    hdf_files = generate_hdf(
        str(Path(__file__).absolute().parents[1] / "regression" / "regression.train"), output_basename, batch_size
    )

    create_dataset_from_multiple_hdf(hdf_files, batch_size=batch_size)


if __name__ == "__main__":
    main()
