# coding: utf-8
import ctypes
from pathlib import Path
from platform import system

import numpy as np
from scipy import sparse

try:
    from lightgbm.basic import _LIB as LIB
except ModuleNotFoundError:
    print("Could not import lightgbm Python-package, looking for lib_lightgbm at the repo root")
    if system() in ("Windows", "Microsoft"):
        lib_file = Path(__file__).absolute().parents[2] / "Release" / "lib_lightgbm.dll"
    else:
        lib_file = Path(__file__).absolute().parents[2] / "lib_lightgbm.so"
    LIB = ctypes.cdll.LoadLibrary(lib_file)

LIB.LGBM_GetLastError.restype = ctypes.c_char_p

dtype_float32 = 0
dtype_float64 = 1
dtype_int32 = 2
dtype_int64 = 3


def c_str(string):
    return ctypes.c_char_p(str(string).encode("utf-8"))


def load_from_file(filename, reference):
    ref = None
    if reference is not None:
        ref = reference
    handle = ctypes.c_void_p()
    LIB.LGBM_DatasetCreateFromFile(c_str(str(filename)), c_str("max_bin=15"), ref, ctypes.byref(handle))
    print(LIB.LGBM_GetLastError())
    num_data = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data))
    num_feature = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature))
    print(f"#data: {num_data.value} #feature: {num_feature.value}")
    return handle


def save_to_binary(handle, filename):
    LIB.LGBM_DatasetSaveBinary(handle, c_str(filename))


def load_from_csr(filename, reference):
    data = np.loadtxt(str(filename), dtype=np.float64)
    csr = sparse.csr_matrix(data[:, 1:])
    label = data[:, 0].astype(np.float32)
    handle = ctypes.c_void_p()
    ref = None
    if reference is not None:
        ref = reference

    LIB.LGBM_DatasetCreateFromCSR(
        csr.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(dtype_int32),
        csr.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        csr.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(dtype_float64),
        ctypes.c_int64(len(csr.indptr)),
        ctypes.c_int64(len(csr.data)),
        ctypes.c_int64(csr.shape[1]),
        c_str("max_bin=15"),
        ref,
        ctypes.byref(handle),
    )
    num_data = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data))
    num_feature = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature))
    LIB.LGBM_DatasetSetField(
        handle,
        c_str("label"),
        label.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(len(label)),
        ctypes.c_int(dtype_float32),
    )
    print(f"#data: {num_data.value} #feature: {num_feature.value}")
    return handle


def load_from_csc(filename, reference):
    data = np.loadtxt(str(filename), dtype=np.float64)
    csc = sparse.csc_matrix(data[:, 1:])
    label = data[:, 0].astype(np.float32)
    handle = ctypes.c_void_p()
    ref = None
    if reference is not None:
        ref = reference

    LIB.LGBM_DatasetCreateFromCSC(
        csc.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(dtype_int32),
        csc.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        csc.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(dtype_float64),
        ctypes.c_int64(len(csc.indptr)),
        ctypes.c_int64(len(csc.data)),
        ctypes.c_int64(csc.shape[0]),
        c_str("max_bin=15"),
        ref,
        ctypes.byref(handle),
    )
    num_data = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data))
    num_feature = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature))
    LIB.LGBM_DatasetSetField(
        handle,
        c_str("label"),
        label.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(len(label)),
        ctypes.c_int(dtype_float32),
    )
    print(f"#data: {num_data.value} #feature: {num_feature.value}")
    return handle


def load_from_mat(filename, reference):
    mat = np.loadtxt(str(filename), dtype=np.float64)
    label = mat[:, 0].astype(np.float32)
    mat = mat[:, 1:]
    data = np.asarray(mat.reshape(mat.size), dtype=np.float64)
    handle = ctypes.c_void_p()
    ref = None
    if reference is not None:
        ref = reference

    LIB.LGBM_DatasetCreateFromMat(
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(dtype_float64),
        ctypes.c_int32(mat.shape[0]),
        ctypes.c_int32(mat.shape[1]),
        ctypes.c_int(1),
        c_str("max_bin=15"),
        ref,
        ctypes.byref(handle),
    )
    num_data = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumData(handle, ctypes.byref(num_data))
    num_feature = ctypes.c_int(0)
    LIB.LGBM_DatasetGetNumFeature(handle, ctypes.byref(num_feature))
    LIB.LGBM_DatasetSetField(
        handle,
        c_str("label"),
        label.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(len(label)),
        ctypes.c_int(dtype_float32),
    )
    print(f"#data: {num_data.value} #feature: {num_feature.value}")
    return handle


def free_dataset(handle):
    LIB.LGBM_DatasetFree(handle)


def test_dataset(tmp_path):
    binary_example_dir = Path(__file__).absolute().parents[2] / "examples" / "binary_classification"
    train = load_from_file(binary_example_dir / "binary.train", None)
    test = load_from_mat(binary_example_dir / "binary.test", train)
    free_dataset(test)
    test = load_from_csr(binary_example_dir / "binary.test", train)
    free_dataset(test)
    test = load_from_csc(binary_example_dir / "binary.test", train)
    free_dataset(test)
    train_binary = str(tmp_path / "train.binary.bin")
    save_to_binary(train, train_binary)
    free_dataset(train)
    train = load_from_file(train_binary, None)
    free_dataset(train)


def test_booster(tmp_path):
    binary_example_dir = Path(__file__).absolute().parents[2] / "examples" / "binary_classification"
    train = load_from_mat(binary_example_dir / "binary.train", None)
    test = load_from_mat(binary_example_dir / "binary.test", train)
    booster = ctypes.c_void_p()
    model_path = tmp_path / "model.txt"
    LIB.LGBM_BoosterCreate(train, c_str("app=binary metric=auc num_leaves=31 verbose=0"), ctypes.byref(booster))
    LIB.LGBM_BoosterAddValidData(booster, test)
    produced_empty_tree = ctypes.c_int(0)
    for i in range(1, 51):
        LIB.LGBM_BoosterUpdateOneIter(booster, ctypes.byref(produced_empty_tree))
        result = np.array([0.0], dtype=np.float64)
        out_len = ctypes.c_int(0)
        LIB.LGBM_BoosterGetEval(
            booster, ctypes.c_int(0), ctypes.byref(out_len), result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        if i % 10 == 0:
            print(f"{i} iteration test AUC {result[0]:.6f}")
    LIB.LGBM_BoosterSaveModel(booster, ctypes.c_int(0), ctypes.c_int(-1), ctypes.c_int(0), c_str(str(model_path)))
    LIB.LGBM_BoosterFree(booster)
    free_dataset(train)
    free_dataset(test)
    booster2 = ctypes.c_void_p()
    num_total_model = ctypes.c_int(0)
    LIB.LGBM_BoosterCreateFromModelfile(c_str(str(model_path)), ctypes.byref(num_total_model), ctypes.byref(booster2))
    data = np.loadtxt(str(binary_example_dir / "binary.test"), dtype=np.float64)
    mat = data[:, 1:]
    preds = np.empty(mat.shape[0], dtype=np.float64)
    num_preds = ctypes.c_int64(0)
    data = np.asarray(mat.reshape(mat.size), dtype=np.float64)
    LIB.LGBM_BoosterPredictForMat(
        booster2,
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(dtype_float64),
        ctypes.c_int32(mat.shape[0]),
        ctypes.c_int32(mat.shape[1]),
        ctypes.c_int(1),
        ctypes.c_int(1),
        ctypes.c_int(0),
        ctypes.c_int(25),
        c_str(""),
        ctypes.byref(num_preds),
        preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    LIB.LGBM_BoosterPredictForFile(
        booster2,
        c_str(str(binary_example_dir / "binary.test")),
        ctypes.c_int(0),
        ctypes.c_int(0),
        ctypes.c_int(0),
        ctypes.c_int(25),
        c_str(""),
        c_str(tmp_path / "preds.txt"),
    )
    LIB.LGBM_BoosterPredictForFile(
        booster2,
        c_str(str(binary_example_dir / "binary.test")),
        ctypes.c_int(0),
        ctypes.c_int(0),
        ctypes.c_int(10),
        ctypes.c_int(25),
        c_str(""),
        c_str(tmp_path / "preds.txt"),
    )
    LIB.LGBM_BoosterFree(booster2)


def test_max_thread_control():
    # at initialization, should be -1
    num_threads = ctypes.c_int(0)
    ret = LIB.LGBM_GetMaxThreads(ctypes.byref(num_threads))
    assert ret == 0
    assert num_threads.value == -1

    # updating that value through the C API should work
    ret = LIB.LGBM_SetMaxThreads(ctypes.c_int(6))
    assert ret == 0

    ret = LIB.LGBM_GetMaxThreads(ctypes.byref(num_threads))
    assert ret == 0
    assert num_threads.value == 6

    # resetting to any negative number should set it to -1
    ret = LIB.LGBM_SetMaxThreads(ctypes.c_int(-123))
    assert ret == 0
    ret = LIB.LGBM_GetMaxThreads(ctypes.byref(num_threads))
    assert ret == 0
    assert num_threads.value == -1
