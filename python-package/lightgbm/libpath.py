# coding: utf-8
"""Find the path to LightGBM dynamic library files."""

import ctypes
from os import environ
from pathlib import Path
from platform import system
from typing import List

__all__: List[str] = []


def _find_lib_path() -> List[str]:
    """Find the path to LightGBM library files.

    Returns
    -------
    lib_path: list of str
       List of all found library paths to LightGBM.
    """
    curr_path = Path(__file__).resolve()
    dll_path = [
        curr_path.parents[1],
        curr_path.parents[0] / "bin",
        curr_path.parents[0] / "lib",
    ]
    if system() in ("Windows", "Microsoft"):
        dll_path.append(curr_path.parents[1] / "Release")
        dll_path.append(curr_path.parents[1] / "windows" / "x64" / "DLL")
        dll_path = [p / "lib_lightgbm.dll" for p in dll_path]
    elif system() == "Darwin":
        dll_path = [p / "lib_lightgbm.dylib" for p in dll_path]
    else:
        dll_path = [p / "lib_lightgbm.so" for p in dll_path]
    lib_path = [str(p) for p in dll_path if p.is_file()]
    if not lib_path:
        dll_path_joined = "\n".join(map(str, dll_path))
        raise Exception(f"Cannot find lightgbm library file in following paths:\n{dll_path_joined}")
    return lib_path


# we don't need lib_lightgbm while building docs
_LIB: ctypes.CDLL
if environ.get("LIGHTGBM_BUILD_DOC", "False") == "True":
    from unittest.mock import Mock  # isort: skip

    _LIB = Mock(ctypes.CDLL)  # type: ignore
else:
    _LIB = ctypes.cdll.LoadLibrary(_find_lib_path()[0])
