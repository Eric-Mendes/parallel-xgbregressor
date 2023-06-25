import dask
import dask_ml
import dask.array as da
from dasf.transforms.base import Transform
from dasf.datasets import Dataset
from dasf.utils.types import is_dask_array
import time
from dasf.utils.decorators import task_handler
import numpy as np

class CustomArraysToDataFrame(Transform):
    def __init__(self, shift: int, axis: int):
        self.shift = shift
        self.axis = axis

    def _from3d_to_2d(self, data):
        if not is_dask_array(data):
            data = da.array(data)

        return da.roll(data, shift=self.shift, axis=self.axis)

    def __transform_generic(self, X, y):
        assert len(X) == len(y), "Data and labels should have the same length."

        for x in X:
            data2d = self._from3d_to_2d(data=x)

        return data2d

    def _lazy_transform_cpu(self, X=None, **kwargs):
        X = list(kwargs.values())
        y = list(kwargs.keys())

        start = time.monotonic()
        result = self.__transform_generic(X, y)
        end = time.monotonic()
        print(f"{self.__class__.__name__} levou {end-start} segundos.")
        return result

class xCustomArraysToDataFrame(Transform):
    def _helper_fn(self, data):
        if not is_dask_array(data):
            data = da.array(data)

        return data[:, :-1]

    def __transform_generic(self, X, y):
        assert len(X) == len(y)

        for x in X:
            return self._helper_fn(data=x)

    def _lazy_transform_cpu(self, **kwargs):
        X = list(kwargs.values())
        y = list(kwargs.keys())
        
        start = time.monotonic()
        result = self.__transform_generic(X, y)
        end = time.monotonic()
        print(f"{self.__class__.__name__} levou {end-start} segundos.")
        return result

class yCustomArraysToDataFrame(Transform):
    def _helper_fn(self, data):
        if not is_dask_array(data):
            data = da.array(data)

        return data[:, -1]

    def __transform_generic(self, X, y):
        assert len(X) == len(y)

        for x in X:
            return self._helper_fn(data=x)

    def _lazy_transform_cpu(self, **kwargs):
        X = list(kwargs.values())
        y = list(kwargs.keys())

        start = time.monotonic()
        result = self.__transform_generic(X, y)
        end = time.monotonic()
        print(f"{self.__class__.__name__} levou {end-start} segundos.")
        return result

class ModelFileName(Transform):
    def __transform_generic(self, model_file_name):
        return model_file_name

    def _lazy_transform_cpu(self, model_file_name):
        start = time.monotonic()
        result = self.__transform_generic(model_file_name)
        end = time.monotonic()
        print(f"{self.__class__.__name__} levou {end-start} segundos.")
        return result
