from typing import Callable, Iterable
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed, parallel_backend


# %%

class ProgressParallel(Parallel):

    # https://stackoverflow.com/a/61900501/6025592

    def __init__(self,
                 *args,
                 use_tqdm=True,
                 total=None,
                 **kwargs):

        self._use_tqdm = use_tqdm
        self._total = total
        self._pbar = None
        super().__init__(*args, **kwargs)

    def __call__(self,
                 *args,
                 **kwargs):

        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):

        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


class ParallelMixin:

    @staticmethod
    def do_parallel(
            _fun: Callable,
            _itr: Iterable,
            concatenate_result: bool = True,
            **kwargs: dict) -> np.array:
        """
        Applies a function to each element of an iterable in parallel and
        returns the results as a numpy array.

        Args:
        _fun (Callable): The function to apply to each element of the iterable.
        _itr (Iterable): The iterable to apply the function to.
        concatenate_result (bool, optional): If True, concatenates the results
        along the second axis. Default is True.
        **kwargs (dict): Additional keyword arguments to pass to the function.

        Returns:
        np.array: The results of applying the function to the iterable,
        in a numpy array.

        Examples:
        def square(x, **kwargs):
            return x**2
        arr = [1, 2, 3, 4, 5]
        result = do_parallel(square, arr, concatenate_result=False)
        print(result)
        # Output: [1, 4, 9, 16, 25]
        """

        backend = kwargs.get('backend', 'threading')

        with parallel_backend(backend, n_jobs=-1):
            lst_processed = ProgressParallel()(
                delayed(_fun)(el, **kwargs)
                for el in _itr)

        if concatenate_result:
            return np.concatenate(
                [arr.reshape(-1, 1) for arr in lst_processed], axis=1)

        return lst_processed
