"""
Support for parallelisation

This always feels so much harder than it should be
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
from collections.abc import Iterable
from multiprocessing.context import BaseContext
from typing import Callable, TypeVar

import pandas as pd
import tqdm.auto
from loguru import logger
from typing_extensions import Concatenate, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")
U = TypeVar("U")


def assert_only_working_on_variable_unit_variations(indf: pd.DataFrame) -> None:
    """
    Assert that we're only working on variations in variable and unit

    In other words, we don't have variations in scenarios, models etc.

    Parameters
    ----------
    indf
        Data to verify

    Raises
    ------
    AssertionError
        There are variations in columns other than variable and unit
    """
    non_v_u_cols = list(set(indf.index.names).difference(["variable", "unit"]))
    variations_in_other_cols: pd.MultiIndex = indf.pix.unique(non_v_u_cols)  # type: ignore

    if variations_in_other_cols.shape[0] > 1:
        raise AssertionError(f"{variations_in_other_cols=}")


def run_parallel(
    func_to_call: Callable[Concatenate[U, P], T],
    iterable_input: Iterable[U],
    input_desc: str,
    n_processes: int,
    mp_context: BaseContext | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[T, ...]:
    """
    Run a function in parallel

    Yet another abstraction for this,
    because the ones we had weren't doing what we wanted.

    Parameters
    ----------
    func_to_call
        Function to call

    iterable_input
        Input with which to call the function.

    input_desc
        Description of the input (used to make the progress bars more helpful)

    n_processes
        Number of processes to use during the processing.

        If set to `1`, we run the process serially
        (very helpful for debugging).

    mp_context
        Multiprocessing context to use.

        By default, we use a fork context if available.
        If forking is not available, we use the system's default.
        If `None`, we will revert to the default if `n_processes` is greater than 1.

        The whole multiprocessing context universe is a bit complex,
        particularly given we also have logging.
        In short, spawn is slower, but safer and is supported by windows.
        Yet forking seems to be the only thing
        that allows our logging to come through without issue
        (although maybe we're doing something wrong, it's a bit unclear).
        Full docs on multiprocessing contexts are here:
        https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods.

    *args
        Arguments to use for every call of `func_to_call`.

    **kwargs
        Keyword arguments to use for every call of `func_to_call`.

    Returns
    -------
    :
        Result of calling `func_to_call` with every element in `iterable_input`
        in combination with `args` and `kwargs`.
    """
    if n_processes == 1:
        logger.debug("Running serially")
        res = [
            func_to_call(inv, *args, **kwargs)
            for inv in tqdm.auto.tqdm(iterable_input, desc=input_desc)
        ]

    else:
        if mp_context is None:
            try:
                mp_context = multiprocessing.get_context("fork")
            except ValueError:
                mp_context = multiprocessing.get_context()

        logger.info(f"Submitting {input_desc} to {n_processes} parallel processes")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_processes, mp_context=mp_context
        ) as executor:
            futures = [
                executor.submit(
                    func_to_call,
                    inv,
                    *args,
                    **kwargs,
                )
                for inv in tqdm.auto.tqdm(
                    iterable_input, desc=f"Submitting {input_desc} to queue"
                )
            ]

            res = [
                future.result()
                for future in tqdm.auto.tqdm(
                    concurrent.futures.as_completed(futures),
                    desc="Retrieving parallel results",
                    total=len(futures),
                )
            ]

    return tuple(res)
