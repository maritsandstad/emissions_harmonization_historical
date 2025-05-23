"""
Support for our testing

You probably will never use this as a user of the package.

It's not in the tests folder,
because we don't put `__init__.py` files in our `tests`
because we've been bitten
by the subtleties of that too many times
(further reading here:
https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html#choosing-an-import-mode).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pandas_indexing as pix  # type: ignore

from gcages.io import load_timeseries_csv

if TYPE_CHECKING:
    import _pytest

AR6_IPS: tuple[tuple[str, str], ...] = (
    ("AIM/CGE 2.2", "EN_NPi2020_900f"),
    ("COFFEE 1.1", "EN_NPi2020_400f_lowBECCS"),
    ("GCAM 5.3", "NGFS2_Current Policies"),
    ("IMAGE 3.0", "EN_INDCi2030_3000f"),
    ("MESSAGEix-GLOBIOM 1.0", "LowEnergyDemand_1.3_IPCC"),
    ("MESSAGEix-GLOBIOM_GEI 1.0", "SSP2_openres_lc_50"),
    ("REMIND-MAgPIE 2.1-4.2", "SusDev_SDP-PkBudg1000"),
    ("REMIND-MAgPIE 2.1-4.3", "DeepElec_SSP2_ HighRE_Budg900"),
    ("WITCH 5.0", "CO_Bridge"),
)
"""
AR6 illustrative pathway scenarios

In each sub-tuple, the first element is the model, second is the scenario name.
"""


@functools.cache
def get_all_model_scenarios(filepath: Path) -> pd.DataFrame:
    """
    Load the file with all the model-scenario combinations

    Parameters
    ----------
    filepath
        File to load from

    Returns
    -------
    :
        Loaded model-scenario combinations
    """
    model_scenarios = pd.read_csv(filepath)

    return model_scenarios


def create_model_scenario_test_cases(
    model_scenarios: pd.DataFrame,
) -> tuple[_pytest.mark.structures.ParameterSet, ...]:
    """
    Create a test case for each model-scenario combination

    Parameters
    ----------
    model_scenarios
        Model-scenarios combinations from which to generate the tests cases

    Returns
    -------
    :
        Created test cases
    """
    # Late import to avoid explosions if pytest isn't installed
    import pytest

    return tuple(
        pytest.param(model, scenario, id=f"{model}__{scenario}")
        for (model, scenario), _ in model_scenarios.groupby(["Model", "Scenario"])
    )


@functools.cache
def get_ar6_all_emissions(
    model: str, scenario: str, test_data_dir: Path
) -> pd.DataFrame:
    """
    Get all emissions from AR6 for a given model-scenario

    Parameters
    ----------
    model
        Model

    scenario
        Scenario

    test_data_dir
        Test data directory where the data is saved

    Returns
    -------
    :
        All emissions from AR6 for `model`-`scenario`
    """
    filename_emissions = f"ar6_scenarios__{model}__{scenario}__emissions.csv"
    filename_emissions = filename_emissions.replace("/", "_").replace(" ", "_")
    emissions_file = test_data_dir / filename_emissions

    res = load_timeseries_csv(
        emissions_file,
        index_columns=["model", "scenario", "variable", "region", "unit"],
        out_column_type=int,
    )

    return res


@functools.cache
def get_ar6_raw_emissions(
    model: str, scenario: str, test_data_dir: Path
) -> pd.DataFrame:
    """
    Get all raw emissions from AR6 for a given model-scenario

    Parameters
    ----------
    model
        Model

    scenario
        Scenario

    test_data_dir
        Test data directory where the data is saved

    Returns
    -------
    :
        All raw emissions from AR6 for `model`-`scenario`
    """
    all_emissions = get_ar6_all_emissions(
        model=model, scenario=scenario, test_data_dir=test_data_dir
    )
    res: pd.DataFrame = all_emissions.loc[pix.ismatch(variable="Emissions**")].dropna(
        how="all", axis="columns"
    )

    return res


@functools.cache
def get_ar6_harmonised_emissions(
    model: str, scenario: str, test_data_dir: Path
) -> pd.DataFrame:
    """
    Get all harmonised emissions from AR6 for a given model-scenario

    Parameters
    ----------
    model
        Model

    scenario
        Scenario

    test_data_dir
        Test data directory where the data is saved

    Returns
    -------
    :
        All harmonised emissions from AR6 for `model`-`scenario`
    """
    all_emissions = get_ar6_all_emissions(
        model=model, scenario=scenario, test_data_dir=test_data_dir
    )
    res: pd.DataFrame = all_emissions.loc[
        pix.ismatch(variable="**Harmonized**")
    ].dropna(how="all", axis="columns")

    return res


@functools.cache
def get_ar6_infilled_emissions(
    model: str, scenario: str, test_data_dir: Path
) -> pd.DataFrame:
    """
    Get all infilled emissions from AR6 for a given model-scenario

    Parameters
    ----------
    model
        Model

    scenario
        Scenario

    test_data_dir
        Test data directory where the data is saved

    Returns
    -------
    :
        All infilled emissions from AR6 for `model`-`scenario`
    """
    all_emissions = get_ar6_all_emissions(
        model=model, scenario=scenario, test_data_dir=test_data_dir
    )
    res: pd.DataFrame = all_emissions.loc[pix.ismatch(variable="**Infilled**")].dropna(
        how="all", axis="columns"
    )

    return res


@functools.cache
def get_ar6_temperature_outputs(
    model: str, scenario: str, test_data_dir: Path
) -> pd.DataFrame:
    """
    Get temperature outputs we've downloaded from AR6 for a given model-scenario

    Parameters
    ----------
    model
        Model

    scenario
        Scenario

    test_data_dir
        Test data directory where the data is saved

    Returns
    -------
    :
        All temperature outputs we've downloaded from AR6 for `model`-`scenario`
    """
    filename_temperatures = f"ar6_scenarios__{model}__{scenario}__temperatures.csv"
    filename_temperatures = filename_temperatures.replace("/", "_").replace(" ", "_")
    temperatures_file = test_data_dir / filename_temperatures

    res = load_timeseries_csv(
        temperatures_file,
        index_columns=["model", "scenario", "variable", "region", "unit"],
        out_column_type=int,
    )

    return res


@functools.cache
def get_ar6_metadata_outputs(test_data_dir: Path) -> pd.DataFrame:
    """
    Get metadata from AR6

    Parameters
    ----------
    test_data_dir
        Test data directory where the data is saved

    Returns
    -------
    :
        Metadata from AR6
    """
    filename = "AR6_Scenarios_Database_metadata_indicators_v1.1_meta.csv"

    res = load_timeseries_csv(
        test_data_dir / filename,
        index_columns=["model", "scenario"],
    )

    return res


def assert_frame_equal(
    res: pd.DataFrame, exp: pd.DataFrame, rtol: float = 1e-8, **kwargs: Any
) -> None:
    """
    Assert two `pd.DataFrame`'s are equal.

    This is a very thin wrapper around
    [`pd.testing.assert_frame_equal`][pandas.testing.assert_frame_equal]
    that makes some use of [`pandas_indexing`][pandas_indexing]
    to give slightly nicer and clearer errors.

    Parameters
    ----------
    res
        Result

    exp
        Expected value

    rtol
        Relative tolerance

    **kwargs
        Passed to [`pd.testing.assert_frame_equal`][pandas.testing.assert_frame_equal]

    Raises
    ------
    AssertionError
        The frames aren't equal
    """
    for idx_name in res.index.names:
        idx_diffs = res.pix.unique(idx_name).symmetric_difference(  # type: ignore
            exp.pix.unique(idx_name)  # type: ignore
        )
        if not idx_diffs.empty:
            msg = f"Differences in the {idx_name} (res on the left): {idx_diffs=}"
            raise AssertionError(msg)

    pd.testing.assert_frame_equal(
        res.T, exp.T, check_like=True, check_exact=False, rtol=rtol, **kwargs
    )
