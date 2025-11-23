# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Run a simple climate model
#
# Here we run a simple climate model...

# %% [markdown]
# ## Imports

# %%
import logging
import multiprocessing
import os
import platform
import tempfile
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import openscm_units
import pandas_indexing as pix
import pandas_openscm
import seaborn as sns
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.scm_running import run_scms
from pandas_openscm.index_manipulation import update_index_levels_func

from emissions_harmonization_historical.ciceroscm_utils import (
    ciceroscm_output_to_dataframe,
    dataframe_to_ciceroscm_emissions,
)
from emissions_harmonization_historical.constants_5000 import (
    HISTORY_HARMONISATION_DB,
    INFILLED_SCENARIOS_DB,
    RCMIP_PROCESSED_DB,
    REPO_ROOT,
    SCM_OUT_DIR,
    SCM_OUTPUT_DB,
)
from emissions_harmonization_historical.scm_running import (
    get_complete_scenarios_for_magicc,
    load_magicc_cfgs,
)

# Suppress expected MAGICC warnings about extending solar forcing to 2500
# The solar RF data is already extended to 2500. but MAGICC's Fortran code
# warns that it's using extrapolated (not observed) data beyond 2100.
warnings.filterwarnings("ignore", message=".*Extending solar RF.*")
warnings.filterwarnings("ignore", message=".*magicc logged a WARNING message.*")
warnings.filterwarnings(
    "ignore", message=r"magicc logged a WARNING message\. Check the 'stderr' key.*", category=UserWarning
)
warnings.filterwarnings("ignore", category=UserWarning, module="pymagicc.core")
# Also suppress at the source
logging.getLogger("pymagicc").setLevel(logging.ERROR)

# %% [markdown]
# ## Set up

# %%
pandas_openscm.register_pandas_accessor()

# %%
UR = openscm_units.unit_registry
Q = UR.Quantity

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
model: str = "AIM"
scm: str = "MAGICCv7.6.0a3"
scm: str = "CICEROSCM"  # instead of "MAGICCv7.6.0a3"

# %%
output_dir_model = SCM_OUT_DIR / model
output_dir_model.mkdir(exist_ok=True, parents=True)
output_dir_model

# %% [markdown]
# ## Load data

# %% [markdown]
# ### Complete scenarios (extended to 2500)

# %%
# Load extended scenarios (1750-2500) for climate model runs
# These are the 7 marker scenarios extended beyond 2100
complete_scenarios = INFILLED_SCENARIOS_DB.load(
    pix.isin(stage="extended") & pix.ismatch(model=f"*{model}*")
).reset_index("stage", drop=True)

# Filter out internal diagnostic variables that aren't part of CMIP7 naming convention
internal_variables = [
    "Emissions|CO2|Gross Positive Emissions",
    "Emissions|CO2|Gross Removals",
]
complete_scenarios = complete_scenarios.loc[
    ~complete_scenarios.index.get_level_values("variable").isin(internal_variables)
]

# %%
# Check year range to verify we have extended scenarios
print(f"Year range in complete_scenarios: {complete_scenarios.columns.min()} to {complete_scenarios.columns.max()}")
print(f"Number of scenarios: {len(complete_scenarios.pix.unique('scenario'))}")
print(f"Scenarios: {list(complete_scenarios.pix.unique('scenario'))}")

# %% [markdown]
# ### History
#
# Just in case we need it for MAGICC

# %%
# TODO: make the db portable
# history = pd.concat([
#     pd.read_feather(f) for f in HISTORY_HARMONISATION_DB.db_dir.glob("*.feather")
#     if "index" not in f.name and "filemap" not in f.name
# ]).loc[pix.isin(purpose="global_workflow_emissions")].reset_index("purpose", drop=True)
# history

# %%
history = HISTORY_HARMONISATION_DB.load(pix.ismatch(purpose="global_workflow_emissions")).reset_index(
    "purpose", drop=True
)

# history.loc[:, :2023]

# %% [markdown]
# ## Configure SCM

# %%
output_variables = (
    # GSAT
    "Surface Air Temperature Change",
    # # GMST
    "Surface Air Ocean Blended Temperature Change",
    # # ERFs
    "Effective Radiative Forcing",
    "Effective Radiative Forcing|Anthropogenic",
    "Effective Radiative Forcing|Aerosols",
    "Effective Radiative Forcing|Aerosols|Direct Effect",
    "Effective Radiative Forcing|Aerosols|Direct Effect|BC",
    "Effective Radiative Forcing|Aerosols|Direct Effect|OC",
    "Effective Radiative Forcing|Aerosols|Direct Effect|SOx",
    "Effective Radiative Forcing|Aerosols|Indirect Effect",
    "Effective Radiative Forcing|Greenhouse Gases",
    "Effective Radiative Forcing|CO2",
    "Effective Radiative Forcing|CH4",
    "Effective Radiative Forcing|N2O",
    "Effective Radiative Forcing|F-Gases",
    "Effective Radiative Forcing|Montreal Protocol Halogen Gases",
    "Effective Radiative Forcing|Ozone",
    "Effective Radiative Forcing|Tropospheric Ozone",
    "Effective Radiative Forcing|Stratospheric Ozone",
    "Effective Radiative Forcing|Solar",
    "Effective Radiative Forcing|Volcanic",
    # # Heat uptake
    "Heat Uptake",
    "Heat Uptake|Ocean",
    # # Atmospheric concentrations
    "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|CH4",
    "Atmospheric Concentrations|N2O",
    # # Carbon cycle
    # "Net Atmosphere to Land Flux|CO2",
    # "Net Atmosphere to Ocean Flux|CO2",
    # "CO2_CURRENT_NPP",
    # # Permafrost
    # "Net Land to Atmosphere Flux|CO2|Earth System Feedbacks|Permafrost",
    # "Net Land to Atmosphere Flux|CH4|Earth System Feedbacks|Permafrost",
    "Sea Level Rise",
)

# %%
if scm in ["MAGICCv7.5.3", "MAGICCv7.6.0a3"]:
    if scm == "MAGICCv7.6.0a3":
        if platform.system() == "Darwin":
            if platform.processor() == "arm":
                magicc_exe_path = REPO_ROOT / "magicc" / "magicc-v7.6.0a3" / "bin" / "magicc-darwin-arm64"
            else:
                raise NotImplementedError(platform.processor())
        elif platform.system() == "Windows":
            raise NotImplementedError(platform.system())
        elif platform.system().lower().startswith("linux"):
            magicc_exe_path = REPO_ROOT / "magicc" / "magicc-v7.6.0a3" / "bin" / "magicc"
            # Set library path for GCC 13.3.0 (MAGICC was built with this version)
            # This ensures gfortran libraries are found even when modules aren't loaded
            gcc_lib_path = "/opt/software/easybuild/software/GCCcore/13.3.0/lib64"
            if "LD_LIBRARY_PATH" in os.environ:
                os.environ["LD_LIBRARY_PATH"] = f"{gcc_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
            else:
                os.environ["LD_LIBRARY_PATH"] = gcc_lib_path

            # Use /scratch instead of /tmp for MAGICC worker temporary directories
            # /tmp is only 10 GB and fills up with 32 parallel MAGICC processes
            os.environ["MAGICC_WORKER_ROOT_DIR"] = "/scratch/bensan"
        else:
            raise NotImplementedError(platform.system())

        magicc_expected_version = "v7.6.0a3"
        magicc_prob_distribution_path = (
            REPO_ROOT / "magicc" / "magicc-v7.6.0a3" / "configs" / "magicc-ar7-fast-track-drawnset-v0-3-0.json"
        )

    elif scm == "MAGICCv7.5.3":
        os.environ["DYLD_LIBRARY_PATH"] = "/opt/homebrew/opt/gfortran/lib/gcc/current/"
        if platform.system() == "Darwin":
            if platform.processor() == "arm":
                magicc_exe = "magicc-darwin-arm64"
            else:
                raise NotImplementedError(platform.processor())
        elif platform.system() == "Windows":
            magicc_exe = "magicc.exe"
        elif platform.system().lower().startswith("linux"):
            magicc_exe = "magicc"
        else:
            raise NotImplementedError(platform.system())

        magicc_exe_path = REPO_ROOT / "magicc" / "magicc-v7.5.3" / "bin" / magicc_exe
        magicc_expected_version = "v7.5.3"
        magicc_prob_distribution_path = REPO_ROOT / "magicc" / "magicc-v7.5.3" / "configs" / "600-member.json"

    else:
        raise NotImplementedError(scm)

    os.environ["MAGICC_EXECUTABLE_7"] = str(magicc_exe_path)

    climate_models_cfgs = load_magicc_cfgs(
        prob_distribution_path=magicc_prob_distribution_path,
        output_variables=output_variables,
        startyear=1750,
    )

    complete_scm = get_complete_scenarios_for_magicc(
        scenarios=complete_scenarios,
        history=history,
    )

    # Convert year columns from float to int to avoid MAGICC namelist errors
    # MAGICC's Fortran namelist reader expects integer years, not floats
    complete_scm.columns = complete_scm.columns.astype(int)

elif scm.startswith("CICERO"):
    # CICERO-SCM doesn't need the historical data prepended the same way MAGICC does
    # It can start from any year in the scenario data
    complete_scm = complete_scenarios.copy()

    # CICERO-SCM will be run directly (not through openscm-runner)
    # So we don't need climate_models_cfgs in the same format
    climate_models_cfgs = None

    # DIAGNOSTIC: Print all available emissions variables
    print("\nAvailable emissions variables in complete_scm:")
    emissions_vars = sorted([v for v in complete_scm.pix.unique("variable") if v.startswith("Emissions|")])
    for var in emissions_vars:
        print(f"  {var}")
    print(f"\nTotal: {len(emissions_vars)} emissions variables\n")

else:
    raise NotImplementedError(f"SCM {scm} not supported")

# Check year range after preparation
if scm.startswith("MAGICC") or scm.startswith("CICERO"):
    print(f"Year range in complete_scm: {complete_scm.columns.min()} to {complete_scm.columns.max()}")


# %%


# complete_scm

# %%
if scm.startswith("MAGICC"):
    climate_models_cfgs["MAGICC7"][0]["out_dynamic_vars"]
elif scm.startswith("CICERO"):
    print(f"CICERO-SCM will output: {len(output_variables)} variables")

# %% [markdown]
# ### If MAGICC, check how yuck the jump will be
#
# Answer: not ideal but we're going to have to live with it.


# %%
if scm.startswith("MAGICC"):
    reporting_to_rcmip = partial(
        convert_variable_name,
        to_convention=SupportedNamingConventions.RCMIP,
        from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
    )
    rcmip_to_reporting = partial(
        convert_variable_name,
        from_convention=SupportedNamingConventions.RCMIP,
        to_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
    )

    rcmip_hist = RCMIP_PROCESSED_DB.load(
        pix.isin(
            region="World",
            scenario="ssp245",
            variable=complete_scenarios.pix.unique("variable").map(reporting_to_rcmip),
        ),
        progress=True,
    ).loc[:, 1990:2014]
    rcmip_hist = rcmip_hist.openscm.update_index_levels({"variable": rcmip_to_reporting})
    # rcmip_hist

    pdf = pix.concat([rcmip_hist, complete_scm]).loc[:, 1990:2030].openscm.to_long_data().dropna()
    # pdf

    fg = sns.relplot(
        data=pdf,
        x="time",
        y="value",
        hue="scenario",
        col="variable",
        col_order=sorted(pdf["variable"].unique()),
        col_wrap=4,
        kind="line",
        facet_kws=dict(sharey=False),
    )
    for ax in fg.axes.flatten():
        ax.set_ylim(ymin=0.0)

# %% [markdown]
# ## Run SCM

# %%
if scm.startswith("MAGICC"):
    complete_openscm_runner = update_index_levels_func(
        complete_scm,
        {
            "variable": partial(
                convert_variable_name,
                from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                to_convention=SupportedNamingConventions.OPENSCM_RUNNER,
            )
        },
    )
    complete_openscm_runner
elif scm.startswith("CICERO"):
    # CICERO-SCM will use CMIP7 ScenarioMIP names directly (converted in ciceroscm_utils)
    print(f"CICERO-SCM will process {len(complete_scm)} rows")


# %%
class db_hack:
    """Save files in groups while we can't pass groupby through the function below"""

    def __init__(self, actual_db):
        self.actual_db = actual_db

    def load_metadata(self, *args, **kwargs):
        """Load metadata"""
        return self.actual_db.load_metadata(*args, **kwargs)

    def save(self, *args, **kwargs):
        """Save"""
        return self.actual_db.save(
            *args,
            **kwargs,
            groupby=["model", "scenario", "variable"],
            allow_overwrite=True,
        )


# %%
db = db_hack(SCM_OUTPUT_DB)

# %%
# Limit parallel processes to avoid memory issues on high-core-count systems
# Each MAGICC process loads full scenario data, so too many processes causes OOM
# Rule of thumb: ~4-8 GB per MAGICC process for extended scenarios
max_processes = min(multiprocessing.cpu_count(), 32)  # Cap at 32 processes
print(f"Running with {max_processes} parallel processes (system has {multiprocessing.cpu_count()} cores)")

# %%
# Run the climate model
if scm.startswith("MAGICC"):
    # Use openscm-runner for MAGICC
    run_scms(
        scenarios=complete_openscm_runner,
        climate_models_cfgs=climate_models_cfgs,
        output_variables=output_variables,
        scenario_group_levels=["model", "scenario"],
        n_processes=max_processes,
        db=db,
        verbose=True,
        progress=True,
        batch_size_scenarios=15,
        force_rerun=True,  # CHANGED: Must re-run for extended scenarios to 2500
    )

elif scm.startswith("CICERO"):
    # Run CICERO-SCM directly
    import shutil
    import tempfile
    from pathlib import Path

    from ciceroscm import CICEROSCM

    print(f"Running CICERO-SCM for {len(complete_scm.pix.unique('scenario'))} scenarios")

    # Create temporary directory for emissions files
    temp_dir = Path(tempfile.mkdtemp(prefix="ciceroscm_"))

    # Get paths to CICERO-SCM test data for gas parameters and defaults
    ciceroscm_test_dir = REPO_ROOT / "ciceroscm" / "tests" / "test-data"

    try:
        # Process each scenario
        for scenario in complete_scm.pix.unique("scenario"):
            scenario_data = complete_scm.loc[complete_scm.index.get_level_values("scenario") == scenario]
            model_name = scenario_data.pix.unique("model")[0]

            print(f"Processing {model_name} - {scenario}")

            # Convert to CICERO-SCM format
            # Pad from 1700 (nystart) even though emissions start at 1750 (emstart)
            # This allows concentration-driven spinup period 1700-1749
            emissions_file = temp_dir / f"{model_name}_{scenario.replace(' ', '_')}.txt"
            cscm_df, metadata = dataframe_to_ciceroscm_emissions(
                scenario_data,
                output_file=emissions_file,
                start_year=1700,  # Pad with zeros from 1700 for concentration spinup
            )
            print(f"  Converted {metadata['n_components']} emission components")
            print(f"  Components: {metadata['components']}")

            # Get year range from metadata
            start_year, end_year = metadata["year_range"]

            # CICERO-SCM needs concentrations for the initial period before emissions start
            # The test concentration file starts at 1700, so we'll use it for the early period
            # and switch to emissions at 1750 (where our data begins)

            # Prepare land use change forcing data padded to match year range
            # Default file (IPCC_LUCalbedo.txt) runs 1750-2500, but we need 1700-2500
            import pandas as pd

            luc_forcing_file = ciceroscm_test_dir / "IPCC_LUCalbedo.txt"
            luc_data = np.loadtxt(luc_forcing_file)
            # Pad with 50 zeros for years 1700-1749
            luc_data_padded = np.concatenate([np.zeros(50), luc_data])
            # Convert to DataFrame (CICERO expects DataFrame with .iloc)
            luc_data_df = pd.DataFrame(luc_data_padded)

            # Initialize CICERO-SCM
            # Note: concentrations_file needed for initial period (nystart to emstart)
            # Following test pattern: nystart < emstart to allow concentration-driven spinup
            # Test concentration file starts at 1700, our emissions at 1750
            # Using v1RCMIP gas parameters (simpler set matching our emissions data)
            cscm = CICEROSCM(
                {
                    "gaspam_file": str(ciceroscm_test_dir / "gases_v1RCMIP.txt"),
                    "emissions_file": str(emissions_file),
                    "concentrations_file": str(ciceroscm_test_dir / "ssp245_conc_RCMIP.txt"),
                    "rf_luc_data": luc_data_df,  # Pass padded land use forcing as DataFrame
                    "nystart": 1700,  # Start earlier to use concentration data for spinup
                    "emstart": 1750,  # Switch to emissions where our data begins
                    "nyend": int(end_year),
                }
            )

            # Run CICERO-SCM using _run (not run_model) with results_as_dict
            cscm._run(
                {
                    "results_as_dict": True,
                    "carbon_cycle_outputs": True,
                },
                pamset_emiconc={
                    "qbmb": 0.0,
                    "qo3": 0.5,
                    "qdirso2": -0.00308,
                    "qindso2": -0.37 / 57.052577209999995,
                    "qbc": 0.0279,
                    "qoc": -0.00433,
                    "qh2o_ch4": 0.091915,
                    "ref_yr": 2010,
                },
            )

            print(f"  CICERO-SCM run completed, processing {len(cscm.results)} output variables")

            # Convert output to DataFrame format
            output_df = ciceroscm_output_to_dataframe(
                cscm_results=cscm.results,
                scenario_name=scenario,
                model_name=model_name,
                climate_model=scm,
                output_variables=output_variables,
                start_year=1700,  # Must match nystart in CICERO-SCM parameters
            )

            # Save to database
            db.save(output_df)
            print(f"  Saved {len(output_df.pix.unique('variable'))} variables to database")

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

else:
    raise NotImplementedError(f"SCM {scm} not supported")

# %%
# Check what was actually saved by run_scms to the database
scm_output_check = SCM_OUTPUT_DB.load(pix.ismatch(model=f"*{model}*", climate_model=f"*{scm}*"))
if not scm_output_check.empty:
    print(f"Year range in SCM_OUTPUT_DB: {scm_output_check.columns.min()} to {scm_output_check.columns.max()}")
    print(f"Variables in SCM_OUTPUT_DB: {sorted(scm_output_check.pix.unique('variable'))}")

    # DIAGNOSTIC: Check scenarios and their year ranges
    print("DIAGNOSTIC: Scenarios in database:")
    for scenario in sorted(scm_output_check.pix.unique("scenario")):
        scenario_data = scm_output_check.loc[scm_output_check.index.get_level_values("scenario") == scenario]
        temp_data = scenario_data.loc[
            scenario_data.index.get_level_values("variable") == "Surface Air Temperature Change"
        ]
        if not temp_data.empty:
            print(f"  {scenario}: {temp_data.columns.min()} to {temp_data.columns.max()}")

    # DIAGNOSTIC: Check year range for Surface Air Temperature Change specifically
    temp_var = scm_output_check.loc[
        scm_output_check.index.get_level_values("variable") == "Surface Air Temperature Change"
    ]
    if not temp_var.empty:
        print(
            f"DIAGNOSTIC: 'Surface Air Temperature Change' year range: "
            f"{temp_var.columns.min()} to {temp_var.columns.max()}"
        )
        # Check if there's a 'stage' index level
        if "stage" in temp_var.index.names:
            print(f"DIAGNOSTIC: 'stage' values in temp data: {sorted(temp_var.pix.unique('stage'))}")
        else:
            print("DIAGNOSTIC: No 'stage' index level found in temperature data!")
    else:
        print("DIAGNOSTIC: 'Surface Air Temperature Change' not found in database!")
else:
    print("No SCM output found in database yet")

# %% [markdown]
# ## Save
#
# The SCM output is already saved in the db.
# Here we also save the emissions that were actually used by the SCM.

# %%
# DIAGNOSTIC: Check what's in complete_scm before saving
print(f"DIAGNOSTIC: complete_scm year range before save: {complete_scm.columns.min()} to {complete_scm.columns.max()}")
print(f"DIAGNOSTIC: complete_scm variables: {sorted(complete_scm.pix.unique('variable')[:5])}...")  # Show first 5

# Check what's already in the database before overwriting
existing_data = SCM_OUTPUT_DB.load(pix.ismatch(model=f"*{model}*", climate_model=f"*{scm}*"))
if not existing_data.empty:
    print(
        f"DIAGNOSTIC: Existing data in SCM_OUTPUT_DB before overwrite: "
        f"{existing_data.columns.min()} to {existing_data.columns.max()}"
    )
    print(f"DIAGNOSTIC: Existing variables: {sorted(existing_data.pix.unique('variable')[:5])}...")

SCM_OUTPUT_DB.save(complete_scm.pix.assign(climate_model=scm), allow_overwrite=True)

# DIAGNOSTIC: Check what's in the database AFTER saving
final_data = SCM_OUTPUT_DB.load(pix.ismatch(model=f"*{model}*", climate_model=f"*{scm}*"))
print(
    f"DIAGNOSTIC: Final data in SCM_OUTPUT_DB after save: " f"{final_data.columns.min()} to {final_data.columns.max()}"
)
print(f"DIAGNOSTIC: Final variables: {sorted(final_data.pix.unique('variable')[:5])}...")
