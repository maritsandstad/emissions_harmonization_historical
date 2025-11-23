"""
Utilities for running CICERO-SCM and converting between data formats.

This module provides:
- Conversion from DataFrame (CMIP7 ScenarioMIP format) to CICERO-SCM input format
- Conversion from CICERO-SCM output to ScmRun/DataFrame format
- Variable name mapping between naming conventions
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Variable name mapping: CMIP7 ScenarioMIP -> CICERO-SCM
# CICERO-SCM uses shorter names and different conventions
VARIABLE_MAPPING_TO_CICEROSCM = {
    # CO2 - CICERO expects fossil and land use separately
    "Emissions|CO2|Energy and Industrial Processes": ("CO2", "fossil"),  # Column name, Description
    "Emissions|CO2|AFOLU": ("CO2", "landuse"),
    # Alternative names (MAGICC style)
    "Emissions|CO2|MAGICC Fossil and Industrial": ("CO2", "fossil"),
    "Emissions|CO2|MAGICC AFOLU": ("CO2", "landuse"),
    # Greenhouse gases
    "Emissions|CH4": ("CH4", "total"),
    "Emissions|N2O": ("N2O", "total"),
    # Halogenated gases
    "Emissions|CFC11": ("CFC-11", "total"),
    "Emissions|CFC12": ("CFC-12", "total"),
    "Emissions|CFC113": ("CFC-113", "total"),
    "Emissions|CFC114": ("CFC-114", "total"),
    "Emissions|CFC115": ("CFC-115", "total"),
    "Emissions|CCl4": ("CCl4", "total"),
    "Emissions|CH3CCl3": ("CH3CCl3", "total"),
    "Emissions|CH3Br": ("CH3Br", "total"),
    # HCFCs
    "Emissions|HCFC22": ("HCFC-22", "total"),
    "Emissions|HCFC141b": ("HCFC-141b", "total"),
    "Emissions|HCFC142b": ("HCFC-142b", "total"),
    "Emissions|HCFC123": ("HCFC-123", "total"),
    # Halons
    "Emissions|Halon1211": ("H-1211", "total"),
    "Emissions|Halon1301": ("H-1301", "total"),
    "Emissions|Halon2402": ("H-2402", "total"),
    # HFCs (with HFC| prefix in variable names)
    "Emissions|HFC|HFC125": ("HFC125", "total"),
    "Emissions|HFC|HFC134a": ("HFC134a", "total"),
    "Emissions|HFC|HFC143a": ("HFC143a", "total"),
    "Emissions|HFC|HFC152a": ("HFC134a", "total"),  # Map to closest available in gaspam
    "Emissions|HFC|HFC227ea": ("HFC227ea", "total"),
    "Emissions|HFC|HFC23": ("HFC23", "total"),
    "Emissions|HFC|HFC236fa": ("HFC245fa", "total"),  # Map to closest available in gaspam
    "Emissions|HFC|HFC245fa": ("HFC245fa", "total"),
    "Emissions|HFC|HFC32": ("HFC32", "total"),
    "Emissions|HFC|HFC365mfc": ("HFC4310mee", "total"),  # Map to closest available in gaspam
    "Emissions|HFC|HFC43-10": ("HFC4310mee", "total"),
    # PFCs
    "Emissions|C2F6": ("C2F6", "total"),
    "Emissions|C6F14": ("C6F14", "total"),
    "Emissions|CF4": ("CF4", "total"),
    "Emissions|cC4F8": ("cC4F8", "total"),
    # Other
    "Emissions|SF6": ("SF6", "total"),
    "Emissions|SO2": ("SO2", "total"),
    "Emissions|Sulfur": ("SO2", "total"),  # Alternative name
    # Short-lived climate forcers
    "Emissions|NOx": ("NOx", "total"),
    "Emissions|CO": ("CO", "total"),
    "Emissions|VOC": ("NMVOC", "total"),
    "Emissions|NH3": ("NH3", "total"),
    "Emissions|BC": ("BC", "total"),
    "Emissions|OC": ("OC", "total"),
    # Biomass burning aerosols (often not separately reported, will use zeros if missing)
    "Emissions|BC|Biomass Burning": ("BMB_AEROS_BC", "total"),
    "Emissions|OC|Biomass Burning": ("BMB_AEROS_OC", "total"),
}

# Unit mapping: CMIP7 ScenarioMIP -> CICERO-SCM
UNIT_MAPPING_TO_CICEROSCM = {
    "Mt CO2/yr": "Pg_C",  # CO2: convert from CO2 mass to carbon mass
    "Mt CH4/yr": "Tg",  # CH4: 1 Mt = 1 Tg
    "kt N2O/yr": "Tg_N",  # N2O: mass of nitrogen, need conversion
    "Mt SO2/yr": "Tg_SO2",  # SO2: 1 Mt = 1 Tg
    "kt *": "Gg",  # Most halogenated gases: 1 kt = 1 Gg
    "Mt NO2/yr": "Mt_N",  # NOx: mass of nitrogen
    "Mt CO/yr": "Mt",
    "Mt VOC/yr": "Mt",
    "Mt NH3/yr": "Mt",
    "Mt BC/yr": "Tg",
    "Mt OC/yr": "Tg",
}


def convert_units_to_ciceroscm(value: float, from_unit: str, to_unit: str, gas: str | None = None) -> float:  # noqa: PLR0911
    """
    Convert emissions value between units.

    Parameters
    ----------
    value : float
        Value to convert
    from_unit : str
        Original unit
    to_unit : str
        Target unit (CICERO-SCM format)
    gas : str, optional
        Gas name (needed for some conversions)

    Returns
    -------
    float
        Converted value
    """
    # CO2: Mt CO2/yr -> Pg C/yr
    # Conversion: 1 Mt CO2 = 0.001 Gt CO2 = 0.001 * 12/44 Pg C
    if from_unit == "Mt CO2/yr" and to_unit == "Pg_C":
        return value * (12.0 / 44.0) / 1000.0

    # CH4: Mt CH4/yr -> Tg CH4/yr (1 Mt = 1 Tg)
    if from_unit == "Mt CH4/yr" and to_unit == "Tg":
        return value

    # N2O: kt N2O/yr -> Tg N/yr
    # Conversion: mass of N2O -> mass of N (multiply by 28/44)
    if from_unit == "kt N2O/yr" and to_unit == "Tg_N":
        return value * (28.0 / 44.0) / 1000.0

    # SO2: Mt SO2/yr -> Tg SO2/yr
    if from_unit == "Mt SO2/yr" and to_unit == "Tg_SO2":
        return value

    # Halogenated gases: kt/yr -> Gg/yr (1:1 conversion)
    if from_unit.startswith("kt ") and to_unit == "Gg":
        return value

    # NOx: Mt NO2/yr -> Mt N/yr
    # Conversion: mass of NO2 -> mass of N (multiply by 14/46)
    if from_unit == "Mt NO2/yr" and to_unit == "Mt_N":
        return value * (14.0 / 46.0)

    # BC/OC: Mt -> Tg
    if from_unit.startswith("Mt ") and from_unit.endswith("/yr") and to_unit == "Tg":
        return value

    # CO, VOC, NH3: Mt X/yr -> Mt (just remove /yr, value stays same)
    if from_unit.startswith("Mt ") and from_unit.endswith("/yr") and to_unit == "Mt":
        return value

    # Same units, no conversion
    if from_unit == to_unit:
        return value

    msg = f"Unsupported unit conversion: {from_unit} -> {to_unit} for gas {gas}"
    raise ValueError(msg)


def dataframe_to_ciceroscm_emissions(  # noqa: PLR0912, PLR0915
    df: pd.DataFrame,
    output_file: Path | str | None = None,
    start_year: int | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Convert DataFrame in CMIP7 ScenarioMIP format to CICERO-SCM emissions file format.

    The CICERO-SCM format is tab-separated with structure:
    - Row 1: Component names (gases)
    - Row 2: Units
    - Row 3: Description (e.g., "fossil_fuel", "landuse", "total")
    - Row 4: Reference (scenario name)
    - Rows 5+: Year and values for each gas

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with multi-index (model, scenario, region, variable, unit)
        and columns as years
    output_file : Path or str, optional
        If provided, write the emissions data to this file
    start_year : int, optional
        If provided, pad emissions with zeros from start_year to first year in df.
        This is needed when CICERO-SCM uses concentration-driven spinup (nystart < emstart).

    Returns
    -------
    ciceroscm_df : pd.DataFrame
        DataFrame in CICERO-SCM format ready to write as tab-separated file
    metadata : dict
        Metadata about the conversion (units, variables mapped, etc.)
    """
    # Check that we have World region
    if not all(df.index.get_level_values("region") == "World"):
        msg = "CICERO-SCM requires World (global) data only"
        raise ValueError(msg)

    # Extract scenario name
    scenarios = df.index.get_level_values("scenario").unique()
    if len(scenarios) > 1:
        msg = f"Multiple scenarios found: {scenarios}. CICERO-SCM expects single scenario per file."
        raise ValueError(msg)
    scenario_name = scenarios[0]

    # Prepare data structure for CICERO-SCM format
    components = []
    units = []
    descriptions = []
    references = []

    # Dictionary to store time series for each component
    emissions_data = {}

    # Process each variable
    for variable in df.index.get_level_values("variable").unique():
        if variable not in VARIABLE_MAPPING_TO_CICEROSCM:
            continue  # Skip unmapped variables

        ciceroscm_name, description = VARIABLE_MAPPING_TO_CICEROSCM[variable]

        # Get data for this variable
        var_data = df.loc[df.index.get_level_values("variable") == variable]

        # Get unit
        original_unit = var_data.index.get_level_values("unit").unique()[0]

        # Determine target unit
        target_unit = None
        for unit_pattern, cscm_unit in UNIT_MAPPING_TO_CICEROSCM.items():
            if unit_pattern == original_unit or (
                unit_pattern.endswith("*") and original_unit.startswith(unit_pattern[:-1])
            ):
                target_unit = cscm_unit
                break

        if target_unit is None:
            print(f"Warning: No unit mapping for {variable} ({original_unit}), skipping")
            continue

        # Convert units and extract values
        values = var_data.values.flatten()
        converted_values = np.array(
            [convert_units_to_ciceroscm(v, original_unit, target_unit, ciceroscm_name) for v in values]
        )

        # Handle special case: CO2 has two columns (fossil and landuse)
        if ciceroscm_name == "CO2":
            if description == "fossil":
                emissions_data[("CO2", "fossil")] = converted_values
                if ("CO2", "fossil") not in [(c, d) for c, u, d, r in zip(components, units, descriptions, references)]:
                    components.append("CO2")
                    units.append("Pg_C")
                    descriptions.append("fossil_fuel")
                    references.append(scenario_name)
            elif description == "landuse":
                emissions_data[("CO2", "landuse")] = converted_values
                if ("CO2", "landuse") not in [
                    (c, d) for c, u, d, r in zip(components, units, descriptions, references)
                ]:
                    components.append("CO2")
                    units.append("Pg_C")
                    descriptions.append("landuse")
                    references.append(scenario_name)
        else:
            # All other gases
            emissions_data[ciceroscm_name] = converted_values
            components.append(ciceroscm_name)
            units.append(target_unit)
            descriptions.append(description)
            references.append(scenario_name)

    # Add missing components that CICERO-SCM requires but may not be in the data
    # These are filled with zeros since we don't have separate emissions for them
    n_years = len(df.columns)
    required_components = [
        ("HCFC-123", "Gg", "total"),  # Often missing halogenated gas
        ("BMB_AEROS_OC", "Tg", "total"),  # Biomass burning organic carbon aerosol
        ("BMB_AEROS_BC", "Tg", "total"),  # Biomass burning black carbon aerosol
    ]

    for component_name, unit, description in required_components:
        if component_name not in components:
            emissions_data[component_name] = np.zeros(n_years)
            components.append(component_name)
            units.append(unit)
            descriptions.append(description)
            references.append(scenario_name)

    # Create CICERO-SCM format DataFrame
    years = df.columns.values.astype(int)  # Ensure years are integers

    # Pad with zeros from start_year if needed (for concentration-driven spinup)
    if start_year is not None and start_year < years[0]:
        n_padding_years = years[0] - start_year
        padding_years = np.arange(start_year, years[0], dtype=int)
        years = np.concatenate([padding_years, years])

        # Extend all emissions data with zeros for the padding period
        for key in emissions_data:
            padding_zeros = np.zeros(n_padding_years)
            emissions_data[key] = np.concatenate([padding_zeros, emissions_data[key]])

    # Build the output in CICERO-SCM format
    rows = [
        ["Component", *components],
        ["Unit", *units],
        ["Description", *descriptions],
        ["Reference", *references],
    ]

    # Add data rows
    for i, year in enumerate(years):
        row = [int(year)]  # Ensure year is written as integer
        for j, component in enumerate(components):
            desc = descriptions[j]
            if component == "CO2":
                key = ("CO2", desc.replace("fossil_fuel", "fossil"))
            else:
                key = component

            if key in emissions_data:
                row.append(f"{emissions_data[key][i]:.8f}")
            else:
                row.append("0.00000000")  # Default to zero if missing
        rows.append(row)

    ciceroscm_df = pd.DataFrame(rows)

    # Write to file if requested
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        ciceroscm_df.to_csv(output_file, sep="\t", header=False, index=False)

    metadata = {
        "scenario": scenario_name,
        "n_components": len(components),
        "components": components,
        "year_range": (int(years[0]), int(years[-1])),
    }

    return ciceroscm_df, metadata


def ciceroscm_output_to_dataframe(  # noqa: PLR0913
    cscm_results: dict,
    scenario_name: str,
    model_name: str,
    *,
    climate_model: str = "CICEROSCM",
    output_variables: tuple[str, ...] | None = None,
    start_year: int = 1700,
) -> pd.DataFrame:
    """
    Convert CICERO-SCM output dictionary to DataFrame format compatible with the database.

    Parameters
    ----------
    cscm_results : dict
        Results dictionary from CICERO-SCM._run() with results_as_dict=True
        Contains keys like 'dT_glob', 'dT_NH', 'dT_SH', 'Total_forcing', etc.
    scenario_name : str
        Scenario name (e.g., "SSP2 - Medium Emissions")
    model_name : str
        IAM model name (e.g., "IMAGE 3.4")
    climate_model : str
        Climate model name for metadata
    output_variables : tuple of str, optional
        Specific variables to extract. If None, extracts standard set.
    start_year : int
        Start year of the CICERO-SCM run (nystart parameter). Default 1700.

    Returns
    -------
    pd.DataFrame
        DataFrame with multi-index (model, scenario, region, variable, unit, climate_model)
        and columns as years
    """
    dfs = []

    # Map CICERO-SCM output names to standard variable names
    variable_mapping = {
        "dT_glob": ("Surface Air Temperature Change", "K"),
        "Total_forcing": ("Effective Radiative Forcing", "W/m^2"),
        "CO2": ("Atmospheric Concentrations|CO2", "ppm"),
        "CH4": ("Atmospheric Concentrations|CH4", "ppb"),
        "N2O": ("Atmospheric Concentrations|N2O", "ppb"),
    }

    # Filter to requested output variables if specified
    if output_variables:
        # Only include variables that are in our mapping
        requested_cscm_vars = {
            cscm_name for cscm_name in variable_mapping.keys() if variable_mapping[cscm_name][0] in output_variables
        }
    else:
        requested_cscm_vars = variable_mapping.keys()

    # Extract data for each variable
    for cscm_name in requested_cscm_vars:
        if cscm_name in cscm_results:
            std_name, unit = variable_mapping[cscm_name]
            values = cscm_results[cscm_name]

            # Handle both numpy arrays and pandas Series
            if isinstance(values, np.ndarray):
                # Assume years are sequential from start_year
                # CICERO-SCM results are indexed by timestep starting from nystart
                years = np.arange(len(values)) + start_year
            elif isinstance(values, pd.Series):
                years = values.index.values
                values = values.values
            else:
                continue  # Skip if unknown format

            # Create DataFrame row
            index = pd.MultiIndex.from_tuples(
                [(model_name, scenario_name, "World", std_name, unit, climate_model)],
                names=["model", "scenario", "region", "variable", "unit", "climate_model"],
            )

            df_row = pd.DataFrame([values], index=index, columns=years)
            dfs.append(df_row)

    # Combine all variables
    if dfs:
        return pd.concat(dfs)
    else:
        return pd.DataFrame()
