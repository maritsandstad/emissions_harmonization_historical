"""
Test script to verify CICERO-SCM data format conversion.

This script:
1. Loads a sample scenario from the database
2. Converts it to CICERO-SCM emissions format
3. Writes the file and displays sample output
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas_indexing as pix

from emissions_harmonization_historical.ciceroscm_utils import dataframe_to_ciceroscm_emissions
from emissions_harmonization_historical.constants_5000 import INFILLED_SCENARIOS_DB

# Load a simple test scenario
print("Loading IMAGE SSP2 scenario...")
scenarios = INFILLED_SCENARIOS_DB.load(
    pix.isin(
        model="IMAGE 3.4",
        scenario="SSP2 - Medium Emissions",
        region="World",
    ),
    progress=False,
)

# Filter to just emissions (not concentrations)
emissions = scenarios.loc[scenarios.index.get_level_values("variable").str.startswith("Emissions|")]

print(f"Loaded {len(emissions)} emission variables")
print(f"Year range: {emissions.columns.min()} to {emissions.columns.max()}")
print("\nVariables:")
for var in sorted(emissions.pix.unique("variable"))[:10]:
    print(f"  - {var}")
print(f"  ... and {len(emissions.pix.unique('variable')) - 10} more")

# Convert to CICERO-SCM format
print("\nConverting to CICERO-SCM format...")
output_file = Path("output_test/test_emissions_ciceroscm.txt")
cscm_df, metadata = dataframe_to_ciceroscm_emissions(emissions, output_file=output_file)

print("\nConversion successful!")
print(f"  Scenario: {metadata['scenario']}")
print(f"  Components: {metadata['n_components']}")
print(f"  Year range: {metadata['year_range'][0]} to {metadata['year_range'][1]}")
print(f"  Output file: {output_file}")

print("\nFirst few rows of output:")
print(cscm_df.head(10).to_string(index=False, header=False))

print("\n✓ Test completed successfully!")
print(f"  You can now inspect the file: {output_file}")
