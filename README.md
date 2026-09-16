# HelioAgent

HelioAgent is a Python prototype for designing and simulating a heliostat field. It uses SolarPILOT through the CoPylot API to generate a field layout and SolTrace through the local PySolTrace wrapper to trace rays and evaluate receiver performance.

## Requirements

- Windows
- Miniconda or Anaconda
- The native DLLs in `api/`:
  - `solarpilot.dll`
  - `coretrace_api.dll`
  - `embree4.dll`
- A SolarPILOT DLL build compatible with this Python wrapper. The repository currently includes `api/solarpilot.dll`.

The native wrappers resolve DLL paths from the current working directory. Run commands from the repository root, not from another directory:

```powershell
cd path\to\HelioAgent
```

## Environment setup

Create the Conda environment from the included environment file:

```powershell
conda env create -f environment.yml
conda activate helioagent
```

To update an existing environment after changing the file:

```powershell
conda env update -f environment.yml --prune
```

## Running the project

`main.py` builds a SolarPILOT field, creates the SolTrace geometry, runs the configured simulations, and writes CSV and PNG results to `Results\`.

```powershell
python main.py
```

The default run is computationally intensive. Before a first test run, reduce `field.des_sim_ndays`, `NRAYS`, or the field size in `main.py` as needed. The script creates the `Results\` directory only if it already exists, so create it first:

```powershell
New-Item -ItemType Directory -Force Results
```

Typical outputs include:

- `baseline_sim_results.csv`
- `correction_sim_results.csv`
- `efficiency_temperature_time_series.png`
- `efficiency_difference_vs_temperature.png`

The default weather input is `weather_files/USA CA Daggett Barstow-daggett Ap (TMY3).csv`.

## Focal-length test

`test_focal_length_update.py` exercises focal-length changes and writes SolTrace input files while running ray-trace comparisons:

```powershell
New-Item -ItemType Directory -Force Results
python test_focal_length_update.py
```

The test performs large ray traces and may take a significant amount of time.

## Project layout

- `main.py` - main field-generation and time-series simulation script
- `copylot_to_soltrace.py` - SolarPILOT-to-SolTrace field conversion and simulation helpers
- `heliostat.py` - heliostat geometry and focal-length behavior
- `util.py` - solar-position and geometry utilities
- `api/copylot.py` - ctypes wrapper for SolarPILOT
- `api/pysoltrace.py` - Python wrapper for SolTrace
- `weather_files/` - TMY weather inputs
- `test_focal_length_update.py` - focal-length update integration test

## Troubleshooting

- `FileNotFoundError` or DLL load errors usually mean the command was not started from the repository root, a required DLL is missing from `api/`, or a native DLL dependency is incompatible with the current Python/Windows environment.
- If imports fail, verify that the Conda environment is active and that the environment was created from `environment.yml`.
- The simulation uses multiprocessing and native libraries; keep the `if __name__ == "__main__":` guards in executable scripts.
