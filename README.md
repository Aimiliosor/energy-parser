# Energy Parser

Interactive CLI tool that ingests energy CSV/XLSX files, standardizes them, runs quality checks, and outputs clean XLSX files.

## Features

- Reads CSV and XLSX energy data files with automatic encoding detection
- Auto-detects date formats, column types, and data granularity
- Identifies date, consumption, and production columns
- Converts energy units (W, kW, Wh, kWh, MWh, MW) to a standard kW format
- Quality checks for missing timestamps, duplicate rows, missing values, and outliers
- Interactive corrections with cleaned output

## Requirements

- Python 3.10+
- pandas >= 2.0
- openpyxl >= 3.1
- chardet >= 5.0
- rich >= 13.0

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run.py
```

The CLI guides you through 5 phases:

1. **Input & Analysis** - Load a file and auto-analyze its structure
2. **Column Identification** - Confirm or select date, consumption, and production columns
3. **Transform & Output** - Standardize dates and units, save initial parsed XLSX
4. **Quality Check** - Detect missing timestamps, duplicates, gaps, and outliers
5. **Corrections** - Optionally apply fixes and export a clean XLSX
