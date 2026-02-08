# Spartacus - ReVolta Energy Analysis Tool

## Project Overview
Spartacus is an energy data analysis tool developed for ReVolta srl. It processes energy consumption and production data from CSV files, performs quality checks, statistical analysis, and generates professional PDF reports.

## Git Workflow
- **Auto-commit**: YES - Automatically commit all changes without asking
- **Auto-push**: YES - Automatically push commits to remote repository
- **Commit message format**: Use descriptive messages like:
  - "[Feature] Add peak consumption analysis"
  - "[Fix] Correct background image display"
  - "[Update] Improve GUI button styling"
  - "[Refactor] Optimize data validation logic"
- **Branch**: Work on main branch (or current branch)

## Permission Settings
- **File modifications**: Auto-approve - don't ask for permission
- **Git operations**: Auto-approve - commit and push automatically
- **Package installations**: Auto-approve if needed for features
- **Session mode**: Accept all changes by default

## Code Style & Standards
- **Language**: Python 3.13
- **Style**: Follow PEP 8 guidelines
- **Docstrings**: Use clear, concise docstrings for functions and classes
- **Comments**: Add explanatory comments for complex logic
- **Type hints**: Use where appropriate for clarity

## Branding & Design
- **Company**: ReVolta srl
- **Tool name**: Spartacus
- **Color palette**:
  - Primary: #2C495E (dark blue-gray), #FFFFFF (white)
  - Secondary: #EC465D (red), #B4BCD6 (light blue-gray)
- **Logo**: Logo_new_white.png (transparent background)
- **Images**: Energy storage background image
- **Copyright**: "Property of ReVolta srl. All rights reserved."

## Dependencies
- pandas>=2.0
- openpyxl>=3.1
- chardet>=5.0
- rich>=13.0
- Pillow (for image handling)
- reportlab (for PDF generation)
- matplotlib (for charts)
- tkinter (GUI)
- pyinstaller (for executable creation)

## Key Features
1. CSV data import with automatic encoding/delimiter detection
2. Data quality validation and KPI reporting
3. Statistical analysis (peaks, averages, percentiles)
4. Seasonal consumption/production patterns
5. Professional PDF report generation
6. Modern GUI with ReVolta branding
7. Command-line interface option
8. Standalone executable distribution

## Development Guidelines
- Test features in both GUI and CLI modes
- Ensure all graphics scale with window size
- Maintain data integrity through all transformations
- Track original vs corrected data for accurate analysis
- Use ReVolta colors consistently in visualizations
- Keep user experience simple and intuitive for non-technical users

## Testing
- Self-test data integrity after transformations
- Validate input vs output data correspondence
- Quality check all statistical calculations
- Test GUI responsiveness and layout
- Verify PDF generation with proper branding
