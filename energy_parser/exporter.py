import os
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, Alignment, numbers
from openpyxl.utils import get_column_letter
from rich.console import Console

console = Console()


def save_xlsx(df: pd.DataFrame, output_path: str, is_final: bool = False):
    """Save DataFrame to formatted XLSX file.

    Args:
        df: DataFrame with 'Date & Time' and value columns
        output_path: Full path for the output file
        is_final: If True, apply full formatting (final output)
    """
    label = "Final" if is_final else "Initial"
    console.print(f"\n  Saving {label.lower()} XLSX: [bold]{os.path.basename(output_path)}[/bold]")

    # Format Date & Time as string for consistent display
    df_out = df.copy()
    df_out["Date & Time"] = df_out["Date & Time"].dt.strftime("%Y-%m-%d %H:%M")

    # Write with pandas first
    df_out.to_excel(output_path, index=False, engine="openpyxl")

    # Now open with openpyxl for formatting
    wb = load_workbook(output_path)
    ws = wb.active

    # Header formatting
    header_font = Font(bold=True, size=11)
    for col_idx in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")

    # Column widths
    for col_idx in range(1, ws.max_column + 1):
        col_letter = get_column_letter(col_idx)
        header = ws.cell(row=1, column=col_idx).value or ""

        if "Date" in str(header):
            ws.column_dimensions[col_letter].width = 20
        else:
            ws.column_dimensions[col_letter].width = 18

    # Number formatting for value columns (2 decimal places)
    if is_final:
        for col_idx in range(2, ws.max_column + 1):
            for row_idx in range(2, ws.max_row + 1):
                cell = ws.cell(row=row_idx, column=col_idx)
                if cell.value is not None:
                    cell.number_format = "#,##0.00"
                    cell.alignment = Alignment(horizontal="right")

    # Date column alignment
    for row_idx in range(2, ws.max_row + 1):
        cell = ws.cell(row=row_idx, column=1)
        cell.alignment = Alignment(horizontal="left")

    wb.save(output_path)
    console.print(f"  [green]Saved: {output_path}[/green]")
    console.print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
