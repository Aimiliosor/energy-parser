import os
import chardet
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def clean_path(path: str) -> str:
    """Clean file path from drag-and-drop (strip quotes and whitespace)."""
    return path.strip().strip('"').strip("'")


def detect_encoding(file_path: str) -> str:
    """Detect file encoding using chardet."""
    with open(file_path, "rb") as f:
        raw = f.read(100_000)
    result = chardet.detect(raw)
    encoding = result["encoding"]
    confidence = result["confidence"]
    console.print(f"  Detected encoding: [bold]{encoding}[/bold] (confidence: {confidence:.0%})")
    return encoding


def detect_delimiter(file_path: str, encoding: str) -> str:
    """Detect delimiter by analyzing first 20 lines."""
    candidates = [";", ",", "\t", "|"]
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        lines = [f.readline() for _ in range(20)]
    lines = [line for line in lines if line.strip()]

    if not lines:
        return ","

    scores = {}
    for delim in candidates:
        counts = [line.count(delim) for line in lines]
        if all(c > 0 for c in counts):
            variance = max(counts) - min(counts)
            avg = sum(counts) / len(counts)
            scores[delim] = (avg, -variance)

    if scores:
        best = max(scores, key=lambda d: scores[d])
        delim_name = {";": "semicolon", ",": "comma", "\t": "tab", "|": "pipe"}
        console.print(f"  Detected delimiter: [bold]{delim_name.get(best, repr(best))}[/bold]")
        return best

    console.print("  [yellow]Could not detect delimiter, defaulting to comma[/yellow]")
    return ","


def detect_has_header(file_path: str, encoding: str, delimiter: str) -> bool:
    """Heuristic: first row has headers if it contains mostly non-numeric, non-date text."""
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        first_line = f.readline().strip()

    if not first_line:
        return False

    fields = first_line.split(delimiter)
    non_numeric_count = 0
    for field in fields:
        field = field.strip().strip('"')
        try:
            float(field.replace(",", "."))
            continue
        except ValueError:
            pass
        # Check if it looks like a date (contains - or / with digits)
        if any(c.isdigit() for c in field) and any(c in field for c in "-/"):
            continue
        if field:
            non_numeric_count += 1

    has_header = non_numeric_count >= len(fields) / 2
    console.print(f"  Header row detected: [bold]{'Yes' if has_header else 'No'}[/bold]")
    return has_header


def load_file(file_path: str) -> tuple[pd.DataFrame, dict]:
    """Load a CSV or XLSX file into a DataFrame. Returns (df, metadata)."""
    file_path = clean_path(file_path)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    metadata = {"file_path": file_path, "extension": ext}

    console.print(f"\n[bold cyan]Phase 1: Loading & Analyzing File[/bold cyan]")
    console.print(f"  File: {os.path.basename(file_path)}")

    if ext in (".xlsx", ".xls"):
        console.print("  Format: Excel")
        df = pd.read_excel(file_path, header=0)
        metadata["encoding"] = None
        metadata["delimiter"] = None
        metadata["has_header"] = True
    elif ext in (".csv", ".txt", ".tsv"):
        encoding = detect_encoding(file_path)
        delimiter = detect_delimiter(file_path, encoding)
        has_header = detect_has_header(file_path, encoding, delimiter)

        metadata["encoding"] = encoding
        metadata["delimiter"] = delimiter
        metadata["has_header"] = has_header

        df = pd.read_csv(
            file_path,
            sep=delimiter,
            encoding=encoding,
            header=0 if has_header else None,
            dtype=str,
            keep_default_na=False,
        )
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Ensure all columns are strings for consistent processing
    df.columns = [str(c) for c in df.columns]

    metadata["row_count"] = len(df)
    metadata["col_count"] = len(df.columns)

    console.print(f"  Rows: [bold]{metadata['row_count']}[/bold], Columns: [bold]{metadata['col_count']}[/bold]")

    return df, metadata


def display_preview(df: pd.DataFrame, n_rows: int = 10):
    """Display first n rows as a rich table."""
    table = Table(title=f"Preview (first {min(n_rows, len(df))} rows)", show_lines=True)

    table.add_column("#", style="dim", width=5)
    for col in df.columns:
        table.add_column(str(col), overflow="fold", max_width=30)

    for i, (_, row) in enumerate(df.head(n_rows).iterrows()):
        table.add_row(str(i), *[str(v)[:30] for v in row.values])

    console.print(table)
