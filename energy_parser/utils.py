"""Shared utility functions for Spartacus energy analysis tool."""

import re
from datetime import datetime


def build_output_filename(site_name: str, suffix: str, ext: str) -> str:
    """Build a standardized output filename from site name.

    Args:
        site_name: The site name to include in the filename.
        suffix: Descriptive suffix (e.g. "EnergyAnalysis", "parsed").
        ext: File extension without dot (e.g. "xlsx", "pdf").

    Returns:
        Filename string like "20260208_Site_Name_EnergyAnalysis.xlsx".
    """
    date_str = datetime.now().strftime("%Y%m%d")
    safe_name = re.sub(r'[^\w\s-]', '', site_name).strip()
    safe_name = re.sub(r'\s+', '_', safe_name)
    return f"{date_str}_{safe_name}_{suffix}.{ext}"
