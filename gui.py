"""Spartacus - ReVolta Energy Analysis Tool.

A graphical interface for the Spartacus energy analysis tool.
Property of ReVolta srl. All rights reserved.
"""

import os
import sys
import io
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd
from PIL import Image, ImageTk

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from energy_parser.file_reader import load_file, clean_path, detect_encoding, detect_delimiter
from energy_parser.analyzer import (
    analyze_columns,
    detect_granularity,
    detect_granularity_with_confidence,
    STANDARD_GRANULARITIES,
)
from energy_parser.transformer import transform_data
from energy_parser.quality_check import run_quality_check
from energy_parser.corrector import (
    fill_from_previous_day,
    correct_duplicates,
    correct_time_gaps,
    correct_missing_values,
)
from energy_parser.exporter import save_xlsx
from energy_parser.data_validator import run_validation
from energy_parser.statistics import run_statistical_analysis
from energy_parser.utils import build_output_filename
from energy_parser.report_generator import (
    generate_pdf_report, generate_seasonal_chart,
    generate_peak_timeline_chart, generate_peak_heatmap,
    generate_peak_duration_chart, generate_peak_value_trend_chart,
    generate_histogram_chart, generate_cdf_chart,
    generate_peak_hour_frequency_chart,
)
from energy_parser.battery_sizing import BatterySizer


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)


# Color Palette
COLORS = {
    "primary": "#2C495E",      # Dark blue-gray
    "white": "#FFFFFF",
    "secondary": "#EC465D",    # Red/pink
    "light_gray": "#B4BCD6",   # Light blue-gray
    "bg": "#F5F7FA",           # Light background
    "success": "#28A745",      # Green
    "warning": "#FFC107",      # Yellow
    "text_dark": "#1A1A2E",    # Dark text
    "orange": "#FF8C00",       # Orange (poor quality)
}

VALID_UNITS = ["W", "kW", "Wh", "kWh", "MWh", "MW"]


class ModernButton(tk.Canvas):
    """Custom rounded button with modern styling."""

    def __init__(self, parent, text, command=None, bg=COLORS["primary"],
                 fg=COLORS["white"], width=180, height=40, **kwargs):
        super().__init__(parent, width=width, height=height,
                        highlightthickness=0, bg=parent.cget("bg"), **kwargs)

        self.command = command
        self.text = text
        self.bg_color = bg
        self.fg_color = fg
        self.width = width
        self.height = height
        self.enabled = True

        self.draw_button()

        self.bind("<Enter>", self.on_hover)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)

    def draw_button(self, hover=False):
        self.delete("all")
        radius = 10

        if not self.enabled:
            # Disabled state: flat, no shadow
            self.create_rounded_rect(2, 2, self.width-2, self.height-2, radius,
                                    fill="#D0D5E0", outline="#B8BEC8")
            self.create_text(self.width/2, self.height/2, text=self.text,
                            fill="#888888", font=("Segoe UI", 10, "bold"))
            return

        # Shadow (offset 2px down-right for depth)
        self.create_rounded_rect(4, 4, self.width-1, self.height-1, radius,
                                fill="#A0A8B8", outline="")

        # Button body
        color = self.lighten_color(self.bg_color, 0.2) if hover else self.bg_color
        border_color = self.darken_color(self.bg_color, 0.15)
        self.create_rounded_rect(2, 2, self.width-3, self.height-3, radius,
                                fill=color, outline=border_color)

        # Text
        self.create_text(self.width/2, self.height/2, text=self.text,
                        fill=self.fg_color, font=("Segoe UI", 10, "bold"))

    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1+radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def lighten_color(self, color, factor):
        """Lighten a hex color by a factor."""
        color = color.lstrip("#")
        r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    def darken_color(self, color, factor):
        """Darken a hex color by a factor."""
        color = color.lstrip("#")
        r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
        r = max(0, int(r * (1 - factor)))
        g = max(0, int(g * (1 - factor)))
        b = max(0, int(b * (1 - factor)))
        return f"#{r:02x}{g:02x}{b:02x}"

    def on_hover(self, event):
        if self.enabled:
            self.config(cursor="hand2")
            self.draw_button(hover=True)

    def on_leave(self, event):
        self.config(cursor="")
        self.draw_button(hover=False)

    def on_click(self, event):
        if self.enabled and self.command:
            self.command()

    def set_enabled(self, enabled):
        self.enabled = enabled
        self.draw_button()


class EnergyParserGUI:
    """Main GUI Application for Spartacus."""

    def __init__(self, root):
        self.root = root
        self.root.title("Spartacus - ReVolta Energy Analysis Tool")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        self.root.configure(bg=COLORS["bg"])

        # Set icon if available
        icon_path = resource_path("favicon 1.ico")
        if os.path.exists(icon_path):
            try:
                self.root.iconbitmap(icon_path)
            except:
                pass

        # Data state
        self.file_path = None
        self.df = None
        self.metadata = None
        self.analysis = None
        self.transformed_df = None
        self.quality_report = None
        self.hours_per_interval = 1.0
        self.granularity_label = "unknown"
        self.kpi_data = None
        self.stats_result = None
        self.battery_result = None

        # Column selections
        self.date_col_var = tk.StringVar(value="0")
        self.cons_col_var = tk.StringVar(value="1")
        self.prod_col_var = tk.StringVar(value="none")
        self.cons_unit_var = tk.StringVar(value="kW")
        self.prod_unit_var = tk.StringVar(value="kW")

        # Site information
        self.site_name_var = tk.StringVar(value="")
        self.grid_capacity_var = tk.StringVar(value="")

        # Image references (keep references to prevent garbage collection)
        self.bg_image_original = None
        self.bg_photo = None
        self.logo_image_original = None
        self.logo_photo = None

        # Load images
        self.load_images()

        # Setup UI
        self.setup_styles()
        self.create_widgets()

        # Bind resize event for responsive images
        self.root.bind("<Configure>", self.on_resize)
        self._last_width = 0
        self._last_height = 0

    def load_images(self):
        """Load and store original images for responsive resizing."""
        # Load background image
        bg_path = resource_path("Energy storage.jpg")
        if os.path.exists(bg_path):
            try:
                self.bg_image_original = Image.open(bg_path)
            except:
                pass

        # Load logo - try white version first, then navy
        logo_paths = [
            resource_path("Logo_new_white.png"),
            resource_path("logo white.png"),
            resource_path("logo white.jpeg"),
            resource_path("logo White.png"),
            resource_path("logo Navy.jpeg"),
        ]
        for logo_path in logo_paths:
            if os.path.exists(logo_path):
                try:
                    self.logo_image_original = Image.open(logo_path)
                    break
                except:
                    continue

    def on_resize(self, event):
        """Handle window resize for responsive images."""
        # Only respond to root window resizes, not child widgets
        if event.widget != self.root:
            return

        # Debounce - only update if size changed significantly
        width = self.root.winfo_width()
        height = self.root.winfo_height()

        if abs(width - self._last_width) < 50 and abs(height - self._last_height) < 50:
            return

        self._last_width = width
        self._last_height = height

        # Update logo
        self.update_logo()

    def update_background(self):
        """Update background image on the content canvas."""
        if not self.bg_image_original or not hasattr(self, 'content_canvas'):
            return

        try:
            width = self.content_canvas.winfo_width()
            height = self.content_canvas.winfo_height()

            if width < 100 or height < 100:
                return

            # Create resized image maintaining aspect ratio, centered
            img = self.bg_image_original.copy()
            img_ratio = img.width / img.height
            canvas_ratio = width / height

            if img_ratio > canvas_ratio:
                new_height = height
                new_width = int(height * img_ratio)
            else:
                new_width = width
                new_height = int(width / img_ratio)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Crop to center
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            img = img.crop((left, top, left + width, top + height))

            # Semi-transparent overlay (~37% image visibility)
            overlay = Image.new('RGBA', (width, height), (245, 247, 250, 160))
            img = img.convert('RGBA')
            img = Image.alpha_composite(img, overlay)

            self._canvas_bg_photo = ImageTk.PhotoImage(img)
            self.content_canvas.itemconfig("bg", image=self._canvas_bg_photo)
            self._reposition_bg()
        except Exception:
            pass

    def update_logo(self):
        """Update logo size based on header height."""
        if not self.logo_image_original or not hasattr(self, 'logo_label'):
            return

        try:
            # Calculate logo size based on header (proportional)
            header_height = 80
            logo_height = int(header_height * 0.6)
            logo_width = int(logo_height * (self.logo_image_original.width / self.logo_image_original.height))

            # Limit max width
            max_width = int(self.root.winfo_width() * 0.15)
            if logo_width > max_width:
                logo_width = max_width
                logo_height = int(logo_width * (self.logo_image_original.height / self.logo_image_original.width))

            img = self.logo_image_original.copy()
            img = img.resize((logo_width, logo_height), Image.Resampling.LANCZOS)

            self.logo_photo = ImageTk.PhotoImage(img)
            self.logo_label.configure(image=self.logo_photo)
        except:
            pass

    def setup_styles(self):
        """Configure ttk styles for modern look."""
        style = ttk.Style()
        style.theme_use("clam")

        # Configure frame styles
        style.configure("Card.TFrame", background=COLORS["white"])
        style.configure("Main.TFrame", background=COLORS["bg"])

        # Configure label styles
        style.configure("Header.TLabel",
                       background=COLORS["primary"],
                       foreground=COLORS["white"],
                       font=("Segoe UI", 11, "bold"),
                       padding=10)

        style.configure("Title.TLabel",
                       background=COLORS["white"],
                       foreground=COLORS["primary"],
                       font=("Segoe UI", 14, "bold"))

        style.configure("Subtitle.TLabel",
                       background=COLORS["white"],
                       foreground=COLORS["text_dark"],
                       font=("Segoe UI", 10))

        style.configure("Status.TLabel",
                       background=COLORS["bg"],
                       foreground=COLORS["text_dark"],
                       font=("Segoe UI", 9))

        # Configure combobox
        style.configure("TCombobox", padding=5)

        # Configure progress bar
        style.configure("Custom.Horizontal.TProgressbar",
                       background=COLORS["secondary"],
                       troughcolor=COLORS["light_gray"])

        # Configure Treeview
        style.configure("Treeview",
                       background=COLORS["white"],
                       foreground=COLORS["text_dark"],
                       rowheight=25,
                       fieldbackground=COLORS["white"])
        style.configure("Treeview.Heading",
                       background=COLORS["primary"],
                       foreground=COLORS["white"],
                       font=("Segoe UI", 9, "bold"))
        style.map("Treeview.Heading",
                 background=[("active", COLORS["primary"])])

    def create_widgets(self):
        """Create all GUI widgets."""
        # Header (packed at top, opaque branded bar)
        self.create_header(self.root)

        # Footer (packed at bottom first so it stays below content)
        self.create_footer(self.root)

        # Content area: canvas with background image + scrollable content
        canvas_container = tk.Frame(self.root, bg=COLORS["bg"])
        canvas_container.pack(fill=tk.BOTH, expand=True)

        self.content_canvas = tk.Canvas(canvas_container, highlightthickness=0,
                                         bg=COLORS["bg"])
        scrollbar = ttk.Scrollbar(canvas_container, orient="vertical",
                                   command=self._on_scroll)
        self.content_canvas.configure(yscrollcommand=scrollbar.set)

        self.content_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Background image item on canvas (stays at viewport origin)
        self._canvas_bg_photo = None
        self.content_canvas.create_image(0, 0, anchor="nw", tags="bg")

        # Scrollable content frame (placed as a window on the canvas)
        self.content_frame = tk.Frame(self.content_canvas, bg=COLORS["bg"])
        self._content_window_id = self.content_canvas.create_window(
            (20, 10), window=self.content_frame, anchor="nw", tags="content")

        self.content_frame.bind(
            "<Configure>",
            lambda e: self.content_canvas.configure(
                scrollregion=self.content_canvas.bbox("all"))
        )
        self.content_canvas.bind("<Configure>", self._on_canvas_configure)

        # Bind mousewheel
        self.content_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Create sections
        self.create_file_section()
        self.create_site_info_section()
        self.create_preview_section()
        self.create_config_section()
        self.create_actions_section()
        self.create_results_section()
        self.create_kpi_section()
        self.create_statistics_section()
        self.create_battery_section()
        self.create_tools_section()

        # Initial background update
        self.root.after(100, self.update_background)

    def _on_scroll(self, *args):
        """Handle scrollbar movement — scroll content and reposition bg."""
        self.content_canvas.yview(*args)
        self._reposition_bg()

    def _on_mousewheel(self, event):
        """Handle mousewheel — scroll content and reposition bg."""
        self.content_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self._reposition_bg()

    def _reposition_bg(self):
        """Keep background image fixed at the viewport origin."""
        x = self.content_canvas.canvasx(0)
        y = self.content_canvas.canvasy(0)
        self.content_canvas.coords("bg", x, y)
        self.content_canvas.tag_lower("bg")

    def _on_canvas_configure(self, event):
        """Update content frame width to fill canvas and refresh background."""
        canvas_width = event.width
        self.content_canvas.itemconfig(
            self._content_window_id, width=max(1, canvas_width - 40))
        self.update_background()

    def create_header(self, parent):
        """Create the header with logo and title."""
        header_frame = tk.Frame(parent, bg=COLORS["primary"], height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)

        # Left side - Logo (no background, transparent)
        left_frame = tk.Frame(header_frame, bg=COLORS["primary"])
        left_frame.pack(side=tk.LEFT, padx=20, pady=10)

        # Display logo
        if self.logo_image_original:
            try:
                logo_img = self.logo_image_original.copy()
                logo_img.thumbnail((150, 50), Image.Resampling.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(logo_img)
                self.logo_label = tk.Label(left_frame, image=self.logo_photo, bg=COLORS["primary"])
                self.logo_label.pack(side=tk.LEFT)
            except Exception as e:
                # Fallback to text
                self.logo_label = tk.Label(left_frame, text="ReVolta",
                                           font=("Segoe UI", 20, "bold"),
                                           bg=COLORS["primary"], fg=COLORS["white"])
                self.logo_label.pack(side=tk.LEFT)
        else:
            # Fallback to text if no logo found
            self.logo_label = tk.Label(left_frame, text="ReVolta",
                                       font=("Segoe UI", 20, "bold"),
                                       bg=COLORS["primary"], fg=COLORS["white"])
            self.logo_label.pack(side=tk.LEFT)

        # Center - Title
        center_frame = tk.Frame(header_frame, bg=COLORS["primary"])
        center_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)

        title_label = tk.Label(center_frame, text="Spartacus",
                              font=("Segoe UI", 18, "bold"),
                              bg=COLORS["primary"], fg=COLORS["white"])
        title_label.pack(pady=5)

        subtitle_label = tk.Label(center_frame,
                                 text="Property of ReVolta srl. All rights reserved.",
                                 font=("Segoe UI", 9),
                                 bg=COLORS["primary"], fg=COLORS["light_gray"])
        subtitle_label.pack()

        # Right side - removed the small energy image (now using as background)

    def create_card(self, parent, title):
        """Create a card-style container."""
        outer_frame = tk.Frame(parent, bg=COLORS["bg"])
        outer_frame.pack(fill=tk.X, pady=10)

        # Card with shadow effect
        card = tk.Frame(outer_frame, bg=COLORS["white"], relief=tk.FLAT, bd=0)
        card.pack(fill=tk.X, padx=2, pady=2)

        # Title bar
        title_bar = tk.Frame(card, bg=COLORS["primary"])
        title_bar.pack(fill=tk.X)

        title_label = tk.Label(title_bar, text=title,
                              font=("Segoe UI", 11, "bold"),
                              bg=COLORS["primary"], fg=COLORS["white"],
                              padx=15, pady=8)
        title_label.pack(side=tk.LEFT)

        # Content area
        content = tk.Frame(card, bg=COLORS["white"], padx=15, pady=15)
        content.pack(fill=tk.X)

        return content

    def create_file_section(self):
        """Create the file selection section."""
        content = self.create_card(self.content_frame, "1. Select File")

        # File selection row
        file_row = tk.Frame(content, bg=COLORS["white"])
        file_row.pack(fill=tk.X, pady=5)

        self.file_entry = ttk.Entry(file_row, width=70, font=("Segoe UI", 10))
        self.file_entry.pack(side=tk.LEFT, padx=(0, 10))

        self.browse_btn = ModernButton(file_row, "Browse...",
                                       command=self.browse_file,
                                       bg=COLORS["light_gray"],
                                       fg=COLORS["primary"], width=120)
        self.browse_btn.pack(side=tk.LEFT)

        self.load_btn = ModernButton(file_row, "Load File",
                                    command=self.load_file,
                                    bg=COLORS["secondary"], width=120)
        self.load_btn.pack(side=tk.LEFT, padx=10)

        # File info
        self.file_info_label = tk.Label(content, text="No file loaded",
                                        font=("Segoe UI", 9),
                                        bg=COLORS["white"], fg=COLORS["text_dark"])
        self.file_info_label.pack(anchor=tk.W, pady=5)

    def create_site_info_section(self):
        """Create the site information section."""
        content = self.create_card(self.content_frame, "2. Site Information")

        info_grid = tk.Frame(content, bg=COLORS["white"])
        info_grid.pack(fill=tk.X)

        # Row 0: Site Name
        tk.Label(info_grid, text="Site Name:",
                font=("Segoe UI", 10), bg=COLORS["white"]).grid(
                    row=0, column=0, sticky=tk.W, pady=5)
        site_entry = ttk.Entry(info_grid, textvariable=self.site_name_var,
                               width=40, font=("Segoe UI", 10))
        site_entry.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)

        # Row 1: Grid Connection Capacity
        tk.Label(info_grid, text="Grid Connection Capacity (kW):",
                font=("Segoe UI", 10), bg=COLORS["white"]).grid(
                    row=1, column=0, sticky=tk.W, pady=5)
        capacity_entry = ttk.Entry(info_grid, textvariable=self.grid_capacity_var,
                                   width=15, font=("Segoe UI", 10))
        capacity_entry.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)

    def create_preview_section(self):
        """Create the data preview section."""
        content = self.create_card(self.content_frame, "3. Data Preview")

        # Preview table
        tree_frame = tk.Frame(content, bg=COLORS["white"])
        tree_frame.pack(fill=tk.BOTH, expand=True)

        # Horizontal scrollbar
        h_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # Vertical scrollbar
        v_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.preview_tree = ttk.Treeview(tree_frame, height=8,
                                         xscrollcommand=h_scroll.set,
                                         yscrollcommand=v_scroll.set)
        self.preview_tree.pack(fill=tk.BOTH, expand=True)

        h_scroll.config(command=self.preview_tree.xview)
        v_scroll.config(command=self.preview_tree.yview)

        # Analysis info
        self.analysis_text = tk.Text(content, height=4, width=80,
                                     font=("Consolas", 9),
                                     bg=COLORS["bg"], fg=COLORS["text_dark"],
                                     relief=tk.FLAT, padx=10, pady=10)
        self.analysis_text.pack(fill=tk.X, pady=(10, 0))
        self.analysis_text.config(state=tk.DISABLED)

    def create_config_section(self):
        """Create the configuration section."""
        content = self.create_card(self.content_frame, "4. Column Configuration")

        # Grid for column selection
        config_grid = tk.Frame(content, bg=COLORS["white"])
        config_grid.pack(fill=tk.X)

        # Row 1: Date column
        tk.Label(config_grid, text="Date/Time Column:",
                font=("Segoe UI", 10), bg=COLORS["white"]).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.date_col_combo = ttk.Combobox(config_grid, textvariable=self.date_col_var,
                                           width=30, state="readonly")
        self.date_col_combo.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)

        # Row 2: Consumption column
        tk.Label(config_grid, text="Consumption Column:",
                font=("Segoe UI", 10), bg=COLORS["white"]).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.cons_col_combo = ttk.Combobox(config_grid, textvariable=self.cons_col_var,
                                           width=30, state="readonly")
        self.cons_col_combo.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)

        tk.Label(config_grid, text="Unit:",
                font=("Segoe UI", 10), bg=COLORS["white"]).grid(row=1, column=2, sticky=tk.W, pady=5)
        self.cons_unit_combo = ttk.Combobox(config_grid, textvariable=self.cons_unit_var,
                                            width=10, values=VALID_UNITS, state="readonly")
        self.cons_unit_combo.grid(row=1, column=3, padx=10, pady=5, sticky=tk.W)

        # Row 3: Production column
        tk.Label(config_grid, text="Production Column (optional):",
                font=("Segoe UI", 10), bg=COLORS["white"]).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.prod_col_combo = ttk.Combobox(config_grid, textvariable=self.prod_col_var,
                                           width=30, state="readonly")
        self.prod_col_combo.grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)

        tk.Label(config_grid, text="Unit:",
                font=("Segoe UI", 10), bg=COLORS["white"]).grid(row=2, column=2, sticky=tk.W, pady=5)
        self.prod_unit_combo = ttk.Combobox(config_grid, textvariable=self.prod_unit_var,
                                            width=10, values=VALID_UNITS, state="readonly")
        self.prod_unit_combo.grid(row=2, column=3, padx=10, pady=5, sticky=tk.W)

        # Granularity info
        self.granularity_label_widget = tk.Label(content, text="Detected granularity: --",
                                                 font=("Segoe UI", 10, "italic"),
                                                 bg=COLORS["white"], fg=COLORS["primary"])
        self.granularity_label_widget.pack(anchor=tk.W, pady=(10, 0))

    def create_actions_section(self):
        """Create the actions section."""
        content = self.create_card(self.content_frame, "5. Process Data")

        # Buttons row
        btn_row = tk.Frame(content, bg=COLORS["white"])
        btn_row.pack(fill=tk.X, pady=10)

        self.transform_btn = ModernButton(btn_row, "Transform Data",
                                          command=self.transform_data,
                                          bg=COLORS["secondary"], width=150)
        self.transform_btn.pack(side=tk.LEFT, padx=5)
        self.transform_btn.set_enabled(False)

        self.quality_btn = ModernButton(btn_row, "Quality Check",
                                        command=self.run_quality_check,
                                        bg=COLORS["secondary"], width=150)
        self.quality_btn.pack(side=tk.LEFT, padx=5)
        self.quality_btn.set_enabled(False)

        self.correct_btn = ModernButton(btn_row, "Apply Corrections",
                                        command=self.apply_corrections,
                                        bg=COLORS["secondary"], width=150)
        self.correct_btn.pack(side=tk.LEFT, padx=5)
        self.correct_btn.set_enabled(False)

        self.export_btn = ModernButton(btn_row, "Export to Excel",
                                       command=self.export_data,
                                       bg=COLORS["success"], width=150)
        self.export_btn.pack(side=tk.LEFT, padx=5)
        self.export_btn.set_enabled(False)

        # Progress bar
        progress_frame = tk.Frame(content, bg=COLORS["white"])
        progress_frame.pack(fill=tk.X, pady=10)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame,
                                            variable=self.progress_var,
                                            style="Custom.Horizontal.TProgressbar",
                                            length=400, mode="determinate")
        self.progress_bar.pack(side=tk.LEFT)

        self.progress_label = tk.Label(progress_frame, text="Ready",
                                       font=("Segoe UI", 9),
                                       bg=COLORS["white"], fg=COLORS["text_dark"])
        self.progress_label.pack(side=tk.LEFT, padx=10)

    def create_results_section(self):
        """Create the results display section."""
        content = self.create_card(self.content_frame, "6. Results & Quality Report")

        # Results text area
        self.results_text = tk.Text(content, height=12, width=100,
                                    font=("Consolas", 9),
                                    bg=COLORS["bg"], fg=COLORS["text_dark"],
                                    relief=tk.FLAT, padx=10, pady=10)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Configure text tags for formatting
        self.results_text.tag_configure("header", font=("Consolas", 10, "bold"),
                                        foreground=COLORS["primary"])
        self.results_text.tag_configure("success", foreground=COLORS["success"])
        self.results_text.tag_configure("warning", foreground=COLORS["warning"])
        self.results_text.tag_configure("error", foreground=COLORS["secondary"])

        self.results_text.config(state=tk.DISABLED)

    def create_kpi_section(self):
        """Create the KPI dashboard section."""
        content = self.create_card(self.content_frame, "7. KPI Dashboard")

        self.kpi_placeholder = tk.Label(
            content, text="Run Quality Check to see KPI dashboard",
            font=("Segoe UI", 10, "italic"),
            bg=COLORS["white"], fg=COLORS["light_gray"])
        self.kpi_placeholder.pack(anchor=tk.W, pady=5)

        # Frame for KPI tiles
        self.kpi_frame = tk.Frame(content, bg=COLORS["white"])
        self.kpi_frame.pack(fill=tk.X, pady=5)

        # Frame for detailed results
        self.kpi_detail_frame = tk.Frame(content, bg=COLORS["white"])
        self.kpi_detail_frame.pack(fill=tk.X, pady=5)

    def _create_kpi_tile(self, parent, row, col, title, value, color):
        """Create a single KPI tile in the grid."""
        tile = tk.Frame(parent, bg=COLORS["white"], relief=tk.FLAT, bd=1,
                        highlightbackground=COLORS["light_gray"], highlightthickness=1)
        tile.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

        # Colored top bar
        bar = tk.Frame(tile, bg=color, height=4)
        bar.pack(fill=tk.X)

        # Title label
        tk.Label(tile, text=title, font=("Segoe UI", 8),
                 bg=COLORS["white"], fg=COLORS["light_gray"]).pack(padx=8, pady=(6, 0), anchor=tk.W)

        # Value label
        tk.Label(tile, text=str(value), font=("Segoe UI", 14, "bold"),
                 bg=COLORS["white"], fg=color).pack(padx=8, pady=(0, 8), anchor=tk.W)

    def _status_color(self, value, green_threshold, yellow_threshold, higher_is_better=True):
        """Return a hex color based on threshold."""
        if higher_is_better:
            if value >= green_threshold:
                return COLORS["success"]
            elif value >= yellow_threshold:
                return COLORS["warning"]
            return COLORS["secondary"]
        else:
            if value <= green_threshold:
                return COLORS["success"]
            elif value <= yellow_threshold:
                return COLORS["warning"]
            return COLORS["secondary"]

    def display_kpi_dashboard(self, kpi):
        """Populate KPI tiles from a kpi dict."""
        self.kpi_data = kpi

        # Hide placeholder
        self.kpi_placeholder.pack_forget()

        # Clear existing tiles
        for w in self.kpi_frame.winfo_children():
            w.destroy()
        for w in self.kpi_detail_frame.winfo_children():
            w.destroy()

        # Configure grid columns
        for c in range(3):
            self.kpi_frame.columnconfigure(c, weight=1)

        # Row 0: Quality Score, Completeness, Integrity
        score = kpi["quality_score"]
        self._create_kpi_tile(self.kpi_frame, 0, 0, "Quality Score",
                              f"{score}/100",
                              self._status_color(score, 80, 60))

        comp = kpi["completeness_pct"]
        self._create_kpi_tile(self.kpi_frame, 0, 1, "Completeness",
                              f"{comp:.1f}%",
                              self._status_color(comp, 95, 80))

        integrity = kpi["integrity"]["status"]
        if integrity == "PASS":
            i_color = COLORS["success"]
        elif integrity == "N/A":
            i_color = COLORS["light_gray"]
        else:
            i_color = COLORS["secondary"]
        self._create_kpi_tile(self.kpi_frame, 0, 2, "Integrity", integrity, i_color)

        # Row 1: Missing Values, Timestamp Issues, Outliers
        mv = kpi["missing_values"]
        self._create_kpi_tile(self.kpi_frame, 1, 0, "Missing Values",
                              str(mv),
                              self._status_color(mv, 0, 10, higher_is_better=False))

        ti = kpi["timestamp_issues"]
        self._create_kpi_tile(self.kpi_frame, 1, 1, "Timestamp Issues",
                              str(ti),
                              self._status_color(ti, 0, 5, higher_is_better=False))

        outliers = kpi["outliers"]
        total = outliers["total"]
        self._create_kpi_tile(self.kpi_frame, 1, 2, "Outliers",
                              str(total),
                              self._status_color(total, 0, 5, higher_is_better=False))

        # Row 2: Value Range, Processing Accuracy, Outlier Breakdown
        vr = kpi["value_range"]
        self._create_kpi_tile(self.kpi_frame, 2, 0, "Value Range",
                              f"{vr['min']:.1f} - {vr['max']:.1f}",
                              COLORS["primary"])

        acc = kpi["processing_accuracy_pct"]
        self._create_kpi_tile(self.kpi_frame, 2, 1, "Processing Accuracy",
                              f"{acc:.1f}%",
                              self._status_color(acc, 95, 80))

        orig_pct = kpi.get("original_data_pct", 100.0)
        self._create_kpi_tile(self.kpi_frame, 2, 2, "Original Data",
                              f"{orig_pct:.1f}%",
                              self._status_color(orig_pct, 90, 70))

        # Row 3: Untrustworthiness Score (spanning full width)
        untrust = kpi.get("untrustworthiness", {})
        if untrust:
            u_pct = untrust["pct"]
            tier = untrust["color_tier"]
            tier_colors = {
                "green": COLORS["success"],
                "yellow": COLORS["warning"],
                "orange": COLORS["orange"],
                "red": COLORS["secondary"],
            }
            u_color = tier_colors.get(tier, COLORS["primary"])

            untrust_frame = tk.Frame(self.kpi_frame, bg=COLORS["white"],
                                     relief=tk.FLAT, bd=1,
                                     highlightbackground=COLORS["light_gray"],
                                     highlightthickness=1)
            untrust_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

            # Colored top bar
            bar = tk.Frame(untrust_frame, bg=u_color, height=4)
            bar.pack(fill=tk.X)

            untrust_content = tk.Frame(untrust_frame, bg=COLORS["white"])
            untrust_content.pack(fill=tk.X, padx=12, pady=8)

            tk.Label(untrust_content, text="Untrustworthiness Score",
                     font=("Segoe UI", 8),
                     bg=COLORS["white"], fg=COLORS["light_gray"]).pack(anchor=tk.W)

            score_row = tk.Frame(untrust_content, bg=COLORS["white"])
            score_row.pack(fill=tk.X)

            tk.Label(score_row, text=f"{u_pct}%",
                     font=("Segoe UI", 20, "bold"),
                     bg=COLORS["white"], fg=u_color).pack(side=tk.LEFT)

            tk.Label(score_row, text=f"  {untrust['rating']}",
                     font=("Segoe UI", 12, "bold"),
                     bg=COLORS["white"], fg=u_color).pack(side=tk.LEFT, pady=(4, 0))

            detail_str = (f"{u_pct}% of data flagged as unreliable "
                          f"({untrust['flagged']} out of {untrust['total']} records)")
            tk.Label(untrust_content, text=detail_str,
                     font=("Segoe UI", 9),
                     bg=COLORS["white"], fg=COLORS["text_dark"]).pack(anchor=tk.W, pady=(2, 0))

            # Breakdown chips
            bk = untrust.get("breakdown", {})
            if any(v > 0 for v in bk.values()):
                chip_frame = tk.Frame(untrust_content, bg=COLORS["white"])
                chip_frame.pack(anchor=tk.W, pady=(4, 0))
                chip_labels = {
                    "missing_values": "Missing",
                    "outliers": "Outliers",
                    "timestamp_gaps": "Gaps",
                    "negatives": "Negatives",
                    "duplicates": "Duplicates",
                    "spikes": "Spikes",
                }
                for key, label in chip_labels.items():
                    count = bk.get(key, 0)
                    if count > 0:
                        chip = tk.Label(chip_frame, text=f" {label}: {count} ",
                                        font=("Segoe UI", 8),
                                        bg=COLORS["bg"], fg=COLORS["text_dark"],
                                        relief=tk.FLAT, padx=4, pady=1)
                        chip.pack(side=tk.LEFT, padx=(0, 6))

        # Detailed results
        if kpi.get("detailed_results"):
            tk.Label(self.kpi_detail_frame, text="Detailed Validation Results",
                     font=("Segoe UI", 10, "bold"),
                     bg=COLORS["white"], fg=COLORS["primary"]).pack(anchor=tk.W, pady=(10, 5))

            detail_text = tk.Text(self.kpi_detail_frame, height=8, width=90,
                                  font=("Consolas", 9),
                                  bg=COLORS["bg"], fg=COLORS["text_dark"],
                                  relief=tk.FLAT, padx=10, pady=10)
            detail_text.pack(fill=tk.X)

            detail_text.tag_configure("pass", foreground=COLORS["success"])
            detail_text.tag_configure("warn", foreground=COLORS["warning"])
            detail_text.tag_configure("fail", foreground=COLORS["secondary"])

            for r in kpi["detailed_results"]:
                tag = r["status"].lower()
                detail_text.insert(tk.END, f"  [{r['status']}] ", tag)
                detail_text.insert(tk.END, f"{r['name']}: {r['details']}\n")

            detail_text.config(state=tk.DISABLED)

        # Recommendations
        recs = kpi.get("recommendations", [])
        if recs:
            tk.Label(self.kpi_detail_frame, text="Data Quality Recommendations",
                     font=("Segoe UI", 10, "bold"),
                     bg=COLORS["white"], fg=COLORS["primary"]).pack(anchor=tk.W, pady=(12, 5))

            rec_frame = tk.Frame(self.kpi_detail_frame, bg=COLORS["bg"],
                                 padx=12, pady=10)
            rec_frame.pack(fill=tk.X)

            priority_colors = {
                1: COLORS["secondary"],
                2: COLORS["warning"],
                3: COLORS["primary"],
                4: COLORS["light_gray"],
            }
            priority_labels = {1: "CRITICAL", 2: "IMPORTANT", 3: "ADVISORY", 4: "INFO"}

            for r in recs:
                p = r["priority"]
                p_color = priority_colors.get(p, COLORS["light_gray"])
                p_label = priority_labels.get(p, "INFO")

                rec_item = tk.Frame(rec_frame, bg=COLORS["white"],
                                     relief=tk.FLAT, bd=1,
                                     highlightbackground=COLORS["light_gray"],
                                     highlightthickness=1)
                rec_item.pack(fill=tk.X, pady=3)

                # Left color accent bar
                accent = tk.Frame(rec_item, bg=p_color, width=4)
                accent.pack(side=tk.LEFT, fill=tk.Y)

                rec_content = tk.Frame(rec_item, bg=COLORS["white"], padx=10, pady=6)
                rec_content.pack(side=tk.LEFT, fill=tk.X, expand=True)

                header_frame = tk.Frame(rec_content, bg=COLORS["white"])
                header_frame.pack(anchor=tk.W)

                tk.Label(header_frame, text=f"[{p_label}]",
                         font=("Segoe UI", 8, "bold"),
                         bg=COLORS["white"], fg=p_color).pack(side=tk.LEFT)

                tk.Label(header_frame, text=f"  {r['category']}",
                         font=("Segoe UI", 9, "bold"),
                         bg=COLORS["white"], fg=COLORS["text_dark"]).pack(side=tk.LEFT)

                tk.Label(rec_content, text=r["message"],
                         font=("Segoe UI", 9),
                         bg=COLORS["white"], fg=COLORS["text_dark"],
                         wraplength=700, justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 0))

    def create_statistics_section(self):
        """Create the statistical analysis section."""
        content = self.create_card(self.content_frame, "8. Statistical Analysis")

        # Checkbox grid
        checkbox_frame = tk.Frame(content, bg=COLORS["white"])
        checkbox_frame.pack(fill=tk.X, pady=(0, 10))

        self.stat_vars = {
            "total_kwh": tk.BooleanVar(value=True),
            "mean_kw": tk.BooleanVar(value=True),
            "median_kw": tk.BooleanVar(value=True),
            "std_kw": tk.BooleanVar(value=True),
            "min_max_kw": tk.BooleanVar(value=True),
            "peak_times": tk.BooleanVar(value=True),
            "monthly_totals": tk.BooleanVar(value=True),
            "daily_avg_kwh": tk.BooleanVar(value=True),
            "seasonal_profile": tk.BooleanVar(value=True),
            "peak_analysis": tk.BooleanVar(value=True),
            "frequency_histogram": tk.BooleanVar(value=True),
            "cumulative_distribution": tk.BooleanVar(value=True),
            "peak_hour_frequency": tk.BooleanVar(value=True),
        }

        checkbox_labels = {
            "total_kwh": "Total Energy (kWh)",
            "mean_kw": "Mean Power (kW)",
            "median_kw": "Median Power (kW)",
            "std_kw": "Std Deviation (kW)",
            "min_max_kw": "Min / Max Power",
            "peak_times": "Peak / Off-Peak Times",
            "monthly_totals": "Monthly Totals",
            "daily_avg_kwh": "Daily Average (kWh)",
            "seasonal_profile": "Seasonal Weekly Profile",
            "peak_analysis": "Peak Consumption Analysis",
            "frequency_histogram": "Frequency Histogram",
            "cumulative_distribution": "Cumulative Distribution",
            "peak_hour_frequency": "Peak Event Frequency (24h)",
        }

        keys = list(checkbox_labels.keys())
        for i, key in enumerate(keys):
            row, col = divmod(i, 3)
            cb = tk.Checkbutton(checkbox_frame, text=checkbox_labels[key],
                                variable=self.stat_vars[key],
                                font=("Segoe UI", 9),
                                bg=COLORS["white"], fg=COLORS["text_dark"],
                                activebackground=COLORS["white"],
                                selectcolor=COLORS["white"])
            cb.grid(row=row, column=col, sticky=tk.W, padx=10, pady=2)

        # Button row
        btn_row = tk.Frame(content, bg=COLORS["white"])
        btn_row.pack(fill=tk.X, pady=5)

        select_all_btn = ModernButton(btn_row, "Select All",
                                       command=self._stats_select_all,
                                       bg=COLORS["light_gray"],
                                       fg=COLORS["primary"],
                                       width=110, height=35)
        select_all_btn.pack(side=tk.LEFT, padx=5)

        clear_all_btn = ModernButton(btn_row, "Clear All",
                                      command=self._stats_clear_all,
                                      bg=COLORS["light_gray"],
                                      fg=COLORS["primary"],
                                      width=110, height=35)
        clear_all_btn.pack(side=tk.LEFT, padx=5)

        self.run_analysis_btn = ModernButton(btn_row, "Run Analysis",
                                              command=self.run_analysis,
                                              bg=COLORS["primary"],
                                              width=150, height=35)
        self.run_analysis_btn.pack(side=tk.LEFT, padx=5)

        self.generate_pdf_btn = ModernButton(btn_row, "Generate PDF Report",
                                              command=self.generate_report,
                                              bg=COLORS["secondary"],
                                              width=180, height=35)
        self.generate_pdf_btn.pack(side=tk.LEFT, padx=5)

        # Results text area
        self.stats_text = tk.Text(content, height=10, width=100,
                                   font=("Consolas", 9),
                                   bg=COLORS["bg"], fg=COLORS["text_dark"],
                                   relief=tk.FLAT, padx=10, pady=10)
        self.stats_text.pack(fill=tk.X, pady=(10, 0))
        self.stats_text.tag_configure("header", font=("Consolas", 10, "bold"),
                                       foreground=COLORS["primary"])
        self.stats_text.tag_configure("metric", font=("Consolas", 9, "bold"),
                                       foreground=COLORS["text_dark"])
        self.stats_text.config(state=tk.DISABLED)

        # Chart display frame
        self.chart_frame = tk.Frame(content, bg=COLORS["white"])
        self.chart_frame.pack(fill=tk.X, pady=5)

        # Keep references to chart images to prevent GC
        self._chart_images = []

    def _display_chart(self, chart_bytes):
        """Display a chart PNG in the chart_frame."""
        img = Image.open(io.BytesIO(chart_bytes))
        display_width = 750
        ratio = display_width / img.width
        display_height = int(img.height * ratio)
        img = img.resize((display_width, display_height),
                          Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._chart_images.append(photo)
        label = tk.Label(self.chart_frame, image=photo, bg=COLORS["white"])
        label.pack(pady=3)

    def _stats_select_all(self):
        for var in self.stat_vars.values():
            var.set(True)

    def _stats_clear_all(self):
        for var in self.stat_vars.values():
            var.set(False)

    def run_analysis(self):
        """Run statistical analysis on transformed data."""
        if self.transformed_df is None:
            messagebox.showwarning("Warning",
                                    "Please transform data first (Step 4).")
            return

        # Collect selected metrics
        selected = [k for k, v in self.stat_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning("Warning", "Please select at least one metric.")
            return

        self.update_progress(20, "Running statistical analysis...")

        try:
            self.stats_result = run_statistical_analysis(
                self.transformed_df, self.hours_per_interval, selected)

            self.update_progress(60, "Formatting results...")

            # Display summary in stats_text
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)

            yearly = self.stats_result.get("yearly", {})
            for col_name, col_stats in yearly.items():
                self.stats_text.insert(tk.END, f"\n{col_name}\n", "header")
                self.stats_text.insert(tk.END, "-" * 40 + "\n")

                if "total_kwh" in selected:
                    self.stats_text.insert(tk.END, "  Total Energy: ", "metric")
                    self.stats_text.insert(tk.END, f"{col_stats['total_kwh']:,.2f} kWh\n")
                if "mean_kw" in selected:
                    self.stats_text.insert(tk.END, "  Mean Power: ", "metric")
                    self.stats_text.insert(tk.END, f"{col_stats['mean_kw']:.2f} kW\n")
                if "median_kw" in selected:
                    self.stats_text.insert(tk.END, "  Median Power: ", "metric")
                    self.stats_text.insert(tk.END, f"{col_stats['median_kw']:.2f} kW\n")
                if "std_kw" in selected:
                    self.stats_text.insert(tk.END, "  Std Deviation: ", "metric")
                    self.stats_text.insert(tk.END, f"{col_stats['std_kw']:.2f} kW\n")
                if "min_max_kw" in selected:
                    self.stats_text.insert(tk.END, "  Min / Max: ", "metric")
                    self.stats_text.insert(tk.END,
                        f"{col_stats['min_kw']:.2f} / {col_stats['max_kw']:.2f} kW\n")
                if "peak_times" in selected:
                    self.stats_text.insert(tk.END, "  Peak Time: ", "metric")
                    self.stats_text.insert(tk.END, f"{col_stats['peak_timestamp']}\n")
                    self.stats_text.insert(tk.END, "  Off-Peak Time: ", "metric")
                    self.stats_text.insert(tk.END, f"{col_stats['off_peak_timestamp']}\n")
                if "daily_avg_kwh" in selected:
                    self.stats_text.insert(tk.END, "  Daily Average: ", "metric")
                    self.stats_text.insert(tk.END, f"{col_stats['daily_avg_kwh']:,.2f} kWh\n")
                if "monthly_totals" in selected:
                    self.stats_text.insert(tk.END, "\n  Monthly Totals (kWh):\n", "metric")
                    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    monthly = col_stats.get("monthly_totals", {})
                    for m in range(1, 13):
                        val = monthly.get(m, 0)
                        self.stats_text.insert(tk.END,
                            f"    {month_names[m-1]}: {val:>10,.2f}\n")

            # Peak analysis text display
            if "peak_analysis" in selected:
                peaks = self.stats_result.get("peaks", {})
                for col_name, peak_data in peaks.items():
                    self.stats_text.insert(tk.END,
                        f"\nPeak Consumption Analysis — {col_name}\n", "header")
                    self.stats_text.insert(tk.END, "=" * 50 + "\n")

                    top_peaks = peak_data.get("top_peaks", [])
                    if not top_peaks:
                        self.stats_text.insert(tk.END,
                            "  No significant peaks detected.\n")
                        continue

                    self.stats_text.insert(tk.END, "\n  Top Peaks:\n", "metric")
                    # Parse grid capacity for % display
                    try:
                        _grid_cap = float(self.grid_capacity_var.get().strip())
                    except (ValueError, TypeError):
                        _grid_cap = None
                    for p in top_peaks:
                        grid_pct_str = ""
                        if _grid_cap and _grid_cap > 0:
                            grid_pct = p['value'] / _grid_cap * 100
                            grid_pct_str = f"  ({grid_pct:.1f}% of grid)"
                        self.stats_text.insert(tk.END,
                            f"    #{p['rank']}  {p['value']:>8,.1f} kW{grid_pct_str}  "
                            f"{p['timestamp']}  {p['day_of_week']:<9s}  "
                            f"Duration: {p['duration_hours']:.1f}h\n")

                    chars = peak_data.get("characteristics", {})
                    thresholds = peak_data.get("thresholds", {})
                    patterns = peak_data.get("patterns", {})
                    clustering = chars.get("clustering", {})

                    self.stats_text.insert(tk.END,
                        "\n  Characteristics:\n", "metric")
                    self.stats_text.insert(tk.END,
                        f"    Peaks detected:      {patterns.get('total_peaks_detected', 0)}\n"
                        f"    Avg duration:         {chars.get('avg_duration_hours', 0):.1f} hours\n"
                        f"    Avg rise rate:        {chars.get('avg_rise_rate', 0):,.1f} kW/hour\n"
                        f"    Avg fall rate:        {chars.get('avg_fall_rate', 0):,.1f} kW/hour\n"
                        f"    Clustered peaks:      {clustering.get('clustered_count', 0)}\n"
                        f"    Isolated peaks:       {clustering.get('isolated_count', 0)}\n")

                    self.stats_text.insert(tk.END,
                        "\n  Threshold Analysis:\n", "metric")
                    self.stats_text.insert(tk.END,
                        f"    90th percentile:      {thresholds.get('p90_value', 0):,.2f} kW\n"
                        f"    95th percentile:      {thresholds.get('p95_value', 0):,.2f} kW\n"
                        f"    Time above 90th pct:  {thresholds.get('time_above_p90_hours', 0):,.1f} hours\n"
                        f"    Time above 95th pct:  {thresholds.get('time_above_p95_hours', 0):,.1f} hours\n"
                        f"    Peak-to-avg ratio:    {thresholds.get('peak_to_avg_ratio', 0):.2f}x\n")

                    freq = patterns.get("peak_frequency", {})
                    self.stats_text.insert(tk.END,
                        "\n  Peak Frequency:\n", "metric")
                    self.stats_text.insert(tk.END,
                        f"    Readings above 90th:  {freq.get('above_90th', 0)}\n"
                        f"    Readings above 95th:  {freq.get('above_95th', 0)}\n"
                        f"    Readings above 99th:  {freq.get('above_99th', 0)}\n")

            # Histogram text display
            if "frequency_histogram" in selected:
                histogram = self.stats_result.get("histogram", {})
                for col_name, hist_data in histogram.items():
                    if not hist_data.get("counts"):
                        continue
                    self.stats_text.insert(tk.END,
                        f"\nFrequency Histogram — {col_name}\n", "header")
                    self.stats_text.insert(tk.END, "-" * 40 + "\n")
                    self.stats_text.insert(tk.END,
                        f"  Bins: {hist_data['n_bins']}  |  "
                        f"Bin width: {hist_data['bin_width']:,.2f} kW\n")
                    self.stats_text.insert(tk.END,
                        f"  Mean: {hist_data['mean']:,.2f} kW  |  "
                        f"Median: {hist_data['median']:,.2f} kW  |  "
                        f"Std: {hist_data['std']:,.2f} kW\n")

            # CDF text display
            if "cumulative_distribution" in selected:
                cdf_data = self.stats_result.get("cdf", {})
                for col_name, col_cdf in cdf_data.items():
                    pcts = col_cdf.get("percentiles", {})
                    if not pcts:
                        continue
                    self.stats_text.insert(tk.END,
                        f"\nCumulative Distribution — {col_name}\n", "header")
                    self.stats_text.insert(tk.END, "-" * 40 + "\n")
                    self.stats_text.insert(tk.END,
                        f"  Data points: {col_cdf.get('count', 0):,}\n")
                    self.stats_text.insert(tk.END,
                        f"  50th percentile (median): {pcts.get('p50', 0):,.2f} kW\n"
                        f"  90th percentile:          {pcts.get('p90', 0):,.2f} kW\n"
                        f"  95th percentile:          {pcts.get('p95', 0):,.2f} kW\n")

            # Peak hour frequency text display
            if "peak_hour_frequency" in selected:
                peak_hours = self.stats_result.get("peak_hours", {})
                for col_name, ph_data in peak_hours.items():
                    if ph_data.get("total_peaks", 0) == 0:
                        continue
                    self.stats_text.insert(tk.END,
                        f"\nPeak Event Frequency — {col_name}\n", "header")
                    self.stats_text.insert(tk.END, "=" * 50 + "\n")
                    self.stats_text.insert(tk.END,
                        f"  Threshold (P{int(ph_data['percentile_used'])}): "
                        f"{ph_data['threshold_value']:,.2f} kW\n")
                    self.stats_text.insert(tk.END,
                        f"  Total peak events: {ph_data['total_peaks']:,}\n")

                    peak_h = ph_data["peak_hour"]
                    self.stats_text.insert(tk.END,
                        f"  Peak events most common at: "
                        f"{peak_h:02d}:00 ({ph_data['peak_hour_count']} occurrences)\n")

                    pf = ph_data.get("peak_free_hours", [])
                    if pf:
                        if len(pf) <= 10:
                            pf_str = ", ".join(f"{h:02d}:00" for h in pf)
                        else:
                            pf_str = f"{len(pf)} hours"
                        self.stats_text.insert(tk.END,
                            f"  Peak-free hours: {pf_str}\n")

                    conc = ph_data.get("concentration", {})
                    if conc.get("pct", 0) > 0:
                        self.stats_text.insert(tk.END,
                            f"  Peak concentration: {conc['pct']:.0f}% of peaks "
                            f"between {conc['start_hour']:02d}:00 - "
                            f"{conc['end_hour']:02d}:59\n")

            self.stats_text.config(state=tk.DISABLED)

            self.update_progress(80, "Rendering charts...")

            # Render seasonal charts if selected
            self._chart_images.clear()
            for w in self.chart_frame.winfo_children():
                w.destroy()

            if "seasonal_profile" in selected:
                seasonal = self.stats_result.get("seasonal", {})
                for col_name, seasons_dict in seasonal.items():
                    for season_name in ["Winter", "Spring", "Summer", "Autumn"]:
                        profile_df = seasons_dict.get(season_name)
                        if profile_df is None:
                            continue

                        chart_bytes = generate_seasonal_chart(
                            profile_df, col_name, season_name)

                        # Convert to tkinter-displayable image
                        img = Image.open(io.BytesIO(chart_bytes))
                        # Scale down for display
                        display_width = 750
                        ratio = display_width / img.width
                        display_height = int(img.height * ratio)
                        img = img.resize((display_width, display_height),
                                          Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        self._chart_images.append(photo)

                        label = tk.Label(self.chart_frame, image=photo,
                                          bg=COLORS["white"])
                        label.pack(pady=3)

            # Render peak charts if selected
            if "peak_analysis" in selected:
                peaks = self.stats_result.get("peaks", {})
                for col_name, peak_data in peaks.items():
                    if not peak_data.get("top_peaks"):
                        continue

                    chart_generators = [
                        lambda: generate_peak_value_trend_chart(
                            peak_data["top_peaks"], col_name),
                        lambda: generate_peak_timeline_chart(
                            peak_data.get("peak_timeline", []), col_name),
                        lambda: generate_peak_duration_chart(
                            peak_data["top_peaks"], col_name),
                        lambda: generate_peak_heatmap(
                            peak_data.get("patterns", {}).get(
                                "hourly_distribution", {}),
                            peak_data.get("patterns", {}).get(
                                "daily_distribution", {}),
                            col_name),
                    ]

                    for gen in chart_generators:
                        chart_bytes = gen()
                        img = Image.open(io.BytesIO(chart_bytes))
                        display_width = 750
                        ratio = display_width / img.width
                        display_height = int(img.height * ratio)
                        img = img.resize((display_width, display_height),
                                          Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        self._chart_images.append(photo)

                        label = tk.Label(self.chart_frame, image=photo,
                                          bg=COLORS["white"])
                        label.pack(pady=3)

            # Render histogram charts
            if "frequency_histogram" in selected:
                histogram = self.stats_result.get("histogram", {})
                for col_name, hist_data in histogram.items():
                    if not hist_data.get("counts"):
                        continue
                    chart_bytes = generate_histogram_chart(hist_data, col_name)
                    self._display_chart(chart_bytes)

            # Render CDF charts
            if "cumulative_distribution" in selected:
                cdf_all = self.stats_result.get("cdf", {})
                for col_name, col_cdf in cdf_all.items():
                    if not col_cdf.get("values"):
                        continue
                    chart_bytes = generate_cdf_chart(
                        col_cdf, col_name, cdf_all)
                    self._display_chart(chart_bytes)
                    break  # Only one combined CDF chart needed

            # Render peak hour frequency charts
            if "peak_hour_frequency" in selected:
                peak_hours = self.stats_result.get("peak_hours", {})
                for col_name, ph_data in peak_hours.items():
                    if ph_data.get("total_peaks", 0) == 0:
                        continue
                    chart_bytes = generate_peak_hour_frequency_chart(ph_data, col_name)
                    self._display_chart(chart_bytes)

            self.update_progress(100, "Analysis complete!")

        except Exception as e:
            self.update_progress(0, "Error during analysis")
            messagebox.showerror("Error", f"Statistical analysis failed:\n{str(e)}")

    def generate_report(self):
        """Generate a branded PDF report."""
        if self.stats_result is None:
            messagebox.showwarning("Warning",
                                    "Please run analysis first.")
            return

        # Default filename based on site name
        site_name = self.site_name_var.get().strip()
        if site_name:
            default_name = build_output_filename(site_name, "EnergyAnalysis", "pdf")
        elif self.file_path:
            default_name = os.path.splitext(
                os.path.basename(self.file_path))[0] + "_report.pdf"
        else:
            default_name = "energy_report.pdf"

        if self.file_path:
            default_dir = os.path.dirname(self.file_path)
        else:
            default_dir = os.getcwd()

        save_path = filedialog.asksaveasfilename(
            initialdir=default_dir,
            initialfile=default_name,
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")])

        if not save_path:
            return

        self.update_progress(30, "Generating PDF report...")

        try:
            # Determine logo path
            logo_path = resource_path("Logo_new_white.png")
            if not os.path.exists(logo_path):
                logo_path = None

            # Build site_info dict
            site_info = None
            s_name = self.site_name_var.get().strip()
            g_cap = self.grid_capacity_var.get().strip()
            if s_name:
                site_info = {"site_name": s_name}
                try:
                    site_info["grid_capacity_kw"] = float(g_cap)
                except (ValueError, TypeError):
                    pass

            # Collect battery report data if available
            battery_report_data = None
            if (self.battery_result is not None
                    and hasattr(self, '_battery_sizer')
                    and self._battery_sizer is not None):
                try:
                    battery_report_data = (
                        self._battery_sizer.generate_report_data())
                except Exception:
                    pass  # Non-critical

            result_path = generate_pdf_report(
                output_path=save_path,
                stats_result=self.stats_result,
                kpi_data=self.kpi_data,
                logo_path=logo_path,
                quality_report=self.quality_report,
                site_info=site_info,
                battery_data=battery_report_data,
            )

            self.update_progress(100, "PDF report generated!")

            if messagebox.askyesno("Report Generated",
                                    f"PDF saved to:\n{result_path}\n\nOpen the file?"):
                os.startfile(result_path)

        except Exception as e:
            self.update_progress(0, "Error generating report")
            messagebox.showerror("Error", f"PDF generation failed:\n{str(e)}")

    def create_battery_section(self):
        """Create the battery dimensioning section."""
        content = self.create_card(self.content_frame,
                                   "9. Battery Dimensioning Analysis")

        # Enable checkbox
        self.battery_enabled_var = tk.BooleanVar(value=False)
        enable_cb = tk.Checkbutton(
            content, text="Enable battery dimensioning",
            variable=self.battery_enabled_var,
            font=("Segoe UI", 10, "bold"),
            bg=COLORS["white"], fg=COLORS["text_dark"],
            activebackground=COLORS["white"],
            selectcolor=COLORS["white"],
            command=self._toggle_battery_inputs)
        enable_cb.pack(anchor=tk.W, pady=(0, 5))

        # Tariff inputs frame
        self.battery_input_frame = tk.Frame(content, bg=COLORS["white"])
        self.battery_input_frame.pack(fill=tk.X, padx=20, pady=5)

        self.offtake_var = tk.StringVar(value="250")
        self.injection_var = tk.StringVar(value="50")
        self.peak_tariff_var = tk.StringVar(value="50")

        tariff_grid = tk.Frame(self.battery_input_frame, bg=COLORS["white"])
        tariff_grid.pack(fill=tk.X)

        labels_entries = [
            ("Offtake tariff (\u20ac/MWh):", self.offtake_var, 0),
            ("Injection tariff (\u20ac/MWh):", self.injection_var, 1),
            ("Peak tariff (\u20ac/kW):", self.peak_tariff_var, 2),
        ]
        for text, var, row in labels_entries:
            tk.Label(tariff_grid, text=text,
                     font=("Segoe UI", 10), bg=COLORS["white"]).grid(
                         row=row, column=0, sticky=tk.W, pady=3)
            ttk.Entry(tariff_grid, textvariable=var,
                      width=12, font=("Segoe UI", 10)).grid(
                          row=row, column=1, padx=10, pady=3, sticky=tk.W)

        # Button row
        btn_row = tk.Frame(self.battery_input_frame, bg=COLORS["white"])
        btn_row.pack(fill=tk.X, pady=10)

        self.battery_calc_btn = ModernButton(
            btn_row, "Calculate Battery Requirements",
            command=self.run_battery_analysis,
            bg=COLORS["secondary"], width=260, height=38)
        self.battery_calc_btn.pack(side=tk.LEFT, padx=5)

        self.battery_charts_btn = ModernButton(
            btn_row, "View Detailed Analysis",
            command=self.display_battery_charts,
            bg=COLORS["primary"], width=200, height=38)
        self.battery_charts_btn.pack(side=tk.LEFT, padx=5)
        self.battery_charts_btn.set_enabled(False)

        # Results frame
        self.battery_results_frame = tk.Frame(content, bg=COLORS["white"])
        self.battery_results_frame.pack(fill=tk.X, pady=5)

        # Chart display frame for battery
        self.battery_chart_frame = tk.Frame(content, bg=COLORS["white"])
        self.battery_chart_frame.pack(fill=tk.X, pady=5)

        # Keep references to prevent GC
        self._battery_chart_images = []

        # Start with inputs hidden
        self.battery_input_frame.pack_forget()

    def _toggle_battery_inputs(self):
        """Show/hide battery input fields based on checkbox."""
        if self.battery_enabled_var.get():
            self.battery_input_frame.pack(fill=tk.X, padx=20, pady=5)
        else:
            self.battery_input_frame.pack_forget()

    def run_battery_analysis(self):
        """Run battery dimensioning analysis."""
        if self.transformed_df is None:
            messagebox.showwarning("Warning",
                                   "Please transform data first (Step 5).")
            return

        # Check for production data
        if "Production (kW)" not in self.transformed_df.columns:
            messagebox.showinfo(
                "Production Data Required",
                "Battery dimensioning requires both consumption and "
                "production data.\n\nThis feature is only available for "
                "sites with solar/renewable generation.")
            return

        # Parse tariffs
        try:
            offtake = float(self.offtake_var.get().strip())
            injection = float(self.injection_var.get().strip())
            peak = float(self.peak_tariff_var.get().strip())
        except (ValueError, TypeError):
            messagebox.showerror("Error",
                                 "Please enter valid numeric tariff values.")
            return

        tariffs = {"offtake": offtake, "injection": injection, "peak": peak}

        self.update_progress(10, "Validating data for battery analysis...")

        try:
            sizer = BatterySizer(self.transformed_df,
                                 self.hours_per_interval, tariffs)

            # Validate
            validation = sizer.validate()
            if not validation["valid"]:
                messagebox.showinfo("Cannot Proceed", validation["message"])
                self.update_progress(0, "Battery analysis cancelled")
                return

            self.update_progress(30, "Calculating daily metrics...")
            self.battery_result = sizer.run_analysis()

            self.update_progress(70, "Computing recommendations...")

            # Show warning if applicable
            if validation.get("warning"):
                messagebox.showinfo("Data Coverage Warning",
                                    validation["warning"])

            # Check zero storable
            rec = self.battery_result["recommendations"]
            if rec.get("zero_storable"):
                messagebox.showinfo(
                    "No Excess Production",
                    "No excess production detected. Battery storage "
                    "would not provide savings with current production "
                    "levels.")
                self.update_progress(0, "No storable energy found")
                return

            # Display results
            self._display_battery_results(rec,
                                          self.battery_result["savings"])

            # Store sizer for chart generation and report
            self._battery_sizer = sizer

            self.battery_charts_btn.set_enabled(True)
            self.update_progress(100, "Battery analysis complete!")

        except Exception as e:
            self.update_progress(0, "Error in battery analysis")
            messagebox.showerror("Error",
                                 f"Battery analysis failed:\n{str(e)}")

    def _display_battery_results(self, rec, savings):
        """Display battery sizing results in the GUI."""
        # Clear previous results
        for w in self.battery_results_frame.winfo_children():
            w.destroy()

        # Results container with border
        container = tk.Frame(self.battery_results_frame, bg=COLORS["white"],
                             relief=tk.FLAT, bd=1,
                             highlightbackground=COLORS["light_gray"],
                             highlightthickness=1)
        container.pack(fill=tk.X, padx=5, pady=5)

        # Title bar
        title_bar = tk.Frame(container, bg=COLORS["primary"])
        title_bar.pack(fill=tk.X)
        tk.Label(title_bar, text="BATTERY SIZING RECOMMENDATIONS",
                 font=("Segoe UI", 11, "bold"),
                 bg=COLORS["primary"], fg=COLORS["white"],
                 padx=15, pady=8).pack(side=tk.LEFT)

        # Results grid
        results = tk.Frame(container, bg=COLORS["white"], padx=15, pady=10)
        results.pack(fill=tk.X)

        # Sizing recommendations
        sizing_data = [
            ("Maximum capacity needed:", f"{rec['max_capacity']:,.1f} kWh"),
            ("Average capacity needed:", f"{rec['avg_capacity']:,.1f} kWh"),
            ("Recommended capacity:", f"{rec['recommended_capacity']:,.1f} kWh"),
            ("Recommended power rating:", f"{rec['recommended_power']:,.1f} kW"),
        ]

        for i, (label, value) in enumerate(sizing_data):
            tk.Label(results, text=label, font=("Segoe UI", 10),
                     bg=COLORS["white"], fg=COLORS["text_dark"]).grid(
                         row=i, column=0, sticky=tk.W, pady=2)
            color = COLORS["secondary"] if i == 2 else COLORS["primary"]
            tk.Label(results, text=value,
                     font=("Segoe UI", 10, "bold"),
                     bg=COLORS["white"], fg=color).grid(
                         row=i, column=1, sticky=tk.E, padx=(20, 0), pady=2)

        # Separator
        sep = tk.Frame(container, bg=COLORS["light_gray"], height=1)
        sep.pack(fill=tk.X, padx=15)

        # Savings section
        sav_title = tk.Frame(container, bg=COLORS["primary"])
        sav_title.pack(fill=tk.X)
        tk.Label(sav_title, text="ESTIMATED ANNUAL SAVINGS",
                 font=("Segoe UI", 11, "bold"),
                 bg=COLORS["primary"], fg=COLORS["white"],
                 padx=15, pady=8).pack(side=tk.LEFT)

        sav_frame = tk.Frame(container, bg=COLORS["white"], padx=15, pady=10)
        sav_frame.pack(fill=tk.X)

        savings_data = [
            ("With recommended battery:",
             f"\u20ac{savings['annual_savings']:,.0f}"),
            ("Energy arbitrage savings:",
             f"\u20ac{savings['energy_arbitrage']:,.0f}"),
            ("Peak demand reduction:",
             f"\u20ac{savings['peak_reduction']:,.0f}"),
            ("Self-consumption increase:",
             f"+{savings['self_consumption_increase']:.1f}%"),
        ]

        for i, (label, value) in enumerate(savings_data):
            tk.Label(sav_frame, text=label, font=("Segoe UI", 10),
                     bg=COLORS["white"], fg=COLORS["text_dark"]).grid(
                         row=i, column=0, sticky=tk.W, pady=2)
            color = COLORS["success"] if i == 0 else COLORS["primary"]
            tk.Label(sav_frame, text=value,
                     font=("Segoe UI", 10, "bold"),
                     bg=COLORS["white"], fg=color).grid(
                         row=i, column=1, sticky=tk.E, padx=(20, 0), pady=2)

    def display_battery_charts(self):
        """Display all battery dimensioning charts."""
        if not hasattr(self, '_battery_sizer') or self._battery_sizer is None:
            messagebox.showwarning("Warning",
                                   "Please run battery analysis first.")
            return

        self.update_progress(20, "Generating battery charts...")

        # Clear previous charts
        self._battery_chart_images.clear()
        for w in self.battery_chart_frame.winfo_children():
            w.destroy()

        try:
            sizer = self._battery_sizer
            chart_generators = [
                ("Typical Daily Profile", sizer.generate_average_day_profile),
                ("Monthly Storage Requirements",
                 sizer.generate_monthly_storage_chart),
                ("Annual Storage Pattern",
                 sizer.generate_annual_storage_pattern),
                ("Storage Duration Curve", sizer.generate_duration_curve),
                ("Monthly Savings Potential",
                 sizer.generate_monthly_savings_chart),
                ("Energy Flow Analysis",
                 sizer.generate_self_consumption_chart),
            ]

            for i, (title, gen_func) in enumerate(chart_generators):
                self.update_progress(
                    20 + int(70 * (i + 1) / len(chart_generators)),
                    f"Rendering: {title}...")

                chart_bytes = gen_func()
                img = Image.open(io.BytesIO(chart_bytes))
                display_width = 750
                ratio = display_width / img.width
                display_height = int(img.height * ratio)
                img = img.resize((display_width, display_height),
                                 Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._battery_chart_images.append(photo)

                label = tk.Label(self.battery_chart_frame, image=photo,
                                 bg=COLORS["white"])
                label.pack(pady=3)

            self.update_progress(100, "Battery charts rendered!")

        except Exception as e:
            self.update_progress(0, "Error rendering charts")
            messagebox.showerror("Error",
                                 f"Chart generation failed:\n{str(e)}")

    def create_tools_section(self):
        """Create the tools section with CLI and Claude Code buttons."""
        content = self.create_card(self.content_frame, "10. Developer Tools")

        tools_row = tk.Frame(content, bg=COLORS["white"])
        tools_row.pack(fill=tk.X)

        # Open Command Line button
        cli_btn = ModernButton(tools_row, "Open Command Line",
                              command=self.open_cli,
                              bg=COLORS["text_dark"], width=180)
        cli_btn.pack(side=tk.LEFT, padx=5)

        tk.Label(tools_row, text="Run the CLI version of Spartacus",
                font=("Segoe UI", 9), bg=COLORS["white"],
                fg=COLORS["light_gray"]).pack(side=tk.LEFT, padx=10)

        # Open Claude Code button
        claude_btn = ModernButton(tools_row, "Open Claude Code",
                                  command=self.open_claude_code,
                                  bg="#6B46C1", width=180)  # Purple for Claude
        claude_btn.pack(side=tk.LEFT, padx=(30, 5))

        tk.Label(tools_row, text="Launch Claude Code in project directory",
                font=("Segoe UI", 9), bg=COLORS["white"],
                fg=COLORS["light_gray"]).pack(side=tk.LEFT, padx=10)

    def create_footer(self, parent):
        """Create the footer."""
        footer = tk.Frame(parent, bg=COLORS["primary"], height=30)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)

        footer_text = tk.Label(footer,
                              text="Spartacus v1.0 | ReVolta srl | www.revolta.energy",
                              font=("Segoe UI", 8),
                              bg=COLORS["primary"], fg=COLORS["light_gray"])
        footer_text.pack(pady=7)

    # ============ Event Handlers ============

    def browse_file(self):
        """Open file browser dialog."""
        filetypes = [
            ("All supported", "*.csv;*.xlsx;*.xls;*.txt;*.tsv"),
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx;*.xls"),
            ("Text files", "*.txt;*.tsv"),
        ]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, path)

    def load_file(self):
        """Load the selected file."""
        path = clean_path(self.file_entry.get())
        if not path:
            messagebox.showerror("Error", "Please select a file first.")
            return

        if not os.path.isfile(path):
            messagebox.showerror("Error", f"File not found: {path}")
            return

        self.update_progress(10, "Loading file...")

        try:
            # Load file (suppress rich console output to avoid encoding issues)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.df, self.metadata = load_file(path)
            self.file_path = path

            self.update_progress(40, "Analyzing columns...")

            # Analyze columns
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.analysis = analyze_columns(self.df)

            # Detect granularity with confidence
            self.update_progress(50, "Detecting data granularity...")
            date_candidates = [i for i, a in enumerate(self.analysis) if a["type"] == "datetime"]
            detection_result = None

            if date_candidates:
                date_col_idx = date_candidates[0]
                date_fmt = self.analysis[date_col_idx].get("date_format", "auto")
                if date_fmt == "auto":
                    sample_dates = pd.to_datetime(self.df.iloc[:, date_col_idx],
                                                  dayfirst=True, errors="coerce")
                else:
                    sample_dates = pd.to_datetime(self.df.iloc[:, date_col_idx],
                                                  format=date_fmt, errors="coerce")
                detection_result = detect_granularity_with_confidence(sample_dates)

                # Check confidence threshold (0.7 = 70%)
                if detection_result["confidence"] >= 0.7:
                    # High confidence - auto-accept
                    self.granularity_label = detection_result["label"]
                    self.hours_per_interval = detection_result["hours_per_interval"]
                    self.granularity_confidence = detection_result["confidence"]
                    self.granularity_auto_detected = True
                else:
                    # Low confidence - show dialog
                    self.granularity_auto_detected = False
                    dialog = GranularityDialog(self.root, detection_result)
                    self.root.wait_window(dialog)
                    if dialog.result:
                        self.granularity_label = dialog.result["label"]
                        self.hours_per_interval = dialog.result["hours_per_interval"]
                    else:
                        # User cancelled - use detected value anyway
                        self.granularity_label = detection_result["label"]
                        self.hours_per_interval = detection_result["hours_per_interval"]
            else:
                # No date column found - show dialog
                self.granularity_auto_detected = False
                dialog = GranularityDialog(self.root, None)
                self.root.wait_window(dialog)
                if dialog.result:
                    self.granularity_label = dialog.result["label"]
                    self.hours_per_interval = dialog.result["hours_per_interval"]

            self.update_progress(70, "Updating display...")

            # Update UI
            self.update_file_info()
            self.update_preview()
            self.update_analysis_display()
            self.update_column_combos()

            self.update_progress(100, "File loaded successfully!")

            # Enable transform button
            self.transform_btn.set_enabled(True)

        except Exception as e:
            self.update_progress(0, "Error loading file")
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")

    def update_file_info(self):
        """Update file info display."""
        info = f"File: {os.path.basename(self.file_path)} | "
        info += f"Rows: {self.metadata['row_count']} | "
        info += f"Columns: {self.metadata['col_count']}"
        if self.metadata.get("encoding"):
            info += f" | Encoding: {self.metadata['encoding']}"
        self.file_info_label.config(text=info)

    def update_preview(self):
        """Update the preview treeview."""
        # Clear existing
        self.preview_tree.delete(*self.preview_tree.get_children())
        for col in self.preview_tree["columns"]:
            self.preview_tree.heading(col, text="")

        # Configure columns
        columns = ["#"] + list(self.df.columns)
        self.preview_tree["columns"] = columns
        self.preview_tree["show"] = "headings"

        for col in columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=100, minwidth=50)

        # Add rows
        for i, (_, row) in enumerate(self.df.head(10).iterrows()):
            values = [str(i)] + [str(v)[:30] for v in row.values]
            self.preview_tree.insert("", tk.END, values=values)

    def update_analysis_display(self):
        """Update the analysis text display."""
        self.analysis_text.config(state=tk.NORMAL)
        self.analysis_text.delete(1.0, tk.END)

        text = "Column Analysis:\n"
        for i, info in enumerate(self.analysis):
            col_type = info["type"]
            details = ""
            if col_type == "datetime":
                details = f"({info.get('date_desc', 'auto')})"
            elif col_type == "numeric":
                if "unit_guess" in info:
                    details = f"(likely {info['unit_guess']})"

            text += f"  [{i}] {info['name']}: {col_type} {details}\n"

        text += f"\nDetected granularity: {self.granularity_label}"

        self.analysis_text.insert(tk.END, text)
        self.analysis_text.config(state=tk.DISABLED)

        # Update granularity label
        self.granularity_label_widget.config(text=f"Detected granularity: {self.granularity_label}")

    def update_column_combos(self):
        """Update column selection comboboxes."""
        columns = [f"{i}: {col}" for i, col in enumerate(self.df.columns)]
        columns_with_none = ["none"] + columns

        self.date_col_combo["values"] = columns
        self.cons_col_combo["values"] = columns
        self.prod_col_combo["values"] = columns_with_none

        # Auto-select based on analysis
        date_candidates = [i for i, a in enumerate(self.analysis) if a["type"] == "datetime"]
        numeric_candidates = [i for i, a in enumerate(self.analysis) if a["type"] == "numeric"]

        if date_candidates:
            self.date_col_var.set(columns[date_candidates[0]])
        else:
            self.date_col_var.set(columns[0] if columns else "")

        if numeric_candidates:
            self.cons_col_var.set(columns[numeric_candidates[0]])
            # Set unit based on detection
            cons_analysis = self.analysis[numeric_candidates[0]]
            detected_unit = cons_analysis.get("unit_guess", "kW")
            self.cons_unit_var.set(detected_unit)

            if len(numeric_candidates) > 1:
                self.prod_col_var.set(columns[numeric_candidates[1]])
                prod_analysis = self.analysis[numeric_candidates[1]]
                self.prod_unit_var.set(prod_analysis.get("unit_guess", "kW"))
            else:
                self.prod_col_var.set("none")
        else:
            self.cons_col_var.set(columns[1] if len(columns) > 1 else columns[0] if columns else "")
            self.prod_col_var.set("none")

    def get_selected_column_index(self, value):
        """Extract column index from combo value."""
        if value == "none" or not value:
            return None
        try:
            return int(value.split(":")[0])
        except:
            return None

    def transform_data(self):
        """Transform the loaded data."""
        if self.df is None:
            messagebox.showerror("Error", "Please load a file first.")
            return

        # Validate site information
        site_name = self.site_name_var.get().strip()
        if not site_name:
            messagebox.showerror("Error",
                "Please enter a Site Name in section 2 (Site Information).")
            return

        grid_cap_str = self.grid_capacity_var.get().strip()
        try:
            grid_cap = float(grid_cap_str)
            if grid_cap <= 0:
                raise ValueError
        except (ValueError, TypeError):
            messagebox.showerror("Error",
                "Please enter a valid positive number for Grid Connection "
                "Capacity (kW) in section 2 (Site Information).")
            return

        self.update_progress(20, "Transforming data...")

        try:
            date_col = self.get_selected_column_index(self.date_col_var.get())
            cons_col = self.get_selected_column_index(self.cons_col_var.get())
            prod_col = self.get_selected_column_index(self.prod_col_var.get())

            if date_col is None or cons_col is None:
                messagebox.showerror("Error", "Please select date and consumption columns.")
                return

            # Get date format
            date_format = self.analysis[date_col].get("date_format", "auto")

            self.update_progress(50, "Converting values...")

            # Suppress rich console output to avoid encoding issues
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.transformed_df = transform_data(
                    df=self.df,
                    date_col=date_col,
                    consumption_col=cons_col,
                    production_col=prod_col,
                    date_format=date_format,
                    consumption_unit=self.cons_unit_var.get(),
                    production_unit=self.prod_unit_var.get() if prod_col is not None else None,
                    hours_per_interval=self.hours_per_interval,
                )

            self.update_progress(100, "Transformation complete!")

            # Update results
            self.show_result(f"Transformation complete!\n"
                           f"  - Valid rows: {len(self.transformed_df)}\n"
                           f"  - Columns: {list(self.transformed_df.columns)}\n"
                           f"  - Date range: {self.transformed_df['Date & Time'].min()} to "
                           f"{self.transformed_df['Date & Time'].max()}")

            # Enable quality check
            self.quality_btn.set_enabled(True)
            self.export_btn.set_enabled(True)

        except Exception as e:
            self.update_progress(0, "Error during transformation")
            messagebox.showerror("Error", f"Transformation failed:\n{str(e)}")

    def run_quality_check(self):
        """Run quality check on transformed data."""
        if self.transformed_df is None:
            messagebox.showerror("Error", "Please transform data first.")
            return

        self.update_progress(30, "Running quality checks...")

        try:
            # Suppress rich console output to avoid encoding issues on Windows
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.quality_report = run_quality_check(
                    self.transformed_df,
                    self.granularity_label,
                    self.hours_per_interval
                )

            self.update_progress(80, "Computing KPI dashboard...")

            # Display report
            self.display_quality_report()

            # Run validation and display KPI dashboard
            if self.quality_report:
                cons_col = self.get_selected_column_index(self.cons_col_var.get())
                prod_col = self.get_selected_column_index(self.prod_col_var.get())
                kpi = run_validation(
                    df=self.transformed_df, quality_report=self.quality_report,
                    original_df=self.df,
                    consumption_col=cons_col, production_col=prod_col,
                )
                self.display_kpi_dashboard(kpi)

            self.update_progress(100, "Quality check complete!")

            # Enable corrections if issues found
            if self.quality_report:
                total_issues = (
                    self.quality_report.get("missing_timestamps", 0)
                    + sum(v["count"] for v in self.quality_report.get("missing_values", {}).values())
                    + len(self.quality_report.get("duplicates", []))
                    + len(self.quality_report.get("outliers", []))
                )
                if total_issues > 0:
                    self.correct_btn.set_enabled(True)

        except Exception as e:
            self.update_progress(0, "Error during quality check")
            messagebox.showerror("Error", f"Quality check failed:\n{str(e)}")

    def display_quality_report(self):
        """Display the quality report in results area."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        if not self.quality_report:
            self.results_text.insert(tk.END, "No quality issues found!", "success")
            self.results_text.config(state=tk.DISABLED)
            return

        report = self.quality_report

        # Header
        self.results_text.insert(tk.END, "QUALITY CHECK REPORT\n", "header")
        self.results_text.insert(tk.END, "=" * 50 + "\n\n")

        # Summary
        start, end = report.get("date_range", (None, None))
        if start and end:
            self.results_text.insert(tk.END, f"Date Range: {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}\n")
        self.results_text.insert(tk.END, f"Granularity: {report.get('granularity', 'unknown')}\n")
        self.results_text.insert(tk.END, f"Total Rows: {report.get('total_rows', 0)}\n")
        self.results_text.insert(tk.END, f"Expected Timestamps: {report.get('expected_timestamps', 0)}\n\n")

        # Issues
        self.results_text.insert(tk.END, "ISSUES FOUND:\n", "header")

        missing_ts = report.get("missing_timestamps", 0)
        if missing_ts > 0:
            self.results_text.insert(tk.END, f"  Missing Timestamps: {missing_ts}\n", "warning")
        else:
            self.results_text.insert(tk.END, f"  Missing Timestamps: 0\n", "success")

        missing_vals = sum(v["count"] for v in report.get("missing_values", {}).values())
        if missing_vals > 0:
            self.results_text.insert(tk.END, f"  Missing Values: {missing_vals}\n", "warning")
        else:
            self.results_text.insert(tk.END, f"  Missing Values: 0\n", "success")

        duplicates = len(report.get("duplicates", []))
        if duplicates > 0:
            self.results_text.insert(tk.END, f"  Duplicate Timestamps: {duplicates}\n", "warning")
        else:
            self.results_text.insert(tk.END, f"  Duplicate Timestamps: 0\n", "success")

        outliers = len(report.get("outliers", []))
        if outliers > 0:
            self.results_text.insert(tk.END, f"  Outliers: {outliers}\n", "warning")
        else:
            self.results_text.insert(tk.END, f"  Outliers: 0\n", "success")

        negatives = len(report.get("negatives", []))
        if negatives > 0:
            self.results_text.insert(tk.END, f"  Negative Values: {negatives}\n", "error")

        # Total
        total = missing_ts + missing_vals + duplicates + outliers
        self.results_text.insert(tk.END, f"\nTotal Issues: {total}\n",
                                 "error" if total > 0 else "success")

        self.results_text.config(state=tk.DISABLED)

    def apply_corrections(self):
        """Apply corrections with user choices via dialog."""
        if self.transformed_df is None or not self.quality_report:
            messagebox.showerror("Error", "Please run quality check first.")
            return

        # Create corrections dialog
        dialog = CorrectionDialog(self.root, self.quality_report)
        self.root.wait_window(dialog)

        if not dialog.result:
            return

        self.update_progress(20, "Applying corrections...")

        try:
            pre_correction_df = self.transformed_df.copy()
            df = self.transformed_df.copy()
            choices = dialog.result

            # Apply corrections based on choices
            # Duplicates
            dup_timestamps = set()
            for d in self.quality_report.get("duplicates", []):
                dup_timestamps.add(d["timestamp"])

            if choices.get("duplicates") == "first":
                df = df.drop_duplicates(subset="Date & Time", keep="first").reset_index(drop=True)
            elif choices.get("duplicates") == "last":
                df = df.drop_duplicates(subset="Date & Time", keep="last").reset_index(drop=True)
            elif choices.get("duplicates") == "average":
                value_cols = [c for c in df.columns if c not in ("Date & Time", "data_source")]
                agg_cols = {col: "mean" for col in value_cols}
                if "data_source" in df.columns:
                    agg_cols["data_source"] = "first"
                df = df.groupby("Date & Time", as_index=False).agg(agg_cols)
                df = df.sort_values("Date & Time").reset_index(drop=True)
                if "data_source" in df.columns and dup_timestamps:
                    mask = df["Date & Time"].isin(dup_timestamps)
                    df.loc[mask, "data_source"] = "duplicate_resolved"

            self.update_progress(40, "Filling gaps...")

            # Time gaps
            gaps = self.quality_report.get("gaps", [])
            if gaps and choices.get("gaps") != "skip":
                freq = pd.Timedelta(hours=self.hours_per_interval)
                value_cols = [c for c in df.columns if c not in ("Date & Time", "data_source")]
                new_rows = []
                max_fill_rows = len(df) * 2  # Safety cap: don't create more rows than 2x original

                for gap in gaps:
                    current = gap["from"] + freq
                    while current < gap["to"] and len(new_rows) < max_fill_rows:
                        row = {"Date & Time": current}
                        for col in value_cols:
                            row[col] = float("nan")
                        if "data_source" in df.columns:
                            row["data_source"] = "gap_filled"
                        new_rows.append(row)
                        current += freq

                if new_rows:
                    new_df = pd.DataFrame(new_rows)
                    df = pd.concat([df, new_df], ignore_index=True)
                    df = df.sort_values("Date & Time").reset_index(drop=True)

            self.update_progress(60, "Filling missing values...")

            # Missing values - fully vectorized approach
            value_cols = [c for c in df.columns if c not in ("Date & Time", "data_source")]
            missing_choice = choices.get("missing")
            if missing_choice and missing_choice != "skip":
                # Capture NaN mask before filling for data_source tracking
                nan_masks = {}
                for col in value_cols:
                    nan_masks[col] = df[col].isna().copy()

                if missing_choice == "previous_day":
                    # Vectorized: merge with shifted version of itself
                    df = df.set_index("Date & Time").sort_index()
                    for col in value_cols:
                        if df[col].isna().any():
                            # Try offsets: 1 day ago, 1 day ahead, 7 days ago
                            for days in [1, -1, 7, -7]:
                                if not df[col].isna().any():
                                    break
                                shifted_idx = df.index + pd.Timedelta(days=days)
                                fill_values = df[col].reindex(shifted_idx).values
                                df[col] = df[col].fillna(pd.Series(fill_values, index=df.index))
                    df = df.reset_index()
                    # Final fallback to interpolation
                    for col in value_cols:
                        df[col] = df[col].interpolate(method="linear", limit_direction="both")
                elif missing_choice == "interpolate":
                    for col in value_cols:
                        df[col] = df[col].interpolate(method="linear", limit_direction="both")
                elif missing_choice == "zero":
                    for col in value_cols:
                        df[col] = df[col].fillna(0)

                # Mark rows where NaN was filled as corrected
                if "data_source" in df.columns:
                    for col in value_cols:
                        if col in nan_masks:
                            was_nan = nan_masks[col]
                            # Align mask with current index (may have changed after set_index/reset_index)
                            common_idx = was_nan.index.intersection(df.index)
                            filled_mask = was_nan.loc[common_idx] & df.loc[common_idx, col].notna()
                            df.loc[filled_mask[filled_mask].index, "data_source"] = "missing_filled"

            self.update_progress(80, "Handling outliers...")

            # Outliers
            outliers = self.quality_report.get("outliers", [])
            outlier_rows = set()
            if choices.get("outliers") == "previous_day" and outliers:
                # For outliers, just use interpolation (faster and usually good enough)
                for o in outliers:
                    df.at[o["row"], o["column"]] = float("nan")
                    outlier_rows.add(o["row"])
                for col in value_cols:
                    df[col] = df[col].interpolate(method="linear", limit_direction="both")
            elif choices.get("outliers") == "cap":
                for o in outliers:
                    threshold = o["median"] * 3
                    if o["type"] == "high":
                        df.at[o["row"], o["column"]] = threshold
                    else:
                        df.at[o["row"], o["column"]] = o["median"] / 10
                    outlier_rows.add(o["row"])

            # Mark corrected outlier rows
            if "data_source" in df.columns and outlier_rows:
                valid_rows_idx = [r for r in outlier_rows if r in df.index]
                if valid_rows_idx:
                    df.loc[valid_rows_idx, "data_source"] = "outlier_corrected"

            self.transformed_df = df
            self.update_progress(90, "Post-correction validation...")

            # Post-correction KPI dashboard
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    post_report = run_quality_check(
                        df, self.granularity_label, self.hours_per_interval, silent=True)
                if post_report:
                    cons_col = self.get_selected_column_index(self.cons_col_var.get())
                    prod_col = self.get_selected_column_index(self.prod_col_var.get())
                    post_kpi = run_validation(
                        df=df, quality_report=post_report,
                        original_df=self.df, pre_correction_df=pre_correction_df,
                        consumption_col=cons_col, production_col=prod_col,
                    )
                    self.display_kpi_dashboard(post_kpi)
            except Exception:
                pass  # Non-critical: don't block corrections on KPI failure

            self.update_progress(100, "Corrections applied!")

            self.show_result("Corrections applied successfully!\n"
                           f"  - Final rows: {len(df)}\n"
                           "Ready to export.")

        except Exception as e:
            import traceback
            self.update_progress(0, "Error applying corrections")
            messagebox.showerror("Error", f"Failed to apply corrections:\n{str(e)}\n\n{traceback.format_exc()}")

    def export_data(self):
        """Export transformed data to Excel."""
        if self.transformed_df is None:
            messagebox.showerror("Error", "No data to export.")
            return

        # Ask for save location
        site_name = self.site_name_var.get().strip()
        if site_name:
            default_name = build_output_filename(site_name, "EnergyAnalysis", "xlsx")
        else:
            default_name = os.path.splitext(os.path.basename(self.file_path))[0] + "_parsed.xlsx"
        default_dir = os.path.dirname(self.file_path)

        save_path = filedialog.asksaveasfilename(
            initialdir=default_dir,
            initialfile=default_name,
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")]
        )

        if not save_path:
            return

        self.update_progress(50, "Exporting to Excel...")

        try:
            # Suppress rich console output to avoid encoding issues
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                save_xlsx(self.transformed_df, save_path, is_final=True)
            self.update_progress(100, "Export complete!")

            self.show_result(f"Data exported successfully!\n"
                           f"  File: {save_path}\n"
                           f"  Rows: {len(self.transformed_df)}")

            # Ask if user wants to open the file
            if messagebox.askyesno("Export Complete",
                                   f"File saved to:\n{save_path}\n\nOpen the file?"):
                os.startfile(save_path)

        except Exception as e:
            self.update_progress(0, "Error exporting")
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")

    def open_cli(self):
        """Open command line with the CLI tool."""
        project_dir = os.path.dirname(os.path.abspath(__file__))

        # Windows: open cmd and run the CLI
        if sys.platform == "win32":
            cmd = f'start cmd /K "cd /d {project_dir} && python run.py"'
            subprocess.Popen(cmd, shell=True)
        else:
            # macOS/Linux
            subprocess.Popen(["python", "run.py"], cwd=project_dir)

    def open_claude_code(self):
        """Launch Claude Code in the project directory."""
        project_dir = os.path.dirname(os.path.abspath(__file__))

        try:
            if sys.platform == "win32":
                # Try to launch claude code
                cmd = f'start cmd /K "cd /d {project_dir} && claude"'
                subprocess.Popen(cmd, shell=True)
            else:
                subprocess.Popen(["claude"], cwd=project_dir)
        except FileNotFoundError:
            messagebox.showinfo("Claude Code",
                              "Claude Code CLI not found.\n\n"
                              "Install it from: https://claude.ai/download")

    def update_progress(self, value, text):
        """Update progress bar and label."""
        self.progress_var.set(value)
        self.progress_label.config(text=text)
        self.root.update_idletasks()

    def show_result(self, text):
        """Show a result message in the results area."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state=tk.DISABLED)


class CorrectionDialog(tk.Toplevel):
    """Dialog for selecting correction options."""

    def __init__(self, parent, report):
        super().__init__(parent)
        self.title("Apply Corrections")
        self.configure(bg=COLORS["white"])
        self.transient(parent)
        self.grab_set()

        # Fixed size window
        self.geometry("600x500")
        self.resizable(False, False)

        self.report = report
        self.result = None

        # Variables
        self.dup_var = tk.StringVar(value="first")
        self.gap_var = tk.StringVar(value="previous_day")
        self.missing_var = tk.StringVar(value="previous_day")
        self.outlier_var = tk.StringVar(value="keep")

        self.create_widgets()

        # Center the dialog
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 600) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 500) // 2
        self.geometry(f"600x500+{x}+{y}")

    def create_widgets(self):
        # Main container with proper layout
        main_frame = tk.Frame(self, bg=COLORS["white"])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title at top (fixed height)
        title_frame = tk.Frame(main_frame, bg=COLORS["white"], height=50)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        title = tk.Label(title_frame, text="Correction Options",
                        font=("Segoe UI", 14, "bold"),
                        bg=COLORS["white"], fg=COLORS["primary"])
        title.pack(pady=12)

        # Button frame at bottom (fixed height) - CREATE THIS FIRST
        btn_frame = tk.Frame(main_frame, bg=COLORS["white"], height=70)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)
        btn_frame.pack_propagate(False)

        # Separator line above buttons
        separator = tk.Frame(btn_frame, bg=COLORS["light_gray"], height=1)
        separator.pack(fill=tk.X)

        # Button container
        btn_container = tk.Frame(btn_frame, bg=COLORS["white"])
        btn_container.pack(expand=True, fill=tk.BOTH, pady=15, padx=20)

        apply_btn = ModernButton(btn_container, "Apply Corrections",
                                command=self.apply,
                                bg=COLORS["secondary"], width=160, height=40)
        apply_btn.pack(side=tk.RIGHT, padx=5)

        cancel_btn = ModernButton(btn_container, "Cancel",
                                 command=self.destroy,
                                 bg=COLORS["light_gray"], width=100, height=40)
        cancel_btn.pack(side=tk.RIGHT, padx=5)

        # Scrollable content area (takes remaining space)
        content_frame = tk.Frame(main_frame, bg=COLORS["white"])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 5))

        # Canvas with scrollbar
        canvas = tk.Canvas(content_frame, bg=COLORS["white"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=COLORS["white"])

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=540)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind mousewheel to canvas
        canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        scroll_frame.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        # Option groups
        duplicates = len(self.report.get("duplicates", []))
        self.create_option_group(scroll_frame, "Duplicate Timestamps",
                                f"{duplicates} found",
                                self.dup_var,
                                [("Keep first occurrence", "first"),
                                 ("Keep last occurrence", "last"),
                                 ("Average values", "average"),
                                 ("Leave as-is", "skip")])

        missing_ts = self.report.get("missing_timestamps", 0)
        self.create_option_group(scroll_frame, "Time Gaps",
                                f"{missing_ts} missing timestamps",
                                self.gap_var,
                                [("Fill from previous day", "previous_day"),
                                 ("Interpolate", "interpolate"),
                                 ("Fill with zero", "zero"),
                                 ("Skip", "skip")])

        missing_vals = sum(v["count"] for v in self.report.get("missing_values", {}).values())
        self.create_option_group(scroll_frame, "Missing Values",
                                f"{missing_vals} found",
                                self.missing_var,
                                [("Fill from previous day", "previous_day"),
                                 ("Interpolate", "interpolate"),
                                 ("Fill with zero", "zero"),
                                 ("Leave as-is", "skip")])

        outliers = len(self.report.get("outliers", []))
        self.create_option_group(scroll_frame, "Outliers",
                                f"{outliers} detected",
                                self.outlier_var,
                                [("Replace from previous day", "previous_day"),
                                 ("Cap at threshold", "cap"),
                                 ("Keep as-is (recommended)", "keep")])

    def create_option_group(self, parent, title, info, variable, options):
        """Create a group of radio options."""
        group = tk.LabelFrame(parent, text=title,
                             font=("Segoe UI", 10, "bold"),
                             bg=COLORS["white"], fg=COLORS["primary"],
                             padx=10, pady=5)
        group.pack(fill=tk.X, pady=10, padx=5)

        info_label = tk.Label(group, text=info,
                             font=("Segoe UI", 9),
                             bg=COLORS["white"], fg=COLORS["light_gray"])
        info_label.pack(anchor=tk.W)

        for text, value in options:
            rb = tk.Radiobutton(group, text=text, variable=variable, value=value,
                               font=("Segoe UI", 9),
                               bg=COLORS["white"], fg=COLORS["text_dark"],
                               activebackground=COLORS["white"],
                               selectcolor=COLORS["white"])
            rb.pack(anchor=tk.W, pady=2)

    def apply(self):
        """Apply button handler."""
        self.result = {
            "duplicates": self.dup_var.get(),
            "gaps": self.gap_var.get(),
            "missing": self.missing_var.get(),
            "outliers": self.outlier_var.get(),
        }
        self.destroy()


class GranularityDialog(tk.Toplevel):
    """Dialog for selecting data granularity when auto-detection fails."""

    def __init__(self, parent, detection_result: dict | None):
        super().__init__(parent)
        self.title("Select Data Granularity")
        self.configure(bg=COLORS["white"])
        self.transient(parent)
        self.grab_set()

        # Fixed size window
        self.geometry("450x350")
        self.resizable(False, False)

        self.detection_result = detection_result
        self.result = None
        self.granularity_var = tk.StringVar(value="15")  # Default to 15 minutes

        self.create_widgets()

        # Center the dialog
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - 450) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 350) // 2
        self.geometry(f"450x350+{x}+{y}")

    def create_widgets(self):
        # Main container
        main_frame = tk.Frame(self, bg=COLORS["white"], padx=30, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title = tk.Label(main_frame, text="Data Granularity",
                        font=("Segoe UI", 16, "bold"),
                        bg=COLORS["white"], fg=COLORS["primary"])
        title.pack(pady=(0, 15))

        # Detection info
        if self.detection_result:
            info_frame = tk.Frame(main_frame, bg=COLORS["bg"], padx=15, pady=10)
            info_frame.pack(fill=tk.X, pady=(0, 15))

            tk.Label(info_frame, text="Auto-detection results:",
                    font=("Segoe UI", 10, "bold"),
                    bg=COLORS["bg"], fg=COLORS["text_dark"]).pack(anchor=tk.W)

            detected = self.detection_result["label"]
            confidence = self.detection_result["confidence"]
            consistency = self.detection_result["consistency"]

            tk.Label(info_frame, text=f"Detected: {detected}",
                    font=("Segoe UI", 9),
                    bg=COLORS["bg"], fg=COLORS["text_dark"]).pack(anchor=tk.W)
            tk.Label(info_frame, text=f"Confidence: {confidence:.0%}",
                    font=("Segoe UI", 9),
                    bg=COLORS["bg"], fg=COLORS["warning"] if confidence < 0.7 else COLORS["success"]).pack(anchor=tk.W)
            tk.Label(info_frame, text=f"Consistency: {consistency:.0%}",
                    font=("Segoe UI", 9),
                    bg=COLORS["bg"], fg=COLORS["text_dark"]).pack(anchor=tk.W)

            tk.Label(info_frame, text="Confidence is too low - please confirm or select manually:",
                    font=("Segoe UI", 9, "italic"),
                    bg=COLORS["bg"], fg=COLORS["secondary"]).pack(anchor=tk.W, pady=(5, 0))
        else:
            tk.Label(main_frame, text="Could not auto-detect granularity.\nPlease select the time interval between data points:",
                    font=("Segoe UI", 10),
                    bg=COLORS["white"], fg=COLORS["text_dark"],
                    justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 15))

        # Radio buttons for granularity selection
        options_frame = tk.Frame(main_frame, bg=COLORS["white"])
        options_frame.pack(fill=tk.X, pady=10)

        for label, minutes, hours in STANDARD_GRANULARITIES:
            rb = tk.Radiobutton(options_frame, text=label,
                               variable=self.granularity_var, value=str(minutes),
                               font=("Segoe UI", 11),
                               bg=COLORS["white"], fg=COLORS["text_dark"],
                               activebackground=COLORS["white"],
                               selectcolor=COLORS["white"])
            rb.pack(anchor=tk.W, pady=5)

        # Buttons at bottom
        btn_frame = tk.Frame(main_frame, bg=COLORS["white"])
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(20, 0))

        confirm_btn = ModernButton(btn_frame, "Confirm",
                                   command=self.confirm,
                                   bg=COLORS["secondary"], width=120, height=38)
        confirm_btn.pack(side=tk.RIGHT, padx=5)

        # If we have detection result, offer to use it
        if self.detection_result:
            use_detected_btn = ModernButton(btn_frame, f"Use {self.detection_result['label']}",
                                            command=self.use_detected,
                                            bg=COLORS["light_gray"], width=140, height=38)
            use_detected_btn.pack(side=tk.RIGHT, padx=5)

    def confirm(self):
        """Confirm button handler."""
        minutes = int(self.granularity_var.get())
        for label, mins, hours in STANDARD_GRANULARITIES:
            if mins == minutes:
                self.result = {"label": label, "hours_per_interval": hours, "minutes": minutes}
                break
        self.destroy()

    def use_detected(self):
        """Use the auto-detected value."""
        if self.detection_result:
            self.result = {
                "label": self.detection_result["label"],
                "hours_per_interval": self.detection_result["hours_per_interval"],
                "minutes": self.detection_result["minutes"],
            }
        self.destroy()


def main():
    """Main entry point."""
    root = tk.Tk()

    # High DPI support for Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    app = EnergyParserGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
