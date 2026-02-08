# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for Spartacus - ReVolta Energy Analysis Tool."""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

rich_hiddenimports = collect_submodules('rich')
rich_datas = collect_data_files('rich')

a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('Logo_new_white.png', '.'),
        ('Energy storage.jpg', '.'),
        ('favicon 1.ico', '.'),
        ('logo Navy.jpeg', '.'),
    ] + rich_datas,
    hiddenimports=[
        'energy_parser',
        'energy_parser.file_reader',
        'energy_parser.analyzer',
        'energy_parser.transformer',
        'energy_parser.quality_check',
        'energy_parser.corrector',
        'energy_parser.exporter',
        'energy_parser.data_validator',
        'energy_parser.cli',
        'energy_parser.statistics',
        'energy_parser.report_generator',
    ] + rich_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Spartacus',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    icon='favicon 1.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Spartacus',
)
