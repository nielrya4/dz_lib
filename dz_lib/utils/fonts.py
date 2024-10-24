import os
from matplotlib import font_manager as fm

def get_sys_fonts():
    return fm.get_fontconfig_fonts()

def get_font(font_path):
    if not os.path.isfile(font_path):
        raise ValueError(f'Font file {font_path} does not exist.')
    font = fm.FontProperties(fname=font_path)
    return font

def get_default_font():
    default_font_properties = fm.FontProperties()
    return default_font_properties
