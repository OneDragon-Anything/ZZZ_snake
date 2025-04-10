from qfluentwidgets import setTheme as qf_setTheme, Theme

current_theme = Theme.LIGHT  # 默认主题

def setTheme(theme):
    global current_theme
    current_theme = theme
    qf_setTheme(theme)