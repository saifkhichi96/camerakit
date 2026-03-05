import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "CameraKit"
author = "Saif Khan"
copyright = "2026, Saif Khan"
release = "2.0.0"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "CameraKit Documentation"
html_static_path = ["_static"]

html_theme_options = {
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#1F6F8B",
        "color-brand-content": "#123C69",
    },
    "dark_css_variables": {
        "color-brand-primary": "#80BCE5",
        "color-brand-content": "#74AEE0",
    },
}

autodoc_mock_imports = [
    "cv2",
    "numpy",
    "toml",
    "easydict",
    "matplotlib",
    "mpl_interactions",
    "PIL",
    "lxml",
]
