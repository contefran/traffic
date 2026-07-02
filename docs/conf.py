"""Sphinx configuration for the Traffic Simulator documentation.

Builds an HTML API reference from the package docstrings with autodoc. The
package is not installed in the docs environment, so the repo root is added to
``sys.path``; numpy/matplotlib are mocked so the (matplotlib-only)
visualization module can be documented without pulling in the heavy stack.
"""

import os
import sys
from datetime import date

# Make the ``traffic_sim`` package importable without installing it.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information ------------------------------------------------------
project = "Traffic Simulator"
author = "Francesco Conte"
copyright = f"{date.today().year}, {author}"
release = "0.1.0"

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",       # pull docstrings from the package
    "sphinx.ext.autosummary",   # module/attribute summary tables
    "sphinx.ext.napoleon",      # tolerate Google/NumPy docstring sections
    "sphinx.ext.viewcode",      # link each object to its highlighted source
    "sphinx.ext.intersphinx",   # cross-link to the Python stdlib docs
]

# The docstrings reference numpy/matplotlib-backed modules; mock them so autodoc
# can import every module (the simulation core itself has no such dependency).
autodoc_mock_imports = ["numpy", "matplotlib"]
autodoc_member_order = "bysource"      # document members in source order
autodoc_typehints = "description"      # render type hints in the body, not signatures
add_module_names = False               # show ``TrafficSim`` not ``traffic_sim.simulation.TrafficSim``
python_use_unqualified_type_names = True
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output --------------------------------------------------------------
html_theme = "furo"
html_title = "Traffic Simulator"
