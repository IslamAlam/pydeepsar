#!/usr/bin/env python
#
# pydeepsar documentation build configuration file, created by
# sphinx-quickstart on Fri Jun  9 13:47:02 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#

import pydeepsar

# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = "1.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom ones.
extensions = [
    "sphinx.ext.napoleon",
    "nbsphinx",
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'numpydoc',
    'sphinx_design',
    'matplotlib.sphinxext.plot_directive',
    'myst_nb',
    'jupyterlite_sphinx',
    'autoapi.extension',
]

autoapi_dirs = ['../../pydeepsar']
autoapi_root =  'api'

nbsphinx_allow_errors = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "DeepSAR Python package"
copyright = "2024, Islam Mansour"
author = "Islam Mansour"

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = pydeepsar.__version__
# The full version, including alpha/beta/rc tags.
release = pydeepsar.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".ipynb_checkpoints", "*.ipynb", "api_/**", ]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


sphinx_gallery_conf = {
    "backreferences_dir": "api",
}
# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]


html_logo = "./images/logo.png"

html_sidebars = {
    "index": "search-button-field",
    "**": ["search-button-field", "sidebar-nav-bs"]
}

html_theme_options = {
    'logo_only': False,
    "github_url": "https://github.com/IslamAlam/pydeepsar",
    "header_links_before_dropdown": 6,
    "icon_links": [],
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_persistent": [],
    "show_version_warning_banner": True,
    "secondary_sidebar_items": ["page-toc"],

}


# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "pydeepsardoc"


# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ("letterpaper" or "a4paper").
    #
    # "papersize": "letterpaper",
    "papersize": "a4paper",

    # The font size ("10pt", "11pt" or "12pt").
    #
    # "pointsize": "10pt",

    # Additional stuff for the LaTeX preamble.
    #
    # "preamble": "",

    # Latex figure (float) alignment
    #
    # "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (master_doc, "pydeepsar.tex",
     "DeepSAR Python package Documentation",
     "Islam Mansour", "manual"),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, "pydeepsar",
     "DeepSAR Python package Documentation",
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, "pydeepsar",
     "DeepSAR Python package Documentation",
     author,
     "pydeepsar",
     "One line description of project.",
     "Miscellaneous"),
]

# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/devdocs', None),
    'neps': ('https://numpy.org/neps', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'asv': ('https://asv.readthedocs.io/en/stable/', None),
    'statsmodels': ('https://www.statsmodels.org/stable', None),
}


# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True
