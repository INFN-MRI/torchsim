"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import dataclasses
import os
import sys

from sphinx_gallery.sorting import ExplicitOrder

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../.."))  # Source code dir relative to this file

# -- Project information -----------------------------------------------------

project = "torchsim"
copyright = "2024, TorchSim Contributors"
author = "TorchSim Contributors"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_copybutton",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    "sphinx_add_colab_link",
    "sphinx_exec_directive",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# generate autosummary even if no references
autosummary_generate = True
# autosummary_imported_members = True
autodoc_inherit_docstrings = True
autodoc_member_order = "bysource"
# Types belong in the docstring, where they are prose a reader can qualify
# ("array-like, one per echo") rather than a signature they have to decode.
# The annotations stay for editors and for mypy.
autodoc_typehints = "none"
# A dataclass default reads as ``triggers=Triggers(excitation=<function ...>)``
# in a class heading, which is neither the type nor anything a caller writes.
autodoc_class_signature = "separated"
# Render a default as the source wrote it, so a dataclass field shows
# ``Triggers()`` rather than the repr of what that call returned.
autodoc_preserve_defaults = True

napoleon_include_private_with_doc = False
napolon_numpy_docstring = True
napoleon_use_admonition_for_references = True


pygments_style = "sphinx"
highlight_language = "python"

# -- Options for Sphinx Gallery ----------------------------------------------

#: The gallery's sections, in the order a reader should meet them.
GALLERY_SECTIONS = [
    "../examples/01-framework",
    "../examples/02-parameter-inference",
    "../examples/03-sequence-optimization",
    "../examples/04-model-based-imaging",
    "../examples/05-misc",
]

sphinx_gallery_conf = {
    "doc_module": "torchsim",
    "backreferences_dir": "generated/gallery_backreferences",
    "reference_url": {"torchsim": None},
    "examples_dirs": ["../examples/"],
    "gallery_dirs": ["generated/autoexamples"],
    "filename_pattern": "/0",
    "ignore_pattern": r"(__init__|conftest|utils).py",
    "nested_sections": True,
    "subsection_order": ExplicitOrder(GALLERY_SECTIONS),
    "within_subsection_order": "FileNameSortKey",
    "binder": {
        "org": "infn-mri",
        "repo": "torchsim",
        "branch": "gh-pages",
        "binderhub_url": "https://mybinder.org",
        "dependencies": [
            "./binder/apt.txt",
            "./binder/environment.yml",
        ],
        "notebooks_dir": "examples",
        "use_jupyter_lab": True,
    },
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/INFN-MRI/torchsim",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "home_page_in_toc": True,
}

# html_logo = "_static/logos/mri-nufft.png"
# html_favicon = "_static/logos/mri-nufft-icon.png"
html_title = "TorchSim Documentation"


def _skip_undocumented_specials(app, what, name, obj, skip, options):
    """Leave out members that render as a heading with nothing under it.

    A dataclass's synthesized ``__init__`` has no source for
    :confval:`autodoc_preserve_defaults` to read, so its defaults come out as
    reprs -- a whole ``Triggers(excitation=<function Excitation>, ...)`` where
    a reader wants the word ``Triggers``. Every field is documented as an
    attribute, which is where its type is stated anyway. An enum's ``__new__``
    carries no docstring at all.
    """
    if name == "__new__":
        return True  # an enum's, which says nothing a reader wants
    if name != "__init__" or skip:
        return None
    owner = getattr(obj, "__qualname__", "").rsplit(".", 1)[0]
    defined_in = sys.modules.get(getattr(obj, "__module__", ""))
    holder = getattr(defined_in, owner, None)
    return True if holder is not None and dataclasses.is_dataclass(holder) else None


def _hide_ignored_code_from_the_page_only() -> None:
    """Keep the page free of the blocks an example hides, and nothing else.

    sphinx-gallery strips its ignore blocks once, before it writes either the
    page or the notebook, so a downloaded notebook is missing whatever the page
    hides and raises on the first cell that needed it. Stripping them as the
    page is written instead leaves the downloadable script and notebook whole,
    which is what the Binder and Colab links open.

    A cell that is hidden in full renders as nothing rather than as an empty
    ``code-block`` directive. Its *output* -- the figures it drew, what it
    printed -- is emitted separately and is kept either way.
    """
    from sphinx_gallery import gen_rst, py_source_parser

    strip = py_source_parser.remove_ignore_blocks

    def keep(code):
        strip(code)  # for its check that every flag has its partner
        return code

    py_source_parser.remove_ignore_blocks = keep

    original = gen_rst.codestr2rst

    def codestr2rst(code, *args, **kwargs):
        shown = strip(code)
        return original(shown, *args, **kwargs) if shown.strip() else ""

    gen_rst.codestr2rst = codestr2rst

    write_notebook = gen_rst.jupyter_notebook

    def jupyter_notebook(script_blocks, *args, **kwargs):
        """The notebook keeps the code, but not the flags that hid it."""
        return write_notebook(
            [block._replace(content=_unflagged(block.content)) for block in script_blocks],
            *args,
            **kwargs,
        )

    gen_rst.jupyter_notebook = jupyter_notebook


def _unflagged(content: str) -> str:
    """The block without the comment lines that mark a hidden region."""
    return "\n".join(
        line
        for line in content.splitlines()
        if line.strip()
        not in ("# sphinx_gallery_start_ignore", "# sphinx_gallery_end_ignore")
    )


def setup(app):
    """Drop sphinx-gallery's code-link pass when :mod:`dbm` is missing.

    That pass caches the URLs it resolves with :mod:`shelve`, and some Python
    distributions package ``dbm`` separately from the interpreter. The links
    it would add are the only thing lost.
    """
    _hide_ignored_code_from_the_page_only()
    app.connect("autodoc-skip-member", _skip_undocumented_specials)
    try:
        import dbm  # noqa: F401
    except ImportError:
        from sphinx_gallery.docs_resolv import embed_code_links

        for listener in list(app.events.listeners.get("build-finished", [])):
            if listener.handler is embed_code_links:
                app.disconnect(listener.id)
