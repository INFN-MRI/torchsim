Developer Guide
===============

How to set up to change TorchSim: what has to be on the machine, how to
install the source so your edits take effect, the style the code is written
in, how to run the tests and build these pages, and how a pull request is
opened.

.. contents::
   :local:
   :depth: 2


What you need on the machine
----------------------------

Beyond what the :doc:`user_guide` asks of everyone -- an isolated environment
and a PyTorch build chosen for your hardware -- working on the source needs:

**A C++17 compiler.** The CPU state machine and the PERK kernel are C++
extensions built from ``src/torchsim/_epg_cpu.cpp`` and
``src/torchsim/_perk_cpu.cpp``. An install from source compiles them, and
there is no Python fallback to fall back to.

.. tab-set::

   .. tab-item:: Linux
      :sync: linux

      .. code-block:: sh

         sudo apt install build-essential      # Debian, Ubuntu
         sudo dnf install gcc-c++ make         # Fedora, RHEL

   .. tab-item:: macOS
      :sync: mac

      .. code-block:: sh

         xcode-select --install

   .. tab-item:: Windows
      :sync: windows

      Install the **Visual Studio Build Tools** with the "Desktop development
      with C++" workload, then build from the *Developer Command Prompt* so
      the compiler is on the path.

**Git**, and a fork of https://github.com/INFN-MRI/torchsim if you intend to
open a pull request.

**An NVIDIA card, if you want to touch the GPU kernels.** They are Triton,
which arrives with the Linux CUDA wheels of PyTorch. You can develop and test
most of the Triton path without a card -- see :ref:`dev-tests` -- but only a
card runs it for real.


Installing for development
--------------------------

.. code-block:: sh

   git clone https://github.com/INFN-MRI/torchsim
   cd torchsim
   pip install -e ".[dev]"

The ``dev`` extra is the whole toolchain in one command: formatters, linters,
the test runner, the documentation build, and the phantom, coil,
reconstruction and NUFFT packages the gallery examples import while they
execute.

An editable install puts the Python sources on the path, so an edit under
``src/torchsim`` takes effect on the next import with nothing to rebuild. The
C++ extensions are different -- they are compiled artifacts:

.. code-block:: sh

   pip install -e ".[dev]"          # after editing any .cpp

**Check that the build actually succeeded.** A failed compile leaves the
previously built ``.so`` importable, so the suite runs green against a kernel
that no longer corresponds to the source you are reading. Read the exit status
of the install, not the last lines of its output, and confirm the extension is
the one you just built:

.. code-block:: sh

   pip install -e ".[dev]" ; echo "exit: $?"
   python -c "import torchsim._epg_cpu as k; print(k.__file__)"

Where the source lives
~~~~~~~~~~~~~~~~~~~~~~

``src/torchsim/sequence/``
    The description an acquisition is assembled from -- events, operators,
    builders -- and the dispatch that turns one into a kernel launch. The
    Triton kernels are ``_epg_triton.py``; the shared parameter ABI, which the
    Python dispatch, the C++ extension and the Triton kernels all read, is
    ``_parameters.py``.

``src/torchsim/model/``
    What a signal model is: the physics, the simulator that orders its events,
    and the binding that resolves a protocol's structure once and rebinds its
    values per call.

``src/torchsim/_epg_cpu.cpp``, ``src/torchsim/_perk_cpu.cpp``
    The CPU kernels. Every path the GPU has -- forward, forward-mode,
    adjoint, forward-over-reverse, the real-subspace specialization, the pool
    models -- exists here too, and the two agree to float32 round-off.

``src/torchsim/simulators/``, ``estimators/``, ``recon/``, ``optim/``
    The sequences that ship, and what is built on top of them.

``tests/``, ``examples/``, ``docs/``
    Mirrored by subpackage, executed by the gallery, and built by Sphinx
    respectively.

:doc:`explanations/implementation` is the tour of how those pieces fit
together; read it before changing any of them.


Style
-----

**Formatting is automated, so do not argue with it.** Run both before you
commit, and CI checks the first:

.. code-block:: sh

   black .
   isort .            # configured with the black profile
   ruff check src

Ruff runs pycodestyle, pyflakes, bugbear, quote, pydocstyle, pyupgrade and
annotation rules, with the **numpydoc** docstring convention. Public functions
and classes carry annotations; the settings are in ``pyproject.toml`` and are
the authority.

**A docstring carries what a caller needs to call it**: one line of what it
does, then Parameters, Returns, Raises. Types belong in the prose of the
docstring, where they can be qualified ("array-like, one per echo"), rather
than in the rendered signature -- the API pages are built with typehints off
for exactly that reason. The annotations stay in the source for editors and
for mypy.

**Write for someone reading the code as it is now.** This is the rule most
worth internalizing here, and it is enforced in review:

* Never write text whose subject is the history of the code. No "used to", no
  "previously", no "this replaces the old X", no naming a bug that has been
  fixed or the change that fixed it.
* Do not justify the present shape by contrast with a shape that is gone.
* Do not restate what the code plainly says.

A comment earns its place by explaining a non-obvious algorithm, or a choice a
reader would otherwise undo. Even then, prefer a well-named function, or a
test whose name states the invariant -- those cannot go stale silently, and a
stale comment actively misleads the next reader. When you are tempted to
explain *why not the other way*, write a test instead.

The same applies to these pages. The documentation describes what TorchSim
does and why that is right on its own terms; it is not a changelog.

**Units are public at the edges and internal underneath.** A caller writes
milliseconds, degrees and Hz; a description timestamps in microseconds and
carries radians. Convert at the boundary, and name the unit in the identifier
(``duration_s``, ``flip_rad``, ``t1_ms``) rather than in a comment beside it.


.. _dev-tests:

Running the tests
-----------------

.. code-block:: sh

   pytest tests/                     # everything, with coverage
   pytest tests/epg/                 # one area
   pytest tests/epg/test_shift.py -k inversion
   pytest tests/ -n auto             # xdist, across cores

Coverage is on by default through ``addopts``, and reports to the terminal and
to ``coverage.xml``.

``tests/`` mirrors the source: ``epg/`` pins the state machine operator by
operator against closed forms -- the shift, the RF rotation, relaxation,
diffusion, flow, spoiling, the two-pool and three-pool longitudinal steps --
while ``sequence/``, ``model/``, ``estimators/``, ``recon/`` and ``optim/``
cover the layers above.

Two things to know before you time a run:

**The ``interpreted`` marker is deselected by default.** Those tests run a
Triton kernel through Triton's CPU interpreter -- no GPU, no compile, and
about a minute each. That is how the GPU plumbing is verified on a machine
with no card:

.. code-block:: sh

   pytest tests/ -m interpreted
   TRITON_INTERPRET=1 python your_script.py     # the same trick, by hand

**Kernel compiles dominate a cold GPU run**, not the arithmetic. A suite that
takes minutes on a card is mostly Triton compiling one specialization per
feature combination it meets; the second run of the same suite is a different
number entirely. Run the whole suite at natural boundaries rather than after
every edit.

When you change physics, add the test that pins it against something outside
TorchSim: a closed form, a published figure, or an isochromat summation you
write in the test itself. The ``tests/epg`` files are written that way, and
each states its invariant in its module docstring.


Building the documentation
--------------------------

.. code-block:: sh

   bash scripts/build_docs.sh              # incremental
   bash scripts/build_docs.sh --clean      # re-execute every example
   PYTHON_BIN=~/envs/torchsim/bin/python bash scripts/build_docs.sh

The script checks that the interpreter it is given can import TorchSim and the
documentation extensions, then builds into ``docs/build/html``. Two things
happen during the build that make it slower than a plain Sphinx run and are
the point of it: **sphinx-gallery executes every example**, and **the figures
of the explanation pages are re-rendered** by
``docs/explanation_figures.py``, which simulates them with the TorchSim in
your working tree. A figure on those pages is therefore never older than the
code it illustrates. Read them:

.. code-block:: sh

   python -m http.server --directory docs/build/html 8000

and open ``localhost:8000``.

To add a figure to an explanation page, write a function in
``docs/explanation_figures.py`` that returns a Matplotlib figure, register it
in the ``FIGURES`` mapping at the bottom, and reference the file it writes with a
``figure`` directive. To add a gallery example, drop a script into the right
``examples/`` section -- the numeric prefix orders it, and the module
docstring becomes the page's introduction.


Opening a pull request
----------------------

1. **Branch from** ``main``, in your fork. One subject per branch; a
   refactor and a fix in the same diff cost the reviewer more than they save
   you.
2. **Make the change, with a test that fails without it.** For a physics
   change that test is against something outside TorchSim, as above.
3. **Run what CI runs**, so you find out here rather than there:

   .. code-block:: sh

      black . --check --diff
      ruff check src
      pytest tests/
      bash scripts/build_docs.sh

4. **Open the pull request.** A template asks for what the change does, how it
   was checked -- the command and what it printed, since "tests pass" does not
   say which ones -- and the checklist above.
5. **CI runs** the style check, the test matrix on the supported Python
   versions, and the documentation build. All three have to be green.
6. **Review** is a conversation about the code, not about you; the same
   applies in the other direction when you review. Push follow-up commits
   rather than force-pushing over the discussion, so a reviewer can read what
   changed.

Everything here happens under the :doc:`misc/code_of_conduct`, which applies
to issues, discussions, pull requests and reviews alike. TorchSim is released
under the :doc:`misc/license`, and a contribution is released under the same
terms; :doc:`misc/contributors` is generated from the repository history.
