.. _install:

Installation
============
Link to `PyPi <https://pypi.org>`_ and provide link for source install.

Quick install (PyPi)
--------------------
Install ``S2WAV`` from PyPi with a single command

.. code-block:: bash

    pip install s2wav

Check that the package has installed by running 

.. code-block:: bash 

	pip list 

and locate ``S2WAV``.


Install from source (GitHub)
----------------------------

When installing from source we recommend working within an existing conda environment, or creating a fresh conda environment to avoid any dependency conflicts,

.. code-block:: bash

    conda create -n s2wav_env python>=3.8
    conda activate s2wav_env

Once within a fresh environment ``S2WAV`` may be installed by cloning the GitHub repository

.. code-block:: bash

    git clone https://github.com/astro-informatics/s2wav
    cd s2wav

and running the install script, within the root directory, with one command 

.. code-block:: bash

    bash build_s2wav.sh

To check the install has worked correctly run the unit tests with 

.. code-block:: bash

	pytest --black tests/ 

.. note:: For installing from source a conda environment is required by the installation bash script, which is recommended, due to a pandoc dependency.
