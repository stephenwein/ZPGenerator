Installation
------------

Requirements:

* Python 3.10 to 3.12
* QuTiP 5.2.3+

To use ZPGenerator, you can install directly from GitHub using:

.. code-block:: bash

   (venv) $ python -m pip install --upgrade pip
   (venv) $ python -m pip install git+https://github.com/stephenwein/ZPGenerator.git@main

Alternatively, if you are interested in contributing to the project, clone the repository and install it in editable mode:

.. code-block:: bash

   (venv) $ git clone https://github.com/stephenwein/ZPGenerator.git
   (venv) $ cd ZPGenerator
   (venv) $ python -m pip install --upgrade pip
   (venv) $ python -m pip install -e .

To build the documentation locally:

.. code-block:: bash

   (venv) $ python -m pip install -r docs/requirements.txt
   (venv) $ sphinx-build -b html docs/source docs/build/html
