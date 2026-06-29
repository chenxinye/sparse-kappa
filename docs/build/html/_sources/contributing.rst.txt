Contributing
============

Development setup
-----------------

.. code-block:: bash

   git clone https://github.com/chenxinye/sparse-kappa.git
   cd sparse-kappa
   pip install -e ".[dev]"

Testing
-------

Run the test suite from repository root:

.. code-block:: bash

   python -m pytest tests -q

Documentation
-------------

Build docs locally:

.. code-block:: bash

   cd docs
   pip install -r requirements.txt
   make html
