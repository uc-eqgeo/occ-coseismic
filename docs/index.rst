.. occ-coseismic documentation master file, created by
   sphinx-quickstart on Tue Aug 20 13:25:16 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Our Changing Coast: Probabilistic cOseismic diSplacemenT hAazard modeL (POSTAL)
===================================

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.

The Probailistic cOseismic diSplacemenT hAazard modeL (POSTAL) can provided hazard estiamtes for vertical land motion due to coseismic rupture.
POSTAL is an expansion of the proof of concept described in Delano et al. (2025), but optimised to work at national scales.
It is developed as part of the Te Ao Horihuri: Te Ao Hou | Our Changing Coast project to assess the hazard faced by coastal communites in New Zealand (Coastal POSTAL). 
Initially designed for use with the outputs of the New Zealand National Seismic Hazard Model, it can be used with any fault rupture sets provided in an OpenSHA format.


Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Preprocessing
      Selecting Sites
      Discretising Faults
         Crustal
         Subduction
   Running POSTAL
   Postprocessing
      Visualising Results
      Interpreting Results