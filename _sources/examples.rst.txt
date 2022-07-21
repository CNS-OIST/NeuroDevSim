.. _examplesnote-label:

Examples
========
The *Examples* directory contains `jupyter notebooks <http://jupyter.org/>`_ and a few python scripts demonstrating NeuroDevSim features. The notebooks save their databases in the *output* directory. For beginning NeuroDevSim users it is best to run them in the order shown below.

Beginner's Examples
-------------------

.. _errorsnote-label:

Errors and Exceptions notebook
++++++++++++++++++++++++++++++

This notebook demonstrates the use of `Python Errors and Exceptions <https://docs.python.org/3/tutorial/errors.html>`_ for users who are not familiar with these Python concepts. It also shows how to recover from a crashed simulation in a notebook.  
  
.. _randomnote-label:

Random Growth notebook
++++++++++++++++++++++

The random branching model that is used in the :ref:`randomf-label` section of the Beginner's tutorial and a small network of such neurons.  

.. _interstitialnote-label:

Interstitial Growth notebook
++++++++++++++++++++++++++++

Demonstrates growth of oblique dendrites after the apical dendrite has extended. This is implemented as conditional growth in fronts that remain *growing*.

.. _environmentnote-label:

Environment notebook
++++++++++++++++++++

Demonstrates the use of  *get_fronts*, *CollisionError* or *Substrate* to interact with the environment during growth. 

Basic Examples
--------------

.. _realnote-label:

Real Morphologies notebook
++++++++++++++++++++++++++

Contains models of a spinal motoneuron and a cortical layer 5 pyramidal neuron.  

.. _retractnote-label:

Retraction notebook
+++++++++++++++++++

Demonstrates retraction of single fronts or entire branches.  

.. _synapsenote-label:

Synapses notebook
+++++++++++++++++

Several axons make synapses with a neuron. Includes also two models of **branch retraction** based either on number of synapses/branch or on summed synaptic activity in each branch and an illustration of the use of *color_scheme=3*.  

.. _migrationnote-label:

Migration notebook
++++++++++++++++++

Shows increasingly complex simulations of somatic migration, including filipod guided migration and migration with a trailing axon using *solve_collision* and *add_branch* to deal with **collisions**. Also includes an illustration of the use of *color_scheme=3*.

Advanced Examples
-----------------

.. _databasenote-label:

Database notebook
+++++++++++++++++

Storing additional attributes in the simulation database and retrieving or plotting the data.  

.. _importnote-label:

Import notebook
+++++++++++++++

Demonstrates the use of *import_simulation*.

.. _interactivenote-label:

Interactive Mode notebook
+++++++++++++++++++++++++

Demonstrates the use of the interactive mode and how to use it to debug a model with *import_simulation*.

Python scripts
--------------

These can only be run in a terminal using ``python filename.py``. All plots and movies are stored in the *plots and movies* directory.

random_growth.py
++++++++++++++++
Identical to *Random_model* example from the :ref:`randomnote-label` with pdf file and movie output.

random_network.py
+++++++++++++++++
Identical to *Small_network* example from the :ref:`randomnote-label` with pdf file and movie output.

plots_and_movies.py
+++++++++++++++++++
Illustrates the use of *nds_plot* to make pdf files and of *nds_movie* to make movies using the ``color_scheme=3`` option as described in :ref:`plots-label`.
