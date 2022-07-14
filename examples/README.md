Examples  
========  

Most examples are jupyter notebooks, they save their databases in the *output* directory. They show concepts that are explained in the Tutorials in the **Documentation** and it is best to do them in the order shown below.

Beginner's Examples
-------------------
**Errors and Exceptions**: demonstrates the use of Errors and Exceptions for users who are not familiar with these Python concepts. It also shows how to recover from a crashed simulation in a notebook.  
  
**Random Growth**: the random branching model that is used in the Beginner's tutorial and a small network of such neurons.  

**Interstitial Growth**: demonstrates growth of oblique dendrites after the apical dendrite has extended. This is implemented as conditional growth in fronts that remain *growing*.  

**Environment**: use of  *get_fronts*, *CollisionError* or *Substrate* to interact with the environment during growth.  

Basic Examples
--------------
**Real Morphologies**: models of a spinal motoneuron and a cortical layer 5 pyramidal neuron.  

**Retraction**: retract single fronts or entire branches.  

**Synapses**: axons make synapses with a neuron. Includes also two models of **branch retraction** based either on number of synapses/branch or on summed synaptic activity in each branch and an illustration of the use of *color_scheme=3*.  

**Migration**: increasingly complex simulations of somatic migration, including filipod guided migration and migration with trailing axon.  Also includes an illustration of the use of *color_scheme=3*.  

Advanced Examples
-----------------
**Database**: storing additional attributes in the simulation database and retrieving or plotting the data.  

**Import**: demonstrates the use of *import_simulation*.

**Interactive Mode**: demonstrates the use of the interactive mode and how to use it to debug a model with *import_simulation*.  

Python scripts
--------------

These can only be run in a terminal using ``python filename.py``. All plots and movies are stored in the *plots and movies* directory.

**random_growth.py**: identical to *Random_model* example from the Random Growth notebook with pdf file and movie output.

**random_network.py**: indentical to *Small_network* example from the Random Growth notebook with pdf file and movie output.

**plots_and_movies.py**: illustrates the use of *nds_plot* to make pdf files and of *nds_movie* to make movies using the ``color_scheme=3`` option. 
