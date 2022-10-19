Changes compared previous versions
**********************************

The first used NeuroDevSim version was 0.8.8. Version 0.9.2 was the first release of the new user interface based on ``manage_front`` and Exceptions, and used new processor scheduling methods. First public version is 1.0.0, which is used as reference to track changes.

NeuroDevSim concepts are derived from the `NeuroMaC software <https://www.frontiersin.org/articles/10.3389/fnana.2014.00092/full>`_ originally developed by Benjamin Torben-Nielsen and extended by Erik De Schutter to version 0.6.2.

Version 1.0.1
=============

1. Changes to methods:

- ``nds_movie`` has *neurons* optional parameter.
- ``nds_plot`` and ``nds_movie`` *neuron_color* optional parameter renamed to *neuron_colors*.

2. New and removed methods:

- ``sim_memory``: *Admin_agent* method that prints memory use.
- removed ``sim_signature`` as it was not useful for this version.

3. Bug fixes:

- unknown local variable 'color_attrib' bug in ``nds_plot`` and ``nds_movie`` fixed.
