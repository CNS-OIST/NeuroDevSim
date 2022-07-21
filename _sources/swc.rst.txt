.. _swc-label:

SWC types used in NeuroDevSim
=============================

The ``Front.swc_type`` refers to entries in a column of the SWC file format, which was defined in `Cannon et al. 1998 <https://www.ncbi.nlm.nih.gov/pubmed/9821633>`_. NeuroDevSim uses standard values defined in this paper and on `NeuroMorpho.org <http://NeuroMorpho.org>`_, but also adds several custom defined SWC types:

===== ================= ==========================
type  structure         comment
===== ================= ==========================
0     undefined         defined in Cannon et al.
1     soma              defined in Cannon et al.
2     axon              defined in Cannon et al.
3     (basal) dendrite  defined in Cannon et al.
4     apical dendrite   defined in Cannon et al.
5     custom            defined in Cannon et al.
6     neurite           defined by NeuroMorpho.org
7     glial process     defined by NeuroMorpho.org
8     oblique dendrite  NeuroDevSim definition
9     tuft dendrite     NeuroDevSim definition
10    smooth dendrite   NeuroDevSim definition
11    spiny dendrite    NeuroDevSim definition
12    filipodium        NeuroDevSim definition
13    spine             NeuroDevSim definition
14    synaptic bouton   NeuroDevSim definition
15-19 reserved          NeuroDevSim definition
===== ================= ==========================

Note that some of these definitions are required for NeuroDevSim to work properly. Somata automatically get ``swc_type=1``, while some of :ref:`migration-label` depends on proper use of axon (2) and filipodium (12) swc_types.
