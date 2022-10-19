.. _concepts-label:

NeuroDevSim Concepts
====================

Overview
--------
NeuroDevSim stands for Neural Development Simulator: a parallel computational framework to simulate the development of neurons in tissue. It supports the following functionality:

- **Growth of 3D neuronal morphologies**: neurons grow starting from a soma that can sprout dendrites and axons. Growth is embodied in *fronts* which mimic the functionality of growth cones: they can elongate, branch, terminate or retract. Fronts can be spheres or cylinders.
- **Cell migration**: somata can migrate before sprouting dendrites. They can make a trailing axon during this migration.
- **Microcircuits**: neurons are generated together in a simulated volume. With the addition of connections rules and synapses circuits emerge.
- **Interactions**: growth can be influenced by environmental cues. Most obvious is guidance through repulsion or attraction by other neurons or chemical cues. Existing structures can block growth: physical overlap between structures is not allowed.

This software is described in the following preprint, please cite it when using NeuroDevSim:

E. De Schutter: Efficient simulation of neural development using shared memory parallelization. https://www.biorxiv.org/content/10.1101/2022.10.17.512465v1

Concepts and software design
----------------------------

- it is a **phenomenological** simulator, in general physics (diffusion, forces,...) is not simulated. Growth is simulated purely phenomenologically as extension by cylindrical fronts, partially driven by random numbers. Environmental interactions can also be purely phenomenological, but more biologically realistic interactions like stochastic detection of remote chemical cues are also supported.
- growth runs in **cycles**. During each cycle all growing fronts perform an action. The unit of time is not specified, in practice a single cycle can correspond to anything in a range from tens of minutes up to many hours or days.
- NeuroDevSim runs in **parallel** using shared memory to allow fast simulation of the growth of many neurons.
- fronts act as agents in :ref:`agent-label` and :ref:`tree-label`.
- use of the Python `Errors and Exceptions framework <https://docs.python.org/3/tutorial/errors.html>`_ to handle collisions and other exceptions.

.. _fronts-label:

Fronts
------
In NeuroDevSim fronts are phenomenological implementations resembling biological growth cones. An active front is a front that is still developing. An inactive front becomes a continuation point, branching point or a terminal tip. Exceptionally an inactive front can still undergo interstitial branching. Neurites are represented by connected cylindrical fronts.  

Fronts have a dual identity. On the one hand they are physical structures with a location and radius in space. On the other hand, a front is a cellular agent that contains its own growth rules describing how and when it should extend, branch or terminate. As explained in more detail in :ref:`agent-label`, when an active front is not terminating, it produces one or more new fronts. The old front becomes inactive and the newly formed front(s) become(s) active fronts.  

The location of the new front is computed in accordance to a front’s construction rules and locally available information. Information can be everything that is known to NeuroDevSim. For instance, homotypic cues can be used, or, the transient laminar information through which a front is traveling. These cues have a direct biophysical interpretation, but also more phenomenological cues such as directional information related to a boundary can be used. Construction rules define how the front interacts with other structures in the simulation volume depending on internal variables: no interaction, repulsion or attraction. Hence, the context is used as a guidance cue. The influence of these cues can be distance-dependent mimicking gradients of secreted molecules. In addition, fronts can modify the environment by secreting substrate: phenomenological representations of secretion molecules that can in turn be used as a guidance cue.

.. image:: front_extension.png
    :align: center

As illustrated in the figure, NeuroDevSim uses vector addition to combine all the construction rules and compute the location of a new front. In this example 3 influences are combined to decide on the location of the new front: there is a tendency to continue along the current direction (black arrow pointing right), repulsion by a front from another neuron (light gray arrow pointing left) and chemical attraction by the gradient towards the top (dark gray arrow pointing upward). All of these arrows are vectors with different lenghts and the new front will be located along the direction determined by the vector sum at a given distance from the front being extended.

Finally, a new front can never intersect with existing structures. Front collisions are handled automatically but can also be model specific. 
 
In practice, a NeuroDevSim model consists of objects derived from ``Front``. The native ``Front`` object has no growth functions implemented, these need to be specified in the definition of a derived front as explained in :ref:`started-label`.

.. _agent-label:

Agent-based modeling
--------------------
NeuroDevSim implements `agent-based modeling <https://en.wikipedia.org/wiki/Agent-based_model>`_, with fronts as agents. The first front of a neuron is its soma, which will become the root of a branching tree. The soma is usally created before the simulation starts, as explained in :ref:`started-label`.

Each ``Front``, including the soma, will call a method to grow called ``manage_front``. As the simulation proceeds from cycle to cycle, each growing front will call its ``manage_front`` method independently from all other ones. ``manage_front`` is  usually called only once, afterwards the front is made inactive. Inside the ``manage_front`` method several specialized methods can be called, the most commonly used one is ``add_child`` which grows an extension of the front. This process is illustrated in the figure below, which shows the status of a serial simulation during four consecutive cycles at the time when the growing front calls ``manage_front``. 

Notice that in a parallel simulation the "waiting" front in the figure may be calling ``manage_front`` at the same time on another process.

.. image:: front_agents.png

As the simulation progresses from cycle to cycle the currently growing fronts become inactive and the fronts that were newly created in the previous cycle will grow in the next cycle. Notice that usually there will be many growing fronts each calling ``manage_front`` during the same cycle; in the figure only in cycle 4.

Admin process
-------------
In addition to computing processes running in parallel, NeuroDevSim also requires a central administrator called **Admin**. This administrator performs all internal housekeeping. It is the first object created during a NeuroDevSim simulation by instantiating ``Admin_agent``. It schedules execution of ``manage_front`` calls so that all processes are kept maximally busy. Processes communicate any updates to fronts and new fronts created to the administrator, which outputs to a centralized database file containing all neuronal morphologies.  

The administrator maintains a central clock, called the **cycle**, to synchronize updating of fronts. This clock ensures that irrelevant issues such as execution time on the computing hardware do not bias simulated growth. Technically, this means that during each cycle every active front performs exactly one call of its ``manage_front`` method.

In addition to running simulations, the administrator can also be run in :ref:`interactive-label`, mainly to debug model scripts.

.. _tree-label:

Neurons as trees
----------------
Neurons in NeuroDevSim consist of fronts and these fronts are organized as `trees <https://en.wikipedia.org/wiki/Tree_structure>`_. The root element of the tree is usually a spherical soma, which is created by the ``add_neurons`` method, and each neuron is a separate tree. In the figure above, the tree hierarchy runs from left to right.

The relationship between the fronts in a tree is described using the kinship terminology of family relations:

- the **parent** is higher in the hierarchy and was created before its children. It creates children using the ``add_child`` method, either one child during an extension event or multiple children during a branching event (see figure).
- a **child** is lower in the hierarchy and was created by its direct parent. It has siblings if it was created during a branching extension.
- fronts can have any number of children, but for neurons it is commonly assumed that only the soma can have many branches while dendrites and axons branch with only two children.

Growth-rate
-----------
There is no explicit growth rate parameter in NeuroDevSim. It is set implicitly by:

- the number of cycles in a simulation: this divides the developmental time for neuronal growth into fixed length time segments.
- the mean extension length of fronts during each cycle. 

The growth rate is therefore the ``mean front extension length / real time corresponding to 1 cycle``.



