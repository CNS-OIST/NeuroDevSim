.. _simulator-label:

simulator module
================
This module contains everything needed to run a NeuroDevSim simulation.

The simulator will print output depending on the *verbose* level set during ``Admin_agent`` instantiation, this verbose level is also used by all processes. Each additional level prints more information:

- verbose=0 : no output is printed except for error messages (red color).
- verbose=1 : simulation start and end messages and start of each cycle messages are printed (blue color).
- verbose=2 : warning messages are printed (magenta color), default setting.
- verbose=3 : methods called by ``manage_front`` start info is printed (black color).
- verbose=4 : ``Admin_agent`` methods start info is printed (black color).
- verbose=5 : processes methods start info is printed (black color).
- verbose=6 : for all above methods also interim info and end messages are printed.
- verbose=7 : low level non-public methods messages are printed.

Some ``Front`` methods have a *printing* parameter which allows additional control over printing.

.. _admin_agent-label:

``Admin_agent`` class
*********************
``Admin_agent`` is the conductor of NeuroDevSim. Every NeuroDevSim simulation starts by instantiating an ``Admin_agent`` and calling its ``add_neurons`` method to set up the actual simulation. Its ``simulation_loop`` method runs the simulation. Finish cleanly by calling the ``destruction`` method.

Should be instantiated only once for each simulation. Can be instantiated repeatedly in the beginning of a ``for`` loop provided each instantiation is removed by ``destruction`` at the end of the loop.

.. autoclass:: simulator.Admin_agent
   :members:
   :undoc-members:
   :show-inheritance:

Model definition classes
************************

``Point`` class
---------------
A ``Point`` encodes a 3D coordinate or vector in µm. It is used in ``Front`` to specify the origin and end coordinates of cylinders. ``Point`` can be instantiated anytime.

As a vector it can be used to compute directions for growth::

    # direction, target and front.end are all Points
    direction = target - front.end
    # compute the distance to target
    distance = direction.length()
    # make direction a unit vector
    u_dir = direction.norm()

Simple arithmetic using ``Point`` is possible, the following operations are supported and produce a new ``Point``:

- ``Point`` + ``Point`` : sums all attributes one by one
- ``Point`` + list  (list should be size 3 vector)
- ``Point`` + *numpy.array*  (*numpy.array* should be size 3 vector)
- ``Point`` - ``Point`` : subtracts all attributes one by one
- ``Point`` - list  (list should be size 3 vector)
- ``Point`` - *numpy.array*  (*numpy.array* should be size 3 vector)
- ``Point`` * integer : multiplies all attributes with integer
- ``Point`` * float : multiplies all attributes with float
- ``Point`` / integer : divides all attributes by integer
- ``Point`` / float : divides all attributes by float

The order is important: integer * ``Point`` is not supported and will result in a 'TypeError: invalid type promotion' error. Simple comparisons are also possible, the following operations are supported and produce a boolean:

- ``Point`` == ``Point``
- ``Point`` != ``Point``


.. autoclass:: simulator.Point
   :members:
   :undoc-members:
   :show-inheritance:

.. _front-label:

``Front`` class
---------------
``Front`` implements the agents that mimic growth and migration in NeuroDevSim. However, the class cannot be used directly. Instead a NeuroDevSim model creates a subclass of ``Front`` that defines a ``manage_front`` method. ``Front`` cannot be instantiated by the user, instead they are created by ``Admin_agent.add_neurons``, ``Front.add_child`` or ``Front.add_branch`` methods.  

Only a subset of ``Front`` **attributes** can be read directly as listed below, the other ones can be obtained with attribute methods.

``Front`` **methods** are labeled and ordered in this documentation as:

- growth methods that control front growth.
- attribute methods that give access to additional attributes.
- tree methods that give access to the neuronal tree structure.
- behavior methods that report the current behavior of the front.
- search methods to return other objects around the front.
- point methods to return points close to the front.
- size methods that return size related information.


.. autoclass:: simulator.Front
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: simulator.average_fronts

.. autofunction:: simulator.swc_name

.. _synfront-label:

``SynFront`` subclass
---------------------
``SynFront`` is a subclass of ``Front`` that enables synaptic connections between fronts. Both pre- and postsynaptic neurons have to use the ``SynFront`` subclass.

.. autoclass:: simulator.SynFront
   :members:
   :undoc-members:
   :show-inheritance:

``Neuron`` class
----------------
``Neuron`` contains information about all fronts belonging to the same neuron.

.. autoclass:: simulator.Neuron
   :members:
   :undoc-members:
   :show-inheritance:

``Substrate`` class
-------------------
``Substrate`` are chemical cues in the environment identified by a *name*. In the simplest case this can just be an unknown amount of cue at specific location(s). In more complex cases a diffusion coefficient is defined and stochastic concentrations can be computed with the *diff_gradient_to* function for either release from a fixed amount or production at a constant rate.

**Initialization:**
``Substrate`` can be instantiated either in the main loop or in any ``Front`` method::

    # instantiate a substrate without diffusion
    orig = Point(10.,10.,10.)
    n_mol = 1
    sub = Substrate(name, orig, birth, n_mol)
    # instantiate a substrate with diffusion from an initial amount
    n_mol = int(1e10)
    sub = Substrate(name, orig, birth, n_mol, diff_c=10.)
    # instantiate a substrate produce at constant rate with diffusion
    sub = Substrate(name, orig, birth, 0, rate=10000, diff_c=10.)

**Initialization parameters:**

- **name** (*string: name of substrate, multiple substrate can have the same name.*)
- **orig** (``Point``: unique location of substrate.)
- **birth** (*integer: cycle when substrate was/is created, available as constellation.cycle in ``Front`` methods.*)
- **n_mol** (*integer: number of molecules, value is important only when diff_c is defined.*)

**Optional parameters:**

- **rate** (*float: production rate of substrate, value is important only when diff_c is defined: if larger than zero production rate based diffusion is computed.*)
- **diff_c** (*float: diffusion coefficient of substrate.*)


.. autoclass:: simulator.Substrate
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: simulator.diff_gradient_to

``Synapse`` class
-----------------
Defines attributes for synaptic connections, read-only. Use the ``SynFront`` methods to create and access synapses.

.. autoclass:: simulator.Synapse
   :members:
   :undoc-members:
   :show-inheritance:

Utility classes
***************

.. _constellation-label:

``Constellation`` class
-----------------------

.. autoclass:: simulator.Constellation
   :members:
   :undoc-members:
   :show-inheritance:

``ID`` class
------------

.. autoclass:: simulator.ID
   :members:
   :undoc-members:
   :show-inheritance:
   
``DataID`` class
------------

.. autoclass:: simulator.DataID
   :members:
   :undoc-members:
   :show-inheritance:
   
``nds_list`` class
------------------

.. autoclass:: simulator.nds_list
   :members:
   :undoc-members:
   :show-inheritance:

Utility methods
***************

.. autofunction:: simulator.nds_version

.. autofunction:: simulator.unit_sample_on_sphere

.. autofunction:: simulator.unit_sample_on_circle

.. autofunction:: simulator.unit_sample_on_cone

.. autofunction:: simulator.angle_two_dirs

.. autofunction:: simulator.dist3D_cyl_to_cyl

.. autofunction:: simulator.dist3D_point_to_cyl

Errors
******

.. autoclass:: simulator.ActiveChildError

.. autoclass:: simulator.BadChildError

.. autoclass:: simulator.BugError

.. autoclass:: simulator.CollisionError

.. autoclass:: simulator.GridCompetitionError

.. autoclass:: simulator.InsideParentError

.. autoclass:: simulator.LockError

.. autoclass:: simulator.NotSelfError

.. autoclass:: simulator.NotSomaError

.. autoclass:: simulator.OverflowError

.. autoclass:: simulator.SynapseError

.. autoclass:: simulator.TypeError

.. autoclass:: simulator.ValueError

.. autoclass:: simulator.VolumeError
