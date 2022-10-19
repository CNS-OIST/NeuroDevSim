.. _usefulerrors-label:

Useful ``Errors``
=================
A complete listing of NeuroDevSim errors can be found in :ref:`simulator-label`. Here errors that can easily be used to improve model code are described briefly.

.. _collisionerror-label:

``CollisionError``
------------------
This is the most important error to catch because it is quite difficult to prevent collisions between a new front being made by, for example, ``add_child`` and existing fronts. If a ``CollisionError`` occurs a different *coordinate* should be tried for the method that triggered it. In most of the examples this is done randomly, but NeuroDevSim provides a ``Front.solve_collision`` method that can also help. This is explained in detail in :ref:`collisions-label`. Examples can also be found in all notebooks.

Often it is useful to know which existing front caused the collision. This information is available in ``CollisionError``::

    def manage_front(self,constellation):
    ...
    try:
        new_front = self.add_child(constellation,new_pos)
        ...
    except CollisionError as error:
        print (self,"collides with",error.collider,"with distance",error.distance)
        
where *error.collider* is the offending front. Note that the standard behavior of all methods is to return a ``CollisionError`` upon the first collision detected. This behavior can be changed to detect all colliding fronts, see :ref:`collisions-label`. 

.. _gridcompetitionerror-label:

``GridCompetitionError``
------------------------
This error is unique to the shared memory parallel computing implemented in NeuroDevSim. An important coding challenge is to prevent two different processes from trying to write to the same memory location at the same time and to prevent reading partial information because another process is writing. This prevention is done through transient locking of specific memory locations. In the context of ``GridCompetitionError`` locking of a *grid* location failed. The *grid* contains information about the location of all existing fronts, stored on a Cartesian 3D grid. It is used to detect collisions and needs to be accessed and updated frequently. A ``GridCompetitionError`` occurs when two or more processes try to access the same grid coordinate simultaneously. Because this happens frequently, the standard approach is for all processes except one to wait till the allowed one completes its task, but sometimes the competition is so fierce that this leads to excessive waiting times and then a ``GridCompetitionError`` occurs.

One way in which NeuroDevSim tries to avoid this problem is by scheduling the order in which fronts call ``manage_front`` so that growing or migrating fronts occupying the same grid coordinate are not processed simultaneously on different processes. To do this efficiently it is important that the ``is_growing()`` and ``is_migrating()`` :ref:`flags-label` are set correctly. For simple simulations this is done automatically by ``self.disable(constellation)`` when growth or migration is finished. But in more complex models these :ref:`flags-label` may have to be set explicitly.

If a ``GridCompetitionError`` occurs the best strategy is often to try calling the method again a few times with the same or different parameters, this approach is taken in many examples. Alternatively, ``manage_front`` can return without disabling *self* so that the same call is made again next cycle, but this may reduce the growth rate. Note that it is not wise to loop many times (more than 10) for a ``GridCompetitionError`` because this may significantly slow down the simulation.

.. _insideparenterror-label:

``InsideParentError``
---------------------
The *coordinate* provided to ``self.add_child`` or similar method is inside the volume occupied by the future parent *self*. The obvious solution is to provide another value for *coordinate*.

.. _volumeerror-label:

``VolumeError``
---------------
The *coordinate* provided to ``self.add_child`` or similar method is outside the simulation volume. Because all growth and migration methods test for this condition anyway, it is more efficient to let the error happen and then deal with it instead of preventing it. In most cases growth should stop after this front is made with its *end* on the border. 

Because **no** collision detection occurs outside of the simulation volume, it is impossible to grow fronts outside of the volume.

An example can be found in 'Demo_attraction' in the :ref:`environmentnote-label`.
