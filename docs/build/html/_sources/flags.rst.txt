.. _flags-label:

``Front`` status flags
======================

``Front`` status flags are important in scheduling ``manage_front`` calls and controlling model behavior. Some flags report on front status and can only be read, others can be changed by the user and some can be freely used. All flag methods are documented in :ref:`simulator-label`. NeuroDevSim tries to keep the control of scheduling simple with the use of the ``enable(constellation)`` and ``disable(constellation)`` methods that directly changes status flags, but in some instances finer control is needed.

Status flags scheduling ``manage_front``
----------------------------------------

``manage_front`` is called only for ``Front`` that are *active*, but the order in which fronts are called depends on the *growing* and *migrating* status flags and, if either is set to True, on the position of ``Front`` in the simulation volume. Correct setting of these flags is important to reduce the likelyhood of :ref:`gridcompetitionerror-label`. The *active*, *growing* and *migrating* flags can be set by the user or changed through the ``enable`` and ``disable`` methods:

================== ======================================== ========================== ===============
Flag query         Set to True                              Set to False               Active if True
================== ======================================== ========================== ===============
``is_active()``    ``enable(constellation)``                ``disable(constellation)`` yes  
``is_growing()``   ``set_growing()``                        ``clear_growing()``        yes  
                   ``enable(constellation,growing=True)``   ``disable(constellation)``    
``is_migrating()`` ``set_migrating()``                      ``clear_migrating()``      yes  
                   ``enable(constellation,migrating=True)`` ``disable(constellation)``   
================== ======================================== ========================== ===============

Note that *active* changes to True if either *growing* or *migrating* are set to True, but the reverse does not apply. All new ``Front`` created by ``Admin_agent.add_neurons`` or ``Front.add_child`` are *active* and *growing*. For fronts created by ``Front.add_branch`` only the last of the series made is *active* and *growing* unless the optional parameter *enable_all* is set to True. Somata created by ``Admin_agent.add_neurons`` can also be *migrating* if the optional parameter is set to True.

.. warning:: ``enable`` and ``disable`` can be called for any front. Only *self* or *new_front* created by ``add_child`` or ``add_branch`` can call ``set_growing()``, ``clear_growing()``, ``set_migrating()`` and ``clear_migrating()``. Calling these methods on other fronts will change the status flags but may not affect their behavior.

As mentioned, using ``disable(constellation)`` to stop growth of *self* is usually sufficient to control scheduling, but sometimes more fine-grained control is required::

    def manage_front(self,constellation):
        ...
        try:
            new_front = self.add_child(constellation,new_pos) # make a new front and store it
            # make front stop growing but keep it active
            self.clear_growing()
            return # completed this call
        ...

In the code example above, *self* is kept *active* but is not expected to call ``add_child`` again. This could be useful if *self* should be able to react to some future condition as it will keep calling ``manage_front``. If for some reason it should grow again at a later cycle, it is safer to call *self.set_growing()* first and wait till the next cycle to call ``add_child``.

In the following code example a parent front is not only enabled again but set to *growing* after retraction of a child::

    def manage_front(self,constellation):
        ...
        parent = self.get_parent(constellation)
        parent.enable(constellation,growing=True)        
        self.retract(constellation) # retract self
        return # do not do anything else with self
        ...
        
The same effect can be obtained using ``self.enable_parent(constellation,growing=True)``.

Finally, a reminder: only keep fronts *active* if needed. Fronts calling ``manage_front`` without it executing any code before ``return`` can slow down simulations significantly. It is possible to ``disable`` fronts transiently using the optional parameters *till_cycle*, *till_cycle_g* or *till_cycle_m*::

    def manage_front(self,constellation):
        ...
        self.disable(constellation,till_cycle_g=100) 
        ...
    
this will disable *self* till cycle 100. On cycle 100 it becomes *active* again with *is_growing()* True.

.. _readflags-label:

Read-only status flags
----------------------

The following status flags are set by NeuroDevSim methods and inform on status of ``Front``:

======================= ============================================== ===========================================
Flag query              Set by method                                  Meaning of True
======================= ============================================== ===========================================
``is_cylinder()``       ``add_neurons``, ``add_child``, ``add_branch`` shape is cylindrical, False for a sphere
``has_moved()``         ``migrate_soma``                               soma migrated at previous or current
                                                                       cycle
``has_migrated()``      ``migrate_soma``                               soma migrated at some cycle
``is_retracted()``      ``retract``, ``retract_branch``                front has been retracted
``has_child_retracted`` ``retract``, ``retract_branch``                a child of this front was retracted at some
                                                                       cycle, is reset to False after new child
                                                                       is made
``is_arc()``            ``add_branch``                                 front is part of an arc made by
                                                                       ``arc_around``
======================= ============================================== ===========================================

These status flags can be read for any ``Front`` but notice that ``has_moved()`` and ``has_migrated()`` can change value during the present cycle and for fronts other than *self* the timing of this change cannot be predicted.

User availabe status flags
--------------------------

The following status flags can be used as boolean variables by the user instead of making a *c_bool* additional attribute:

================ ================= ===================
Flag query       Set to True       Set to False
================ ================= ===================
``is_status1()`` ``set_status1()`` ``clear_status1()``  
``is_status2()`` ``set_status2()`` ``clear_status2()``  
``is_status3()`` ``set_status3()`` ``clear_status3()``  
================ ================= ===================

These status flags can be read and set/cleared for any ``Front``. However, if setting or clearing on a front other than *self* **be sure that only one front can do this** during a given cycle to avoid competition. See :ref:`attributes-label` for more information.
