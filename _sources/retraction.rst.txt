.. _retraction-label:

Modeling neuron pruning
=======================

In development, pruning of neuronal structures can be as important as growth. This is supported by the :ref:`retract-label` and :ref:`retractbranch-label`. Either can be called from ``manage_front`` and will result in the retraction of one or more fronts at the end of the cycle, after all ``manage_front`` calls have completed. Data about the deleted fronts will still be present in the simulation database with their *dead* value set to the cycle when the retraction method was called. Examples can be found in the :ref:`retractnote-label`.

The simplest one to use is the :ref:`retractbranch-label`. It is called for one of the children of *self* and will remove that *child* and all its descendants::

    def manage_front(self,constellation):
        ...
        self.retract_branch(constellation,child)
        ...
    
*child* should be a child of *self*.

Whether such an approach is sufficiently realistic depends on the implicit duration of a cycle. If it is very long, like days, then complete retraction of a neuronal branch may be biologically feasible in this period. But if it is on the order of hours or less, this is no longer realistic. Then it may be better to delete fronts gradually over consecutive cycles, using the :ref:`retract-label`. This can only be called on *self*, with the condition that *self* has no children::

    def manage_front(self,constellation):
        ...
        self.retract(constellation) # remove self from the simulation
        return # do not try to do anything else with self
    
If the *parent* is also to be retracted and while is not currently active, it should be activated::

    def manage_front(self,constellation):
        ...
        self.enable_parent(constellation) # enable parent
        self.retract(constellation) # retract self
        return # do not do anything else with self
        
If instead, all to be retracted fronts are active anyway then  ``self.has_child_retracted()`` may be useful to detect that a child was retracted, see :ref:`readflags-label`. Alternatively, one can just check for ``self.num_children==0``.

The :ref:`retract-label` can also be used to remove a single front of a growing process that got stuck. But unless the *parent* is activated, growth will stop. To continue growth the *parent* of *self* should be enabled and set for growth, this can be done with a single method call::

    def manage_front(self,constellation):
        ...
        self.enable_parent(constellation,growing=True) # enable parent and flag for growth
        self.retract(constellation) # retract self
        return # do not do anything else with self
        ...






