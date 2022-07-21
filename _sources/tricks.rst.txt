Efficient tricks
================

This section contains some tricks that did not fit in other parts of the documentation.

Fronts belonging to same neuron
-------------------------------
The fastest way to check whether two fronts belong to the same neuron is to use the ``get_soma`` method on both and compare the results::

    def manage_front(self,constellation):
        ...
        soma1 = front1.get_soma(constellation)
        soma2 = front2.get_soma(constellation)
        if soma1 == soma2: # same neuron
            ...
        else: # different neuron
            ...

The soma ``ID``, call ``get_soma(constellation,returnID=True)``, is the best attribute to store if one needs to store information about another neuron.

To check neuron identity the ``get_neuron`` method should be used, to check neuron type use ``get_neuron_type``.

Has a front grown?
------------------
There is no status flag similar to ``Front.has_moved()`` or ``Front.has_migrated()`` to check for growth because this can be achieved using the *num_children* attribute::

    def manage_front(self,constellation):
        ...
        if self.num_children > 0: # self has grown
            ...
        else: # self has not grown
            ...
            
*num_children* is always up to date and can be accessed much faster than the result of ``Front.get_children(constellation)``.

