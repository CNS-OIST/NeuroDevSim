.. _dos_donts-label:

Important dos and don'ts
========================
NeuroDevSim is fast thanks to two Python libraries: *multiprocessing* and its *sharedctypes*. The first supports parallel processing on a multicore computer and the second allows sharing of memory between the cores. Unfortunately the use of these methods imposes rules on how a NeuroDevSim model is coded.

In general, fundamental model components of NeuroDevSim, ``Front`` and ``Substrate``, act more like C language structures than like Python objects. Specifically:

No instance attributes
----------------------

The size of a ``Front`` and its subclasses is predefined and cannot be changed, consequently it is not possible to declare new attributes for specific instances inside methods. For example the following code::

    class MyFront(Front):

        def manage_front(self, constellation):
            ...
            # do not do this
            self.foo = 25
            # but this is fine, though usable only within the scope of the current manage_front call
            foo = 25

will not result in a new attribute *foo* being stored in the ``Front``. This code will not generate an error, but any attempt to access *self.foo* in subsequent ``manage_front`` calls will generate a "'Front' object has no attribute 'foo'" error.

User-defined attributes
-----------------------

Additional attributes can be declared in the ``Front`` subclass definition but special syntax is required::

    class MyFront(Front):
        _fields_ = Front._fields_ + [('foo', c_int)]

        def manage_front(self, constellation):
            ...
            # now it is safe to do this
            self.foo = 25

Note that *foo* will be present in **all** instances of *MyFront*, as mentioned in the previous subsection it is not possible to have instance specific attributes. Only fixed size attributes can be declared, lists, dictionaries or strings are not possible. It is not advised to store other ``Front`` as an attribute, instead store its ``ID`` as the attribute.

Defining additional attributes is explained in more detail in :ref:`subclassing-label`.

Do not instantiate ``Front`` or ``Synapse``
-------------------------------------------

While it is possible to instantiate ``Front`` or ``Synapse``, those objects cannot interact with existing ``Front`` and they cannot be stored in the shared arrays, they will disappear after the ``manage_front`` call is completed. Use the ``Admin_agent.add_neurons``, ``Front.add_child`` or ``Front.add_branch`` methods to instantiate ``Front``. See :ref:`synapses-label` about the use of ``add_synapse``.

.. _attributes-label:

Changing attributes of ``Front`` or ``Substrate``
-------------------------------------------------

The public predefined attributes of ``Front`` and ``Substrate`` are read-only. Their value is set when a new front or substrate is created and should not be changed. Doing so has unpredictable consequences, most likely the change will be ignored but it may also crash the simulation.

User-defined attributes can be used freely for *self*. Changing such attributes for a target ``Front`` or ``Substrate`` other than *self* is risky but sometimes a useful short-cut. The challenge is to ensure that:

- there is no competition among different fronts that are trying to change the same attribute in a target front during the same cycle.
- have code that is robust to the unpredictable timing of the change: there is no way to control whether in the *cycle* of the change the target front will call its ``manage_front`` method before or after the change was made. 
 
There are two approaches possible to dealing with the first challenge:

- unique pair-wise relation: avoid possible competition by making sure that only one front can make the change and that the target front does not use this attribute itself in the relevant time frame. This is the most robust approach if possible. An example can be found in the :ref:`migrationnote-label`: the growing tip of the filipod in Filipod migration uses ``soma.set_status1()`` to set the *status1* flag in the soma.
- ``constellation.lock(target)`` the target front before changing the attribute. This approach is guaranteed safe if several fronts can make the change. Unfortunately, if several fronts try to do this during the same cycle it is easy to trigger a lock competition causing a ``LockError``. Therefore this approach is not robust in many simulation contexts.

An example of using ``constellation.lock``::

    class MyFront(Front):
        _fields_ = Front._fields_ + [('foo', c_int)]

        def manage_front(self, constellation):
            ...
            # code called by a front that is not the soma
            soma = self.get_soma(constellation)
            if constellation.lock(soma): # lock soma before changing its attribute
                soma.foo = 25
                result = constellation.unlock(soma)  # and unlock it again

If two processes compete for access, one will need to wait till the lock of the other one is released. Therefore, it is important to ``unlock`` as soon as possible to avoid a ``LockError``. Every front is locked automatically during its ``manage_front`` call.

No direct access to shared arrays
---------------------------------
The underlying shared memory structure is quite complex to be able to deal with issues like different sized ``Front`` (due to additional attributes) and variable sized data structures (like the children of a ``Front``). Therefore direct user access is strongly discouraged, instead many access methods are provided. Most important::

    # Access other fronts by their ID
    fid = a_front.get_id() # obtain a front ID
    ...
    a_front = constellation.front_by_id(fid) # get the front back in another context

    # Access substrate by their ID
    sid = a_sub.get_id() # obtain a substrate ID
    ...
    a_sub = substrate_by_id(sid) # get the substrate back in another context

    # Get the parent front
    parent = self.get_parent(constellation)
    #   or its ID
    parent_ID = self.get_parent(constellation,returnID=True)
    #   or check whether a front is the parent
    if self.is_child(parent): # True if self is a child of parent
        ...

    # Get the child fronts as a list
    children = self.get_children(constellation)
    always_True = len(children) == self.num_children
    #   or get them as a list of IDs
    child_IDs = self.get_children(constellation,returnID=True)
    #   or check whether it is a child
    for child in children:
        if self.is_parent(child): # True if self is parent of child
            ...

    # Get the soma front
    soma = self.get_soma(constellation)
    #   or its ID
    soma_ID = self.get_soma(constellation,returnID=True)

    # Get all fronts belonging to a neuron
    all_fronts = self.get_neuron(constellation)
    #   or as IDs
    all_IDs = self.get_neuron(constellation, returnID=True)


Names are not strings
---------------------
Both the *neuron_name* and the optional *branch_name* are stored as fixed length character sequences. This has two consequences:

1. They have a fixed length of 40 or 20 characters, respectively. For the *neuron_name* 6 characters are reserved for the '_0_', '_1_',... numbering so the *neuron_name* parameter in the  *Admin_agent.add_neurons* method can only be 34 characters long, longer names cause an error.
2. Reading them directly does not return a ``string`` but a sequence of ``bytes``. Instead use methods that returns the proper ``string`` value: ``self.get_neuron_name(constellation)`` or ``self.get_branch_name()``.


Predefined array sizes
----------------------

All the shared memory consists of fixed size arrays. The default *Admin_agent* initialization allows for both small and medium size simulations, but for very large ones it may be necessary to increase some of the optional preset array sizes. This may have to be done by trial and error: NeuroDevSim will generate a ``OverflowError`` error if a preset array size is too small and the error message will instruct which *Admin_agent* initialization parameter needs to be increased. Incrementally increase this parameter value at *Admin_agent* initialization till the model runs without errors. See :ref:`simulator-label` for a complete listing of *Admin_agent* optional initialization parameters.
