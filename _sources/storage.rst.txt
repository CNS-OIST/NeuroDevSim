.. _storage-label:

Storing additional attributes
=============================
It may be useful or necessary to store attributes that are not automatically stored by NeuroDevSim either because they need to be analyzed or plotted, or to prepare for :ref:`import-label`. These may either be user defined :ref:`additional-label` or attributes like ``Neuron`` *firing_rate* for which only the initial value is stored by default.  

The ``admin.attrib_to_db`` method supports such storing. ``attrib_to_db`` is called in the *main* part of the code, anytime after at least one instance of the ``Front`` subclass has been made, for example by ``add_neurons``. This code sample causes ``Front.num_children`` to be stored for all instances of ``SimpleNeuronFront``::

    if __name__ == '__main__':
        ...
        neuron_types = [SimpleNeuronFront,AxonFront]
        admin = Admin_agent(...)
        
        admin.add_neurons(SimpleNeuronFront,"simple_neuron",1,[[...],[...]],30.)

        admin.attrib_to_db(SimpleNeuronFront,"num_children","int")
    
``attrib_to_db`` requires minimally 3 parameters: the *subclass of ``Front``* for which the attribute should be stored, the *name of the attribute* to be stored and a *sql_type*. The latter defines the format that should be used to store the parameter in the  database and should be one of the following:

========= ===================== =========================================
sql_type  ctypes or class       columns in the data table
========= ===================== =========================================
int       c_bool, c_short,      1 *int* column containing the value
          c_int, c_long
real      c_double              1 *real* column containing the value
text      c_char                1 *text* column containing the text
id        ``id``                2 *int* columns containing both parts
point     ``Point``             3 *real* columns containing *x*, *y*, *z*
========= ===================== =========================================

An example of storing an additional attribute::

    class SimpleNeuronFront(SynFront):
        _fields_ = SynFront._fields_ + [('signal', c_double)]
    ...

    if __name__ == '__main__':
        ...
        neuron_types = [SimpleNeuronFront,AxonFront]
        admin = Admin_agent(...)
        
        admin.add_neurons(SimpleNeuronFront,"integrate_neuron",1,[[...],[...]],30.)

        admin.attrib_to_db(SimpleNeuronFront,"signal","real")


``attrib_to_db`` will only store attributes that are not stored yet or that are not being updated. Trying to store an unchanging attribute that is always stored, like for example ``Front.radius`` or ``Neuron.neuron_name`` will cause an error.

By default ``attrib_to_db`` assumes that the attribute to store belongs to ``Front``, but for ``Neuron`` *firing_rate* and *CV_ISI* and for ``Synapse`` *weight* can be stored but the class needs to be specified in the *object* optional parameter::

    admin.attrib_to_db(AxonFront,"firing_rate","real",object=Neuron)
    admin.attrib_to_db(SimpleNeuronFront,"weight","real",object=Synapse)
    
If multiple attributes from the same ``Front`` subclass or object are to be stored they can be specified as a list::

    admin.attrib_to_db(AxonFront,["firing_rate","CV_ISI"],["real","real"],object=Neuron)

Note that *sql_type* then also becomes a list, because not all attributes may have the same type. It is not possible to combine attributes from different *object* types in a list.

Another optional parameter controls how often the attribute is saved. By default it is saved for every cycle after the ``attrib_to_db`` call, but this may not be necessary. If the attribute is stored to prepare for :ref:`import-label` then only its final value is needed. This can be achieved by setting the *last_only* optional parameter to *True*::

    if __name__ == '__main__':
        ...
        neuron_types = [SimpleNeuronFront,AxonFront]
        admin = Admin_agent(...)    
        admin.importable_db = True

        admin.add_neurons(AxonFront,"axon",10,[[...],[...]],1.,axon=[...])
        
        admin.attrib_to_db(AxonFront,"goalID","id",last_only=True)

*goaldID* will now only be stored for the last cycle of the simulation. 

The last optional parameter of ``attrib_to_db`` controls for which neurons data will be stored. The *neuron_name* optional parameter will limit storage to data belonging to neurons with this name (*neuron_name* is a wildcard).

Several use cases of ``attrib_to_db`` are shown in the :ref:`databasenote-label`, this also includes analysis and plotting of stored data. An :ref:`import-label` use case is shown in the :ref:`importnote-label`.
