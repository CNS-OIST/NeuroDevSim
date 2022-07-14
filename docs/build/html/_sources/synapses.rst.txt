.. _synapses-label:

Synapses
========

Growth based synapses are possible when fronts are derived from the :ref:`synfront-label`. A ``Synapse`` can be purely structural but can also be used as an input signal. The use of synapses is extensively demonstrated in the  :ref:`synapsenote-label`.

Making synapses
---------------
A synapse can be made between any two non-migrating fronts that are not more than 5 µm apart. In making the synapse the user defines which front is presynaptic, the other is postsynaptic. At present, there can be only one synapse per front.

To make a synapse use the ``SynFront.add_synapse`` method with a known *other_front*. The weight determines whether it is excitatory (positive float) or inhibitory (negative float)::

    def manage_front(self,constellation):
        ...
        # make excitatory synapse from presynaptic asynfront1 to postsynaptic other_front1
        asynfront1.add_synapse(constellation,other_front1,1.)
        # make inhibitory synapse from postsynaptic asynfront2 to presynaptic other_front2
        asynfront2.add_synapse(constellation,other_front2,-1.,presynaptic=False)
        ...
        
The presence of a synapse can be detected with the ``self.has_synapse()`` method and its properties by ``self.get_synapse(constellation)``, ``self.is_presynaptic(constellation)`` or ``self.is_postsynaptic(constellation)``::

    def manage_front(self,constellation):
        ...
        if self.has_synapse():
            synapse = self.get_synapse(constellation)
            if self.is_presynaptic():
                print (self,"is presynaptic to",constellation.front_by_id(synapse.post_syn))
            else:
                print (self,"has postsynaptic to",constellation.front_by_id(synapse.pre_syn))
        ...

Note that synapses store the identity of the presynaptic (*pre_syn* attribute) and postsynaptic (*pos_syn* attribute) fronts as ``ID``.


.. _syn_input-label:

Using *syn_input*
-----------------
Each postsynaptic ``SynFront`` will update its its *syn_input* before the start of each cycle and this can be used as an input signal  in ``manage_front``. Note that the synaptic input is an average over the entire previous cycle.

The sign of *syn_input* is determined by whether the synapse is excitatory (positive *weight*) or inhibitory (negative *weight*)::

    def manage_front(self,constellation):
        ...
        if self.has_synapse():
            synapse = self.get_synapse(constellation)
            if synapse.weight > 0.:
                print (self,"has an excitatory synapse")
            elif synapse.weight < 0.:
                print (self,"has an inhibitory synapse")
        ...
        
The value of *syn_input* combines presynaptic properties, *firing_rate* and *CV_ISI*, with synaptic *weight*. In the absence of stochasticity (``CV_ISI == 0.``) it reflects an average over time: ``syn_input = firing_rate * weight``. If ``CV_ISI > 0.`` *syn_input* is stochastic and drawn from a normal distribution with mean *syn_input* computed as shown before. The presynaptic *firing_rate* and *CV_ISI* are set for the :ref:`neurons-label`.

The weight of the synapse can be changed to simulate synaptic plasticity::

    def manage_front(self,constellation):
        ...
        if self.is_postsynaptic():
            synapse = self.get_synapse(constellation)
            synapse.set_weight(constellation,5.)
        ...
        
By correlating presynaptic firing rate with postsynaptic responses correlation based synaptic plasticity rules can be implemented. Note, however, that these operate on a slow developmental time scale, it is not possible to simulate `spike-timing dependent plasticity <https://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity>`_ in NeuroDevSim!
        
Note that only the initial value of *weight* is automatically stored in the database, to store updated values of *weight* ``admin.attrib_to_db`` should be used as described in :ref:`storage-label`. Similarly, ``admin.attrib_to_db`` can be used to store *syn_input* values.

