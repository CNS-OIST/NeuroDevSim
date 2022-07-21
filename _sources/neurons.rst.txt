.. _neurons-label:

Neurons
=======

``Neuron`` is a class that contains information about all fronts belonging to the same neuron. It is automatically instantiated for each soma that is created by ``Admin_agent.add_neurons`` and keeps track of the number of fronts, retracted fronts and synapses (for ``SynFront`` only):: 

    def manage_front(self,constellation):
        ...
        neuron = self.get_neuron(constellation)
        print (self.get_neuron_name(constellation),"has",neuron.num_fronts,"fronts")
        ...
        
*num_fronts* and other similar ``Neuron`` attributes are updated at the end of each cycle, so the code above reports the number of fronts at the end of the previous cycle.
        
Neurons also contain two modifiable attributes that control synaptic input at :ref:`synapses-label`: *firing_rate* and *CV_ISI*. The first is the mean firing rate (over the duration of a cycle) and the second its variance, expressed as the coefficient of variation of the interspike intervals. These can be modified at any time with specific methods::

    def manage_front(self,constellation):
        ...
        neuron = self.get_neuron(constellation)
        neuron.set_firing_rate(constellation,5.) # increase firing_rate from default 1.
        neuron.set_CV_ISI(constellation,1.) # increase CV_ISI from default 0.
        print (self.get_neuron_name(constellation),": firing rate",neuron.firing_rate,"with CV",neuron.CV_ISI)
        ...

Changing *firing_rate* or *CV_ISI* has effects only for the synaptic input as shown in :ref:`syn_input-label`. Note that only the initial value of zero *firing_rate* and *CV_ISI* is automatically stored in the database, to store updated values ``admin.attrib_to_db`` should be used as described in :ref:`storage-label`.


