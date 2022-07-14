.. _substrate-label:

Substrate
==========

Substrate is a class that implements chemical cues at different locations. It is the only class in NeuroDevSim that can be directly initiated with syntax ``Substrate(name,location,creation_cycle,amount)``. For example::

    sub = Substrate("attractor",Point(10.,10.,10.),0,1)

creates a substrate named *attractor* at coordinate (10.,10.,10.) which will exist from the beginning of the simulation (cycle 0) and consists of a single molecule. Such a simple substrate is sufficient for a **deterministic** :ref:`substratecue-label`.

However, before it can be used it should be added to the simulation. This can be done in two different ways: in the main code or by a ``Front`` during its ``manage_front`` call. The first case uses ``Admin_agent.add_substrate`` as shown in the :ref:`environmentnote-label`::

    if __name__ == '__main__':
        ...
        # add substrate information, in this case a single random point
        x = 20 + 60 * np.random.random()
        y = 20 + 60 * np.random.random()
        # instantiate a substrate
        sub = Substrate("attractor",Point(x,y,99.),...)
        # add it to the simulation volume
        admin.add_substrate(sub)
        
Alternatively, it can be produced by a ``Front`` that uses ``constellation.add_substrate``::

    def manage_front(self,constellation):
        ...
        sub = Substrate("attractor",self.mid(),0,1)
        constellation.add_substrate(sub)
        
though this is not enforced, the assumption is that the substrate is produced close to the front, in this case at its center.

Either ``add_substrate`` can be called with a single ``Substrate`` or a list of ``Susbstrate`` as parameter. Multiple instantiations of ``Substrate`` with the same name at different locations can be created and added. Once registered by ``add_substrate``, all substrate of a given name can be found with the ``get_substrates`` as described in the :ref:`substratecue-label`.

In addition to the simple definitions used till now, additional parameters can be supplied which allow for **stochastic** use of ``Substrate`` with the ``diff_gradient_to`` method described in :ref:`substratecue-label` and illustrated in :ref:`environmentnote-label`. For this stochastic approach a diffusion constant *diff_c* in µm^2/cycle needs to provided::

    sub = Substrate("attractor",Point(10.,10.,10.),constellation.cycle,1000000000,diff_c=30.)
    
This will place a point source of 1,000,000,000 molecules of attractor at the given location on the given cycle, which will then start to diffuse away. The computed gradient will evolve as the *cycle* parameter passed to ``diff_gradient_to`` increases.

An alternative is to use a continuously producing point source. In that case the *amount* is ignored and, instead, an optional *rate* parameter is provided::

    sub = Substrate("attractor",Point(10.,10.,10.),constellation.cycle,0,rate=1000.,diff_c=30.)
    
For the continuously producing point source, stochastic concentratons can be queried by ``diff_gradient_to`` either some cycles after the start or using a steady state gradient, see :ref:`substratecue-label`.







