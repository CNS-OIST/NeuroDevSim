.. _interactive-label:

Interactive mode
================
It is quite easy to run NeuroDevSim in interactive mode, either in a notebook with plotting or from the terminal. However, the interactive mode comes with severe restrictions:

- nothing is stored, all results are transient
- there is no parallel computing so only simple models can be simulated
- a complete simulation is either interactive or not, one cannot switch

Nevertheless the interactive mode can be quite useful to gain intuition, explore ideas and, especially, debug complex models. These use cases are introduced in the :ref:`interactivenote-label`.

Basic interactive simulation
----------------------------
All that is needed to run an interactive simulation is to instantiate :ref:`admin_agent-label` with zero *num_procs*. A :ref:`front-label` needs to be defined because a *neuron_type* is required. A minimal set-up taken from the notebook example is::

    from neurodevsim.simulator import *

    class RandomFront(Front):
        
        def manage_front(self,constellation):
            pass

    if __name__ == '__main__':

        # initialize Admin_agent
        sim_volume = [[-100., -100., -100.], [100.0,100.0,100.0]]
        neuron_types = [RandomFront]
        admin = Admin_agent(0,"",sim_volume,neuron_types,verbose=0,plot=True) # interactive mode
        constellation = admin.constellation

Notice that the :ref:`manage_front-label` is empty, it will not be used. If desired one can have the normal ``manage_front`` present as reference, this may be useful while debugging. Because there is no database output no file name needs to be specified for ``Admin_agent``.

The :ref:`constellation-label` is obtained because this will be needed in many method calls.

Now any NeuroDevSim method can be called. Because most of these depend on object instantiation, relevant objects should be made first. For example, one could call ``admin.add_neurons`` to make a soma::

    fronts = admin.add_neurons(RandomFront,"rand_neuron",1,[[-30.,0.,0.],[-30,0.,0.]],5.)
    soma = fronts[0]

Notice that in this case we capture the return of ``admin.add_neurons`` so that the *soma* that was created can be accessed. Using ``soma.add_child`` more fronts can now be created to simulate simple growth or any other ``Front`` method can be called as desired. Remember that the fronts are **not stored** in the simulation database and if notebook plotting is enabled, new fronts need to plotted explicitly with ``admin.plot_item``::

    line = admin.plot_item(soma,color='k')
    
There is no need to call ``admin.destruction`` because no extra processes were spawned.

Interactive model debugging
---------------------------
Because NeuroDevSim simulations are not reproducible they can be very difficult to debug in a traditional way. Instead the output of a buggy simulation can be loaded with the ``import_simulation`` method and interactive mode can be used to investigate what went wrong. 

To use this approach effectively it is important to identify "problem" fronts. This information can be obtained either by printing out relevant front IDs during the simulation or by analyzing the database content as explained in :ref:`database-label`. 

Start an interactive session with the same *sim_volume* and *neuron_types* as used for the stored simulation, as shown above. Then import an existing simulation database::

        ...
        admin = Admin_agent(0,"",sim_volume,neuron_types,verbose=0,plot=True) # interactive mode
        constellation = admin.constellation
        
        admin.import_simulation("simulation.db")
        
The complete simulation till the end will be loaded and because ``plot=True`` plotted. For large simulations, plotting takes a lot of time. This can be prevented by plotting only a relevant region of the simulation, using the ``Admin_agent`` *box* attribute::

    admin = Admin_agent(0,"",sim_volume,neuron_types,verbose=0,plot=True,box=[[60.,60.,0.],[90.,90.,30.]])

To use *box* effectively one should know which region to focus on, usually centered around the "problem" front. It is best to use an isometric box, with idential ax lengths for each dimension. Only when a *box* is defined, the list of ``Front`` plotted is available as *admin.plot_items*. If the number of fronts plotted is small, investigating this list is the fastest way to discover what is plotted.

After ``import_simulation`` all fronts that existed at the end of the simulation are present in memory and can be accessed by their ``ID``. 

If "problem" fronts were identified using print statements during the original simulation information like this will have been printed::

    Front ID: neuron type 1, index 4005
    
the corresponding front can now be obtained with::

    my_front = constellation.front_by_id(ID(1,4005))
    
If "problem" fronts were identified in the database, the procedure is a bit more complicated. Each front has two numerical identifiers in the database: a *neuron_id* and a *front_id*, see :ref:`database-label`. Combined, these constitute a ``DataID`` which is unfortunately different from ``ID``, but one can easily be converted into the other using ``constellation.data_to_id``::

    my_ID = constellation.data_to_id(DataID(neuron_id,front_id))
    my_front = constellation.front_by_id(my_ID)

Where is *my_front* in the simulation plot? As this is often not easy to recognize, one can make any plotted front flash during an interactive session::

    admin.flash_front(my_front)
    
If the plot is crowded, it may have to be rotated before the flashes are visible. One can ``flash_front`` as often as necessary.

The next steps depend on the type of problem to solve. Let's, for example, look at a fatal collison during an ``add_child`` call::

    new_pos = ...
    try:
        new_front = my_front.add_child(constellation,new_pos)
        print (new_front)
    except CollisionError as error:
        print (error)
        colerror = error # error is only defined within the except scope
        
Either a *new_front* will be created and printed or a ``CollisionError`` occurs and then this will be printed. If the error occurred, one could then try ``solve_collision``::

    points = my_front.solve_collision(constellation,new_pos,colerror)
    print (len(points),"points:",points)
  
If ``solve_collision`` fails (it returns an empty list), maybe this is due to multiple colliding structures? By default only the first collision is detected as explained in :ref:`collisions-label`, but this can be changed so that **all** collisions are detected::

    constellation.only_first_collision = False # report all collisions
    colerror = None
    try:
        new_front = my_front.add_child(constellation,new_pos)
        print (new_front)
    except CollisionError as error:
        print (error)
        colerror = error # error is only defined within the except scope
    if colerror:
        for f in colerror.collider: # print all the colliding structures
            print (f)
            
The possibilities of the interactive mode are endless... One can test large sections of code, the behavior of specific methods, explore alternatives, etc. Just remember that nothing gets stored!  

Finally, when investigating :ref:`migration-label`, the interactive mode has an additional useful feature: it can provide a history of the migration of any migrating soma that was loaded by ``import_simulation`` with ``get_migration_history``::

    my_somaID = constellation.data_to_id(DataID(neuron_id,front_id))
    my_soma = constellation.front_by_id(my_somaID)
    coordinates, cycles = my_soma.get_migration_history(constellation)

*coordinates* is a list of ``Point`` representing *my_soma.orig*, from the location where it was created up to the last cycle, and *cycles* contains the corresponding cycle for each entry in *coordinates*. One can print this information or plot it::

    lines = admin.plot_item(coordinates,color='k',line=True)
