.. _import-label:

Importing a simulation
======================

Some developmental models may simulate consecutive stages in development and each stage may be optimized separately during model creation. If the simulations take a lot of time to run it may then be advantageous to use a previous simulation of early stages of development to start simulating a later stage. 

This is possible with the ``Admin_agent.import_simulation`` method::

    fname_old = "output/prev_simulation.db"
    fname = "output/new_simulation.db"
    sim_volume = [[0., 0., 0.], [100., 100., 100.]]
    neuron_types = [MyFront1,MyFront2]
    num_procs = 4
    admin = Admin_agent(num_procs,fname,sim_volume,neuron_types,seed=1)
    admin.import_simulation(fname_old)

There are strict limitations to the use of ``import_simulation``:

1. The database should be *importable*: this should be set in the previous simulation as ``admin.importable_db = True``, best immediately after :ref:`admin_agent-label` initalization.
2. The importing simulation shoul call ``import_simulation`` as the first method after initalization of :ref:`admin_agent-label`.
3. :ref:`admin_agent-label` initalization should be almost identical to that used for the previous simulation: *num_procs*, *sim_volume* and *neuron_types* should be identical (*num_procs* can be 0 for :ref:`interactive-label`). The code defining each neuron type should be compatible with the previous simulation. Optional changes to ``Admin_agent`` array size attributes should also be identical.
4.  The database name for the new simulation should be different from that of the previous one. By default the database contents are copied to the new simulation database, but this can be turned off with the optional *copy_db* parameter.
5.  ``Admin_agent`` *seed* and *verbose* can be different.

The new simulation will start at the next cycle after the last one stored in the previous simulation database. Because it uses an *importable* database it has all the information needed to continue the simulation as if it was never interrupted. If any :ref:`additional-label` were declared their values should be stored during the previous simulation, this can be done with :ref:`storage-label` and setting the optional parameter ``last_only=True``.

Several use cases of ``import_simulation`` are shown in the :ref:`importnote-label`.

