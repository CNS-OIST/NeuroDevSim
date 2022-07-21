.. _database-label:

Understanding the database
==========================
All NeuroDevSim simulations are saved in a `SQLite database <https://www.sqlite.org/index.html>`_. Many SQLite database browser apps are available to view the contents of the database, we like the `DB Browser for SQLite <https://sqlitebrowser.org/>`_.

NeuroDevSim comes with a set of methods in the :ref:`processing-label` that print or plot database contents, so most users may never have to read this section of the manual. But understanding of the database structure is necessary to write specific analysis routines and may also be required to use the :ref:`interactive-label` effectively.

Database tables
---------------
Information is stored in different database tables. These tables are shown in order of relevance for the user below and important tables are discussed in more detail next. Optional tables are tables that are only made when relevant: either when the corresponding class is instantiated or when  :ref:`storage-label` has been used.

=================== =========== ==================================================================================
Table name          Optional    Description
=================== =========== ==================================================================================
neuron_data         no          data for every *Neuron* in simulation, necessary for *front_data*
front_data          no          data for every *Front* in simulation, needed for *synapse_data*, *migration_data1*
synapse_data        yes         data for every *Synapse* in simulation
substrate_data      yes         data for all *Substrate* in simulation
migration_data1     yes         coordinates of migrating somata, additional *migration_data2*,... may be present
neurodevsim         no          basic information about simulation: volume, number cycles, software version,...
neuron_types        no          class names of each *neuron_type* in simulation
attributes          yes         list of all tables storing additional attributes (tables not listed here)
arc_data            yes         technical table: data for all *Arcs*, only needed for *import_simulation*
arc_points          yes         technical table: points for all *Arcs*, only needed for *import_simulation*
mig_fronts_data     yes         technical table: order of migrating fronts, only needed for *import_simulation*
sqlite_sequence     no          technical table: standard sqlite table listing all other tables and their length
=================== =========== ==================================================================================

The contents of the database are updated at the end of each cycle. If a simulation crashes or is stopped, the database will be readable and contain correct information up till the last complete cycle. However, changing content in several tables is only updated at the end of the simulation by ``Admin_agent.destruction``.

Reading the database
--------------------
This subsection will be familiar to anybody who has previously written code to read from 
a `SQLite database <https://www.sqlite.org/index.html>`_. One connects to the database using its filename *db_name* and creates a *cursor*::

    import sqlite3
    ...
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    conn.row_factory = sqlite3.Row

The last statement is quite important because it allows to access content by the name of columns, which results in more readable and easier to manage code.

Next one usually loads an entire table *this_table* and analyzes it row by row::

    cursor.execute("select * from this_table")
    result = cursor.fetchall()
    for row in result:
        item1 = row['name_of_column2']
        item2 = row['name_of_column3']
        ...

Each table has as first column an *id* created by SQLite itself, this *id* is not related to the simulation. The content of relevant tables is described briefly in the following subsections, but *id* will not be mentioned. To get started yourself, it maybe helpful to look at the code in :ref:`processing-label`, for example the ``nds_neuron`` method is fairly easy to understand.

NeuroDevSim writes new data to the database at the end of every cycle, so its information is complete up to the last completed cycle in case of a crash of the simulation. **Type** refers to SQLite data types, not NeuroDevSim data types. **Updated** refers to whether the value may be updated from its initial value when the object was created.

neuron_data table:
------------------
This table contains a row for each ``Neuron`` created and (a) new row(s) will be written to the database at the end of each ``admin.add_neurons`` call, it has the following columns:

============== ===== ============================================================== ========
Column_name    Type  Description                                                    Updated
============== ===== ============================================================== ========
neuron_id      int   unique to each neuron, is the neuron identifier in ``DataID``  no
type_id        int   identifies *neuron_type*, is the neuron identifier in ``ID``   no
name           text  name of the neuron                                             no
firing_rate    real  initial *firing_rate* of neuron                                no  
CV_ISI         real  initial *CV_ISI* of neuron                                     no  
num_fronts     int   final number of fronts in neuron                               at end
num_retracted  int   final number of retracted fronts in neuron                     at end
num_synapses   int   final number of synapses in neuron                             at end
============== ===== ============================================================== ========

'at end' means that the column value is updated by ``admin.destruction()``.

front_data table:
------------------
This table contains a row for each ``Front`` created, with the following columns:

=============== ===== ======================================================================= =======
Column_name     Type  Description                                                             Updated
=============== ===== ======================================================================= =======
neuron_id       int   neuron the front belongs to, refers to similar column in *neuron_data*  no
front_id        int   unique to each front of a specific *neuron_type*,                       no
                      front identifier in ``DataID`` and ``ID``
branch          text  optional *branch_name* of front                                         no
swc_type        int   *swc_type* of the front, see :ref:`swc-label`                           no
shape           int   indicates spherical (*shape* == 1) or cylindrical (*shape* == 2) front  no
orig_x          real  *x* of the *orig* coordinate of the front, for a migrating soma this    no
                      is its original  position 
orig_y          real  *y* of the *orig* coordinate of the front                               no
orig_z          real  *z* of the *orig* coordinate of the front                               no
end_x           real  *x* of the *end* coordinate of the front, for spherical fronts          no
                      identical to their *orig* 
end_y           real  *y* of the *end* coordinate of the front                                no
end_z           real  *z* of the *end* coordinate of the front                                no
radius          real  front *radius*                                                          no
parent_id       int   the *front_id* of the parent front or -1 for the root of a neuron tree  maybe
b_order         int   branching *order* of the front, 0 at the root                           no
path_len        real  cumulated *path_length* from soma till *end* of the front               no
birth           int   *birth* of the front, cycle when the front was created                  no
death           int   *death*, -1 or if retracted cycle when the front was retracted          maybe
migration       int   column number in *migration_data* table for migrating soma, 0 if not    no
                      migrating
flags           int   :ref:`flags-label`                                                      maybe
=============== ===== ======================================================================= =======

The *death* column value is updated at the end of the cycle of retraction, otherwise it stays -1. The *parent_id* (may have changed due to :ref:`migration-label`) and *flags* columns are updated at the end of simulation by ``admin.destruction()`` if ``admin.importable_db == True``, otherwise they keep the original value.

Note that *num_children* and the parent to child relation are not stored in the database. This information is implicit in the child to parent relation that is stored in the *parent_id* column.

synapse_data table:
-------------------
This table contains a row for each ``Synapse`` created, with the following columns:

============== ===== ============================================================== ========
Column_name    Type  Description                                                    Updated
============== ===== ============================================================== ========
pre_neuron_id  int   identifies presynaptic neuron, refers to *neuron_id* in        no
                     *neuron_data*
pre_front_id   int   identifies presynaptic front, refers *front_id* in             no
                     *front_data*
post_neuron_id int   identifies postsynaptic neuron, refers to *neuron_id* in       no
                     *neuron_data*
post_front_id  int   identifies postsynaptic front, refers *front_id* in            no
                     *front_data*
weight:        real  initial synaptic *weight*                                      no
birth          int   *birth* of the synapse, cycle when the synapse was created     no
death          int   *death*, -1 or if removed cycle when synapse was removed       maybe
============== ===== ============================================================== ========

The *death* column value is updated at the end of the cycle of removal, otherwise it stays -1.

substrate_data table:
---------------------
This table contains a row for each ``Substrate`` created, with the following columns:

============== ===== ============================================================== ========
Column_name    Type  Description                                                    Updated
============== ===== ============================================================== ========
name           text  *name* of the substrate                                        no
x              real  *x* of the *orig* coordinate of the substrate                  no
y              real  *y* of the *orig* coordinate of the substrate                  no
z              real  *z* of the *orig* coordinate of the substrate                  no
amount         real  *n_mol* of the substrate                                       no
rate           real  *rate* of the substrate                                        no
diff_c         real  *diff_c* of the substrate                                      no
birth          int   *birth* of the substrate, cycle when the substrate was created no
death          int   *death* -1                                                     no
============== ===== ============================================================== ========

At present ``Substrate`` cannot be removed.

migration_data table:
---------------------
The database can contain several migration tables, numbered consecutively as *migration_data1*, *migration_data2*, *migration_data3*,... This is because the number of columns in a SQLite database table is limited, so if more than 600 somata migrated an extra *migration_data* table will be created. 

These tables contains a row for each cycle during which a migration event took place. Its first column is the cycle and then 3 columns for each migrating soma:

============== ===== ============================================================== ========
Column_name    Type  Description                                                    Updated
============== ===== ============================================================== ========
cycle          int   the cycle at which each soma migrated to these positions       no
x_...          real  for each front identified by the elements of ``ID``, the       no
                     x position of the coordinate it migrated to this cycle or 
                     *NULL* if it did not migrate during this cycle
y_...          real  same for the y position of the coordinate                      no
z_...          real  same for the z position of the coordinate                      no
============== ===== ============================================================== ========

If multiple *migration_data* table are present the *cycle* column of each table is unique.

neurodevsim table
-----------------
This table contains only a single row with information about the simulation in the following columns:

============== ===== ================================================================= ========
Column_name    Type  Description                                                       Updated
============== ===== ================================================================= ========
xmin           int   *x* of the left-front-bottom coordinate of the simulation volume  no
ymin           int   *y* of the left-front-bottom coordinate of the simulation volume  no
zmin           int   *z* of the left-front-bottom coordinate of the simulation volume  no
xmax           int   *x* of the right-back-top coordinate of the simulation volume     no
ymax           int   *y* of the right-back-top coordinate of the simulation volume     no
zmax           int   *z* of the right-back-top coordinate of the simulation volume     no
num_cycles     int   total number of cycles simulated                                  at end
num_procs      int   number of computing processes used to instantiate ``Admin_agent`` no
version        real  value representing the NeuroDevSim version number multiplied by   no
                     100., used by many methods to check whether database can be read
run_time       real  run time of the simulation in seconds                             at end
importable     int   can database be read by ``import_simulation`` (1) or not (0)      at end
substrate      int   number of *substrate* tables present in the database (0 or 1)     yes
migration      int   number of *migration* tables present in the database              yes
synapses       int   number of *synapses* tables present in the database (0 or 1)      yes
attributes     int   number of tables storing attributes present in the database       yes
arcs           int   number of arc related tables present in the database (0 or 2)     at end
============== ===== ================================================================= ========

'at end' means that the column value is updated by ``admin.destruction()``.

neuron_types table
------------------
This table contains a row for each ``Front`` subclass listed in *neuron_types* during instantiation of :ref:`admin_agent-label`.

============== ===== ================================================================= ========
Column_name    Type  Description                                                       Updated
============== ===== ================================================================= ========
type_id        int   index into shared arrays, is the neuron identifier in ``ID``      no
neuron_type    text  *class name* of ``Front`` subclass                                no
============== ===== ================================================================= ========

attributes table
----------------
This table contains a row for each attribute that was stored using ``admin.attrib_to_db``, it is updated by this method.

============== ===== ================================================================= ========
Column_name    Type  Description                                                       Updated
============== ===== ================================================================= ========
name           text  *name* of the attribute table                                     no
type           text  SQLite type of the attribute                                      no
neuron_name    text  *neuron_name* optional parameter of ``admin.attrib_to_db``        no
last_only      int   *last_only* optional parameter of ``admin.attrib_to_db``          no
============== ===== ================================================================= ========

Attribute data tables
---------------------
Several such tables may be present with names listed in the *attributes* table: the name of the attribute followed by `_data`. It contents depend on the SQLite type of the attribute.

==================== ========= ================================================================= ========
Column_name          Type      Description                                                       Updated
==================== ========= ================================================================= ========
neuron_id            int       identifies neuron, refers to *neuron_id* column in *neuron_data*  no
front_id             int       identifies front if relevant (0 for ``Neuron``), refers to        no
                               *front_id* column in *front_data*
cycle                int       simulation *cycle* for which data is stored                       no
*1, 2 or 3 columns:*
attribute name       int       integer value of a simple attribute                               no
*or*
attribute name       real      real value of a simple attribute                                  no
*or*
attribute name       text      text value of a simple attribute                                  no
*or*
attribute_ID_0       int       *type_id* component of an ``ID``                                  no
attribute_ID_1       int       *front_id* component of an ``ID``                                 no
*or*
attribute_x          real      *x* value of a ``Point``                                          no
attribute_y          real      *y* value of a ``Point``                                          no
attribute_z          real      *z* value of a ``Point``                                          no
==================== ========= ================================================================= ========

Database from a crashed simulation
----------------------------------
The database of a crashed simulation will be intact but incomplete and never importable. 

All tables will have up to date information till the cycle before the crash, but none of the data marked as **Updated** *at end* in the table listings will have been updated. The easiest way to identify a database as being from a crashed simulation is to check the *num_cycles* column in the *neurodevsim* table: it will be 0 (except after `import_simulation`, then it will be final cycle of the imported database). The number of cycles stored can be determined by the *birth* of the last fronts stored in the *front_data* table.

