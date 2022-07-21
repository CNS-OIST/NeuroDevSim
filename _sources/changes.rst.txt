Changes compared to NeuroMaC
****************************

Major conceptual differences
============================
Conceptually NeuroDevSim works like NeuroMaC, but there are extensive coding style  changes. The main differences are:

1. No subvolumes and no virtual subvolumes, instead a number of parallel processes are specified. Parallelism is based on shared memory instead of interprocess communication, resulting in much faster code (80x faster for motor_neuron model).
2. The use of shared memory implies fixed data structures: the sizes have to be defined at start-up and cannot be changed. Array overflow errors are possible.
3. As a consequence, all objects must be completely defined before simulation and cannot be changed during simulation, no new attributes can be defined during simulation. See the Tutorial Important dos and don'ts section for more details.
4. All object instances are always accessible. However, they cannot be stored as object attributes, instead store their ID and later retrieve access to the object instance using this ID. One can even change attributes in other object instances, but this requires acquiring a *lock* on the instance. See the Tutorial Important dos and don'ts section for more details.
5. Because objects instances cannot expand, several attributes like front children are stored in a rather complex manner. They can no longer be accessed directly but methods are provided to return the data. See New methods below.
6. Coordinates are encoded as ``Points`` with *x*, *y* and *z* attributes. *Numpy arrays* are no longer used. Coordinates can be negative.
7. ``Front`` initialization is no longer required in subclasses, only use it to initialize a new attribute to a value other than zero.
8. Collisions are handled differently in multiple ways. The ``Front`` *point_not_valid* method (which replaces NeuroMaC *xyz_not_valid*) can still be used to prevent collisions but does NOT see soma migration or new fronts managed by other processors at the current cycle. Therefore, even if *point_valid* returns with 1 (no collisions detected), a collision event can be triggered at a later stage when changes managed by different processors are reconciled. The *collision* method is much more robust than before but can no longer be subclassed, instead subclass *wiggle_front*. There is an option to have failed collisions put the calling front into dormant mode instead of terminating growth or migration.
9. NeuroDevSim is not always 100% reproducible for the same seed value. This strongly depends on how crowded the simulation is, for most of the example simulations it is still 100% reproducible. The lack of full reproducibility is caused by slight differences in timing among parallel processes. When this happens the simulation will produce a small set of possible outcomes.
10. Better jupyter notebook support: simulations now run as fast in notebooks as in the terminal. Faster notebook graphics, though they still slow down the simulation.
11. All necessary classes, methods and functions are defined in a single *simulator* module. The *processing* module is equivalent to the NeuroMaC's *scripts* module.

Differences to classes and new classes
======================================
``Front``: *xyz0* is now *orig* and *xyz* is *end*, *orig* and *end* are ``Point``. No *shape* attribute, instead ``Front`` instantiation has a *cylinder* optional parameter which controls whether a cylinder (default) or sphere is created. The shape can be obtained with the new *is_cylinder* method. Neither *parent* nor *children* can be accessed as fields, instead the *get_parent* or *get_children* methods should be used. Similarly, **get_neuron_name** and **get_branch_name** should be used for the names. ``Front`` subclasses can be instantiated directly, *new_der_front* is no longer available.

``ID``: represents the id of a ``Front`` or ``Substrate``.

``Point``: a new class that represents coordinates or vectors, has *x*, *y*, *z* attributes.

Differences to methods and functions
====================================
Where possible, functions have been turned into methods.

**alternate_locations** is a ``Front`` method: it is called as ``front.alternate_locations(point,distance,number)``.

**angle_two_directions** has been renamed to **angle_two_dirs**

**extend_front**: the *cycle* argument has been removed. The 's' return is no longer available, use *constellation.add_substrate()* instead. At present no 'r' return.

**front_by_id**: is a ``Constellation`` method.

**front_distance**: *other* parameter can be a ``Point`` or ``Substrate``.

**front_surface_point** is now *surface_point_to*.

**get_fronts** is a ``Front`` method: it is called as ``front.get_fronts(constellation)`` where *front* is equivalent to the NeuroMaC *reference* parameter. There is a *returnID* optional parameter that if True returns a list of ``ID`` instead of ``Front`` and the *distal* optional parameter has been removed. Other parameters have not changed.

**get_substrates** is a ``Front`` method: it is called as ``front.get_substrates(constellation,name)`` where *front* is equivalent to the NeuroMaC *reference* parameter. There is a *returnID* optional parameter that if True returns a list of ``ID`` instead of ``Front`` and the *distal* optional parameter has been removed. Other parameters have not changed.

**migrate_front**: the *cycle* argument has been removed. The 'ma', mf' and 'mfa' returns no longer require the ``Fronts`` to be returned. Instead this is implicit: the single filipod and/or single axon child, identified by proper swc_type, is automatically used.

**unit_branching_sample** is a ``Front`` point method: it is called as ``front.unit_branching_sample(number)`` where *front* is equivalent to the NeuroMaC *front* parameter. Other parameters have not changed.

**unit_heading_sample** is a ``Front`` point method: it is called as ``front.unit_heading_sample()`` where *front* is equivalent to the NeuroMaC *front* parameter. Other parameters have not changed.

**wiggle_front** is a ``Front`` point method. It no longer changes *self* but returns a ``Point`` that can be used to update a *self* coordinate in the **collision** method.

**xyz_not_valid**: is now **point_valid**.

New methods and functions
=========================
**abs_coord**: ``Point`` method that makes all its attributes positive.

**add_neurons**: ``Admin_agent`` method that replaces *initialize_neurons*, with the same arguments.

**add_substrate**: both an ``Admin_agent`` and a ``Constellation`` method. Replaces the *Admin_agent.set_substrate* method and the 's' return of *extend_front*.

**cross**: ``Point`` method that performs vector cross operation between two 1D vectors.

**dot**: ``Point`` method that performs vector dot operation between two vectors.

**extend_repeat**: ``Front`` behavior method that signals repeating calls to ``extend_front``.

**get_branch_name**: ``Front`` attribute method that returns the optional branch name.

**get_children**: ``Front`` tree method that returns a list of all children as ``Front`` or ``ID``.

**get_colliding_fronts**: `Front`` search method returns a list of all fronts it collided with during its ``extend_front`` or ``migrate_front`` call.

**get_id**: ``Front`` attribute and ``Substrate`` method that returns the ``ID``.

**get_name**: ``Substrate`` method that returns its name.

**get_neighbors**: ``Front`` tree method that returns a list of neigboring fronts as ``Front`` or ``ID``.

**get_neuron**: ``Front`` tree method that returns a list of all neuron fronts as ``Front`` or ``ID``.

**get_neuron_name**: ``Front`` attribute method that returns the name of the neuron.

**get_parent**: ``Front`` tree method that returns the parent ``Front``.

**get_soma**: ``Front`` attribute method that returns the soma ``Front``.

**has_moved**: ``Front`` behavior method that returns whether moved in previous cycle.

**interstitial_front**: ``Front`` growth method that implements interstitial branching.

**in_volume**: ``Point`` method that returns whether it is inside the simulation volume.

**is_child**: ``Front`` tree method that confirms whether self is a child of front.

**is_cylinder**: ``Front`` behavior method that returns shape.

**is_death**: ``Front`` behavior method that returns deletion at birth.

**is_growing**: ``Front`` behavior method that returns active growth status.

**is_interstitial**: ``Front`` behavior method that returns active interstitial status.

**is_migrating**: ``Front`` behavior method that returns active migration status.

**is_retracted**: ``Front`` behavior method that returns whether retracted.

**is_retracting**: ``Front`` behavior method that returns active retraction status.

**is_parent**: ``Front`` tree method that confirms whether self is the parent of front.

**length**: ``Point`` and ``Front`` methods that compute length of a vector or of a front.

**lock**: a ``Constellation`` method that locks a ``Front`` or ``Substrate`` so that its user defined attributes can safely be changed.

**mid**: ``Front`` point method, returns center point on axis of front.

**norm**: ``Point`` method that normalizes vector to unit vector.

**nparray**: ``Point`` method that turns vector into numpy.array.

**point_valid**:  is a ``Front`` point method replacing *xyz_not_valid*. Has a *cylinder* optional parameter that controls whether *point* belongs to a cylinder or to a sphere, returns an integer with 1 indicating a valid point.

**solve_collision**: is a ``Front`` point method that tries to return a point that does not collide. At present only solves collisions with spheres.

**sphere_interpol**: is a ``Front`` point method  returns a point on or close to the membrane surface of spherical self on an arc connecting two points.

**substrate_by_id**: a ``Constellation`` method that returns ``Substrate`` with given ``ID``.

**surface_point_to**: is a ``Front`` point method and called as: `front.surface_point_to(point)` where *point* is equivalent to NeuroMac *self.xyz*. Other arguments have not changed compared to old *front_surface_point*.

**taper**: ``Front`` size method, returns tapered *radius*.

**unlock**: a ``Constellation`` method that unlocks a previously locked ``Front`` or ``Substrate``.

Methods and functions that no longer exist
==========================================
**collision** is no longer a public method, subclass **wiggle_front** instead.

**common_branch_point**: use the **is_child** method on both fronts.

**direction_to**: compute it yourself as the difference between two ``Point`` coordinates.

**direction_to_plane**: compute it yourself as the difference between two ``Point`` coordinates.

**exp_gradient_to**

**front_surface_point**: replaced with **surface_point_to**.

**get_length**: use ``Point`` **length** method.

**get_other_structure**: use ``Front`` **get_fronts** method.

**get_self_structure**: use ``Front`` **get_fronts** method.

**get_structure**: use ``Front`` **get_fronts** method.

**initialize_neurons**: renamed to **add_neurons**

**initialize_subvols**

**internal_front**: will be implemented later.

**load_substrate**

**load_subvols**

**load_swc**: will be implemented later.

**new_der_front**: instantiate a subclass of ``Front`` directly.

**normalize_length**: use ``Point`` **norm** method to get unit vector and multiply that by *length*.

**retract_front**: will be implemented later.

**rotate_vector**

**set_substrate**: renamed to **add_substrate**

**sigmoid**: will be implemented later.

**xyz_not_valid**: replaced with **point_valid**.


