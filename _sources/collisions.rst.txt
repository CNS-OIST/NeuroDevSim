.. _collisions-label:

Preventing and dealing with collisions
======================================
The emphasis on interactions with the environment entails that for most models dealing with collisions is an important part of the code.

Standard behavior
-----------------
The robust approach is to let collisions happen and deal with the resulting :ref:`collisionerror-label`. The code below from the :ref:`tutorial-label` shows an example where  *new_pos* is varied randomly till no collision occurs::

    def manage_front(self,constellation):
        ...
        count = 0 # counts number of add_child trials
        while count < 100:
            extension = self.unit_heading_sample(width=20)
            new_pos = self.end + extension * 5. # compute position of child end
            # check for possible collisions
            try:
                new_front = self.add_child(constellation,new_pos) # make a new front and store it
                ...
            except CollisionError as error:
                count += 1
                continue # pick another new_pos, no attempt to correct the error
            except (GridCompetitionError, InsideParentError, VolumeError):
                count += 1
                continue # pick another new_pos, no attempt to correct the error
        print ("Warning: failed extension for of",self.get_neuron_name(constellation))
                
Obivously this simple approach is not guaranteed to succeed, especially in crowded environments. It is always important to deal with failure of the method, in this example a warning is printed.

Getting more information about collisions
-----------------------------------------

To deal more intelligently with collisions it is important to know which ``Front`` caused the collision, this information is available in the :ref:`collisionerror-label`::

    def manage_front(self,constellation):
        ...
            except CollisionError as error:
                print (self,"collides with",error.collider,"with distance",error.distance)
        ...

Note that standard behavior is to return only the first ``Front`` identified as causing a collision, there may be other ``Fronts`` that also cause collisions and these may even be closer by. Usually collisions with older fronts will be detected first.

It is possible to force a search for **all** colliding fronts before triggering an error::

    def manage_front(self,constellation):
        constellation.only_first_collision = False
        ...
            except CollisionError as error:
                if error.only_first:
                    print (self,"collides with",error.collider,"with distance",error.distance)
                else:
                    print (self,"collides with:")
                    for i in range(len(error.collider)):
                        print ("  #",i,":",error.collider[i],"with distance",error.distance[i])
        ...

The *constellation.only_first_collision* attribute is a boolean that is initialized to True. If this is set to False before the call to ``add_child`` the simulator will check for all collisions with proposed *new_front* before returning with :ref:`collisionerror-label`. Note that coding this correctly is not simple:

1. *constellation.only_first_collision* is local to each parallel processor and cannot be set globally. There are two strategies possible to using it: 

    - either set it at the begin of each ``manage_front`` call as in the example above. This will affect all ``add_child`` calls and slow down the simulation.
    - change it to False just before the ``try`` and ``except`` statements for a selected ``add_child`` call and reset to True afterwards, this will affect only that one ``add_child`` call.
    
2. depending on the setting of *constellation.only_first_collision* :ref:`collisionerror-label` returns either a ``Front`` or a ``[Front,]`` as collider, same for distance. The collider list is unsorted.
    
3. because the setting of *constellation.only_first_collision* may be ambiguous :ref:`collisionerror-label` contains its value used in its *first_only* attribute and will always print correct information.

Based on the information provided by :ref:`collisionerror-label` sophisticated collision resolution routines can be written.

Automatic collision resolution
------------------------------
Some fairly simple collision conditions can be very hard to solve properly by random search. An example is a dendrite or axon trying to grow past a much larger soma, biological growth cones will eventually succeed in making an arc around such a structure, but this requires a sophisticated simulation of chemical cues to work in NeuroDevSim. Instead, the ``solve_collision`` method provides a phenomenological solution that respects the original direction of growth. It is called as::

    points = self.solve_collision(constellation,new_pos,error)
    
``solve_collision`` returns a list of ``Point`` that were free at the time of the call. To generate the solution proposed the ``add_branch`` method should be used, which will create a series of a few fronts if possible::

    def manage_front(self,constellation):
        ...
        while count < max_count:
            new_pos = ...
            try:
                new_front = self.add_child(constellation,new_pos)        
                self.disable(constellation) # success -> disable this front
                return
            except CollisionError as error:
                points = self.solve_collision(constellation,new_pos,error)
                if points: # one or more points was returned
                    try:
                        new_fronts = self.add_branch(constellation,points)
                        # at least one new front made
                        self.disable(constellation) # success -> disable this front
                        return
                    except CollisionError as error:
                        print (self.get_neuron_name(constellation),self,"solve_collision collides with",error.collider)
                        count += 1
                        continue # generate another new_pos, no attempt to correct the error
                    except (GridCompetitionError,InsideParentError,VolumeError):
                        count += 1
                        continue # generate another new_pos, no attempt to correct the error
                else:
                    count += 1
                    continue # generate another new_pos
            except (GridCompetitionError,InsideParentError,VolumeError):
                count += 1
                continue # generate another new_pos, no attempt to correct the error
            ...
            
Note that ``solve_collision`` may fail and return an empty list. ``add_branch`` will try to instantiate fronts for every coordinate returned by ``solve_collision`` but this may fail. If at least one front can be made ``add_branch`` will return normally and the length of the *new_fronts* list returned gives the number of ``Front`` created, otherwise it will return with a new :ref:`collisionerror-label`. The reason that ``add_branch`` may fail partially or completely is that other processors may be instantiating new ``Front`` at coordinates needed after ``solve_collision`` returns and before or while ``add_branch`` is called.

Examples of the use of ``solve_collision`` can be found in the :ref:`migrationnote-label`.

