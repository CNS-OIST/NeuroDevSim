.. _subclassing-label:

Subclassing ``Front``
=====================
Every NeuroDevSim model requires at least one ``Front`` subclass definition with its own *manage_front* method::

    class MyFront(Front):

        def manage_front(self,constellation):

This is how specific models are defined, as explained in :ref:`started-label`. In this subsection additional, optional aspects of subclassing are described in detail.

.. _additional-label:

Additional attributes
---------------------
It may be useful to store  data that is needed to control the behavior of fronts or that introduces new dynamics to the model as an additional ``Front`` attribute. There are strict rules on using additional attributes for a ``Front`` subclass:

1. They need to be declared in the subclass definition using a special syntax shown below.
2. They are all typed using data types defined in the `Python ctypes library <https://docs.python.org/3/library/ctypes.html>`_ or as NeuroDevSim classes. Trying to set an attribute to a value of a different type will throw an error.
3. They are all fixed size: no dictionaries, lists, strings,...
4. They will be present in every instance of the subclass, instance specific attributes are not supported (see the :ref:`dos_donts-label`). So even if the attribute is functionally needed in only one or a few instances of the subclass, it will be present in all of them. Consider carefully whether an extra attribute is really needed.

The syntax to specify additional attributes is unusual, it is specific to how Python supports memory shared arrays. They are defined as a list of tuples, with each tuple containing the attribute name and and a ``ctypes`` data type::

    [('attribute1_name', ctypes_data_type), ('attribute2_name', ctypes_data_type)]

For example, to define an int attribute *foo*::

    class MyFront(Front):
        _fields_ = Front._fields_ + [('foo', c_int)]

and a second float attribute *bar*::

    class MyFront(Front):
        _fields_ = Front._fields_ + [('foo', c_int), ('bar', c_double)]

Different from standard Python behavior, it is important to respect the types of the additional attributes. The following code will cause a 'TypeError: int expected instead of float' because *foo* has been declared to be an integer::

    class MyFront(Front):
        _fields_ = Front._fields_ + [('foo', c_int)]

        def manage_front(self,constellation):
            ...
            # this will cause an error because 1.5 is a float
            self.foo = 1.5
            # this is fine
            self.foo = int(1.5)

The following `ctypes <https://docs.python.org/3/library/ctypes.html>`_ are imported in NeuroDevSim and can be used in attribute definitions:

=========== ========================
ctypes type Python type
=========== ========================
c_bool      bool
c_char      1-character bytes object
c_short     int
c_int       int
c_long      int
c_double    float
=========== ========================

The differences between c_short, c_int and c_long is in the number of bytes used (2, 4, 8 bytes on 64 bit operating systems) and the corresponding range of numbers encoded (−32,768 through 32,767; −2,147,483,648 through 2,147,483,647 and -9,223,372,036,854,775,808 through 9,223,372,036,854,775,807). Additional ``ctypes`` data types exist and can be imported by knowledgeable users.

In addition, one can also use a *class* type, which has been defined elsewhere. At present we recommend using only the predefined ``Point`` class or the ``ID`` class, which is used to identify ``Fronts`` and ``Substrate``::

    class MyFront(Front):
        _fields_ = Front._fields_ + [('friend', ID)]

        def manage_front(self,constellation):
            ...
            # get a list of fronts with neuron_name beginning with "friend"
            my_friends = self.get_fronts(constellation,what='name',name="friend",returnID=True)
            # store the first front returned as an ID
            self.friend = my_friends[0][0]

Using ``Fronts`` or ``Substrate`` as an additional attribute is not recommended because it makes a copy of the original instance and this copy will not be updated. Moreover, for ``Front`` the specific subclass would have to be specified.

Attribute initialization
------------------------
Additional attributes are automatically initialized to a zero value, in the case of a Structure all its fields are set to zero. An additional attribute can be given a starting value immediately after its instantiation by ``add_child``::

    class MyFront(Front):
        _fields_ = Front._fields_ + [('foo', c_int)]

        def manage_front(self,constellation):
            ...
            new_front = self.add_child(constellation,new_pos)
            new_front.foo = 7
            ...
