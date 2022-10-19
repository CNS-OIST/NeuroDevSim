####################################################################################
#
#    NeuroDevSim: Neural Development Simulator
#    Copyright (C) 2019-2022 Okinawa Institute of Science and Technology Graduate
#    University, Japan.
#
#    See the file AUTHORS for details.
#    This file is part of NeuroDevSim. It contains all simulator objects
#    and methods.
#
#    NeuroDevSim is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License version 3,
#    as published by the Free Software Foundation.
#
#    NeuroDevSim is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#################################################################################

from multiprocessing import Process,set_start_method
from multiprocessing.sharedctypes import RawArray,RawValue
from ctypes import Structure, c_bool, c_char, c_double, c_short, c_int, \
            c_long, create_string_buffer, create_unicode_buffer, sizeof
import numpy as np
from scipy.special import erfc
from scipy.stats import binom
from math import sqrt, exp, ceil, degrees, radians, sin, cos, acos, atan2
from scipy.linalg import expm, norm
import os, sys, copy, time, psutil
from operator import itemgetter
import sqlite3
from colorama import Fore
import shutil

# constants
PAUSE = 0.00005 # time.sleep for processes that have to wait, to prevent CPU running hot
LOCKPAUSE = 0.000001 # time.sleep to check lock
NNAMELENGTH = 40     # max number of characters in a neuron_name string
BNAMELENGTH = 20     # max number of characters in a branch_name string
ARCLENGTH = 30      # max number of Points in an arc
MAXTRIALS = 100     # max number of position tests for placing neuron somata

### Classes and methods: public
def nds_version(raw=False):
    """ returns version of NeuroDevSim as a string (default) or a number which equals 100 times the version number.
    
    Parameters
    ----------
    Optional:
    raw : boolean : return a number instead of a string, default: False.

    Returns
    -------
    version : string or float
    """
    if raw:
        return 101.
    else:
        return "NeuroDevSim 1.0.1"
        
class nds_list(list):
    """ Class implementing a list that prints NeuroDevSim objects inside the list nicely. Otherwise acts as a standard list.
    """
    def __str__(self):
        text = '[' # will be returned
        ni = 0   # count number of items on a line
        for item in self:
            if isinstance(item,tuple): # nds_list from get_substrates or get_fronts
                new = "(" + str(item[0]) + ", " + str(item[1]) + ")"
            else:
                new = str(item)
            if (ni == 0) or ((len(text) + len(new)) < 80): # keep same line
                if ni == 0:
                    text += new
                else:
                    text += ", " + new
                ni += 1
            else: # start new line
                text += ',\n'
                ni = 1
                text += " " + new
        text += ']'
        return text

# A 3D point
class Point(Structure):
    """ Class implementing 3D coordinates and vectors.
    
    Attributes:
        x : float : width position in µm.
        y : float : depth position in µm.
        z : float : height position in µm.
    """
    _fields_ = [('x', c_double), ('y', c_double), ('z', c_double)]
    
    def __str__(self):
        return "[{:4.2f}".format(self.x) + ", {:4.2f}".format(self.y) + ", {:4.2f}".format(self.z) + "]"

    # adding two Points
    def __add__(self, other):
        if isinstance(other,Point):
            return Point(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other,list):
            return Point(self.x + other[0], self.y + other[1], self.z + other[2])
        elif isinstance(other,np.ndarray):
            return Point(self.x + other[0], self.y + other[1], self.z + other[2])
        else:
            print (Fore.RED + "Points add Error" + Fore.RESET)

    # subtract two Points
    def __sub__(self, other):
        if isinstance(other,Point):
            return Point(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other,list):
            return Point(self.x - other[0], self.y - other[1], self.z - other[2])
        elif isinstance(other,np.ndarray):
            return Point(self.x - other[0], self.y - other[1], self.z - other[2])
        else:
            print (Fore.RED + "Points sub Error" + Fore.RESET)

    # multiply Point with float
    def __mul__(self, other):
        if not isinstance(other,float) and not isinstance(other,int):
            print (Fore.RED + "Points mul Error" + Fore.RESET)
            return
        return Point(self.x * other, self.y * other, self.z * other)

    # divide Point with float
    def __truediv__(self, other):
        if not isinstance(other,float) and not isinstance(other,int):
            print (Fore.RED + "Points div Error" + Fore.RESET)
            return
        return Point(self.x / other, self.y / other, self.z / other)

    # equality of two Points
    def __eq__(self, other):
        if not isinstance(other,Point):
            raise TypeError("other not a Point")
        return self.x == other.x and self.y == other.y and self.z == other.z

    # non equality of two Points
    def __ne__(self, other):
        if not isinstance(other,Point):
            raise TypeError("other not a Point")
        return self.x != other.x or self.y != other.y or self.z != other.z
        
    def nds_class(self):
        """ Return class name.
         
        Returns
        -------
        Class name "Point" : string
        """
        return "Point"

    # compute length
    def length(self):
        """ Compute length of a vector.
         
        Returns
        -------
        length of vector : float
        """
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    # normalize length
    def norm(self):
        """ Normalize vector to unit vector.
         
        Returns
        -------
        vector : Point
        """
        length = self.length()
        if length > 0.:
            return self / length
        else:
            return self

    # dot operation on 1D arrays results in float result
    def dot(self, other):
        """ Perform vector dot operation between two vectors.

        Parameters
        ----------
        other : Point.

        Returns
        -------
        dot product of vectors : float
        """
        if not isinstance(other,Point):
            print ("Points dot Error")
            return
        return (self.x * other.x + self.y * other.y + self.z * other.z)
        
    # cross operation between 1D arrays, assumes the Points are vectors
    #   relative to (0.,0.,0.)
    def cross(self, other):
        """ Perform vector cross operation between two 1D vectors.

        Parameters
        ----------
        other : ``Point``.

        Returns
        -------
        cross product of vectors : Point
        """
        if not isinstance(other,Point):
            print ("Points cross Error")
            return
        return Point((self.y * other.z) - (self.z * other.y), \
                     (self.z * other.x) - (self.x * other.z), \
                     (self.x * other.y) - (self.y * other.x))

    # turn into numpy array
    def nparray(self):
        """ Turn vector into numpy.array.
         
        Returns
        -------
        vector : numpy.array
        """
        return np.array([self.x, self.y, self.z])

    # returns new point with all coordinates positive
    def abs(self):
        """ Make all attributes of ``Point`` positive.
         
        Returns
        -------
        vector : Point
        """
        return Point(abs(self.x), abs(self.y), abs(self.z))
    
    # randomly change point
    def wiggle(self,constellation,scale,min_size=0.5):
        """ Randomly change position of a point so it is still inside the simulation volume.
         
        Parameters
        ----------
        constellation : ``Constellation`` object.
        scale : float : maximum amount of change applied (summed over all coordinates), should be larger than *min_size*.
        Optional:
        min_size : float : miminum amount of change applied (summed over all coordinates), default 0.5 µm.

        Returns
        -------
        vector : Point
        """
        if scale <= min_size:
            raise ValueError("scale","larger than " + str(min_size))
            return self
        count = 0
        while count < 100: # no endless loop
            # generate random coordinate in range -level to +level
            noise = (2 * scale) * np.random.random(3) - scale
            # impose minimum size on noise
            if norm(noise) < min_size:
                continue
            point = self + noise
            # check whether random point is inside simulation volume
            if point.out_volume(constellation) == 0.:
                return point
            count += 1
        raise BugError("wiggle","unsuccessful")

    # is point otside simulation volume?
    # can pass either full constellation or volume
    # returns: 0. or offending coordinate
    def out_volume(self,constellation):
        """ Returns whether the ``Point`` is outside the simulation volume (including its borders).
        
        If inside volume returns 0., otherwise it returns first coordinate outside the volume.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.

        Returns
        -------
        is outside simulation volume : float
        """
        sim_volume = constellation.sim_volume
        if self.x < sim_volume[0][0]:
            return self.x
        if self.y < sim_volume[0][1]:
            return self.y
        if self.z < sim_volume[0][2]:
            return self.z
        if self.x > sim_volume[1][0]:
            return self.x
        if self.y > sim_volume[1][1]:
            return self.y
        if self.z > sim_volume[1][2]:
            return self.z
        return 0.

    def toPol_point(self):
        """ Convert Point to a Pol_point.
        
        Returns
        -------
        polar point : Pol_point
        """
        r = sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
        return Pol_point(r,acos(self.z / r) * 57.295779513082321, \
                         atan2(self.y,self.x) * 57.295779513082321)

    def is_cylinder(self):
        """ Compatibility with other Structures.
        
        Returns
        -------
        always False : boolean
        """
        return False

    # return closest point in grid
    # grid[0] is not used
    # self should be inside simulation volume
    def _grid(self,volume):
        sim_volume = volume.sim_volume
        index = 1 + int((round((self.x - sim_volume[0][0]) * \
                            volume.grid_div) * volume.grid_size2) + \
                (round((self.y - sim_volume[0][1]) * \
                        volume.grid_div) * volume.grid_size1) + \
                round((self.z - sim_volume[0][2]) * volume.grid_div))
        if (index < 1) or (index >= volume.grid_max):
            # outside volume
            return -1
        return index
        
# list encoding dimensions as unit Point
dimensions = [Point(1.,0.,0.),Point(0.,1.,0.),Point(0.,0.,1.)]

def random_point(min_coord,max_coord):
    """ Returns a random point with each coordinate sampled from the uniform distribution in range [min_coord, max_coord).

    Parameters
    ----------
    min_coord: float : minimum value for each random coordinate
    max_coord: float : maximum value for each random coordinate
    
    Returns
    -------
    random point : Point
    """
    range = max_coord - min_coord
    rnp = np.random.uniform(min_coord,max_coord,size=3)
    return Point(rnp[0], rnp[1], rnp[2])

def unit_sample_on_sphere():
    """ Returns a random point on a unit sphere around the zero point.

    Returns
    -------
    random unit-length vector : Point
    """
    xs = np.random.normal(size=3)
    den = norm(xs)
    return Point(xs[0]/den, xs[1]/den, xs[2]/den)

def unit_sample_on_circle(axis = 2):
    """ Returns a random point on a circle around the zero point.
    
    Default is a horizontal circle but another plane can be selected with the *axis* parameter.

    Parameters
    ----------
    Optional :
    axis : integer : select plane: 0 is yz plane, 1 is xz plane, 2 is zy plane.
    
    Returns
    -------
    random unit-length vector in a plane: Point
    """
    xs = np.random.normal(size=3)
    xs[axis] = 0. # set one dimension to zero before normalizing random vector
    den = norm(xs)
    return Point(xs[0]/den, xs[1]/den, xs[2]/den)

def unit_sample_on_cone(angle1,angle2=None):
    """ Returns a random point in a vertical cone within two angles relative to vertical: range angle1-angle2, where angle2 >= angle1.
    
    If only angle1 is given it applies a 0-angle1 range.
    
    Parameters
    ----------
    angle1 : float : angle in degrees
    Optional :
    angle2 : float : angle in degrees

    Returns
    -------
    unit-length vector within two angles : Point
    """
    # np.pi/180 == 0.017453292519943
    cos_angle1 = cos(angle1 * 0.017453292519943)
    if angle2 and (angle2 >= angle1):
        cos_angle2 = cos(angle2 * 0.017453292519943)
        z = np.random.random() * (cos_angle2 - cos_angle1) + cos_angle1
    else:
        z = np.random.random() * (1.0 - cos_angle1) + cos_angle1
    phi = np.random.random() * 6.283185307179586 # 2 * pi
    z_sqrt = sqrt(1.0 - z**2)
    x = z_sqrt * cos(phi)
    y = z_sqrt * sin(phi)
    return Point(x,y,z)

def angle_two_dirs(dir1,dir2,degree=True):
    """ Returns small angle in degrees between two directions relative to [0.0, 0.0, 0.0]
    
    Parameters
    ----------
    dir1 : ``Point``
    dir2 : ``Point``
    Optional:
    degree: return angle in degrees (True) or radians (False), default True.

    Returns
    -------
    angle in degrees or radians : float
    """
    l1 = dir1.length()
    l2 = dir2.length()
    if (l1 == 0.) or (l2 == 0.):
        print (Fore.MAGENTA + "Warning: angle_two_dirs with zero point",dir1,dir2,Fore.RESET)
        return 0.
    arcos = dir1.dot(dir2)/(l1 * l2)
    if arcos > 1.0: # probably round up error
        return 0.
    angle = np.arccos(arcos)
    if degree:
        # 180/np.pi == 57.29577951308232
        return degrees(angle)
    else:
        return angle

def angle_sub_divs(angle,radius,min_distance,verbose):
    """ Computes how may times an arc around center zero of size *angle* with radius *length* should be subdivided so that the line connecting the subdivision does not intersect with a center zero sphere of min_radius.
    
    Mathematically this means: for a isosceles triangle with *angle* and the isoscele equal to *radius* should be subdivided so that the perpendicular bisector of the short line connecting two neighboring points on the isosceles triangle with subdivided angle has a min_distance length.
    
    Parameters
    ----------
    angle : float > 0. and <= 90. : angle in degrees
    radius : float > 0. : radius of the arc with angle
    min_distance : float > 0. and < radius : minimum distance of connecting line

    Returns
    -------
    number of divisions or zero in case of error : integer
    """
    if angle <= 0.:
        if verbose >= 2:
            print (Fore.MAGENTA + "Warning: angle_sub_divs with wrong angle",angle,Fore.RESET)
        return 0
    # only works for angles up to 90 deg -> reduce to such angle and multiply number of divisions correspondingly
    if angle > 90.:
        if angle % 90. > 0.:
            mul = 1 + int(angle / 90.)
        else:
            mul = int(angle / 90.)
        angle = angle / mul
    else:
        mul = 1
    if (radius < 0.) or (min_distance < 0.) or (min_distance >= radius):
        if verbose >= 2:
            print (Fore.MAGENTA + "Warning: angle_sub_divs with bad radius or min_distance",radius,min_distance,Fore.RESET)
        return 0
    rangle = angle / 57.29577951308232 # in radians
    for n in range(1,50):
        beta = rangle / (2 * n)
        # compute side A of triangle with
        #    gamma = right angle, angle beta and side C = radius:
        #  alpha = 1.570796326794897 - beta    from 180 - 90 - beta
        #  A = C * sin(alpha) / sin(gamma)    with sin(gamma) = 1 -> A = C * sin(alpha)
        if radius * sin(1.570796326794897 - beta) > min_distance:
            return mul * n
    if verbose >= 2:
        print (Fore.MAGENTA + "Warning: angle_sub_divs not found",angle,radius,min_distance,Fore.RESET)
    return 0

# make point from coordinates given as a list
def _point_list(coord):
    return Point(coord[0],coord[1],coord[2])

# A 3D polar coordinate point
class Pol_point(Structure):
    """ Class implementing 3D polar coordinates.
    
    Attributes:
        r : float > 0.: radial position in µm.
        theta : float 0. - 180. : angle in degrees, constrained to range by initialization.
        phi : float 0. < 360 : angle in degrees, constrained to range by initialization.
    """
    _fields_ = [('r', c_double), ('theta', c_double), ('phi', c_double)]
    
    def __init__(self,r,theta,phi):
        self.r = abs(r)
        # constrain theta to range 0 <= 180, wrap around
        if theta < 0.:
            theta = - theta
        if theta > 180.:
            while theta > 360.:
                thetha -= 360.
            if theta > 180.:
                theta = 360. - theta
        self.theta = theta
        # constrain phi to range 0 < 360
        while phi < 0.:
            phi += 360.
        while phi >= 360:
            phi -= 360.
        self.phi = phi
    
    def __str__(self):
        return "[{:4.2f}".format(self.r) + ", {:4.2f}".format(self.theta) + ", {:4.2f}".format(self.phi) + "]"

    def nds_class(self):
        """ Return class name.
         
        Returns
        -------
        Class name "Pol_point" : string
        """
        return "Pol_point"

    def toPoint(self):
        """ Convert Pol_point to a Point.
        
        Returns
        -------
        cartesian point : Point
        """
        rtheta   = self.theta * 0.017453292519943 # to radian pi/180
        rphi     = self.phi * 0.017453292519943
        return Point(self.r* sin(rtheta) * cos(rphi),self.r* sin(rtheta) * sin(rphi),\
                     self.r * cos(rtheta))

# do not change order of this list: rest of the code assumes fixed swc_type values
swc_types = ["undefined","soma","axon","(basal) dendrite","apical dendrite",\
             "custom","neurite","glial process","oblique dendrite","tuft dendrite",\
             "smooth dendrite","spiny dendrite","filipodium","spine","synaptic bouton",\
             "reserved1","reserved2","reserved3","reserved4","reserved5"]

def swc_name(swc):
    """ Returns string for swc type
    
    Parameters
    ----------
    swc : integer : swc type value

    Returns
    -------
    swc name : string
    """
    if (swc >=0) and (swc < len(swc_types)):
        return swc_types[swc]
    else:
        return ("undefined")

# front definition
# Private Attributes:
#   _nid: first index into self._fronts array
#   _fid: second index into self._fronts array: front is stored in self._fronts[nid][fid]
#   _sid: soma as second index into self._fronts array
#   _pid: parent as second index into self._fronts array
#   _cid: child: is a second index into self._fronts array if num_children == 1 or
#         index into children if num_children > 1
class Front(Structure):
    """ Main class implementing growing/migrating agents.
    
    This class is not meant to be instantiated, instead subclass it with a model specific *extend_front* method.
    
    Only the **attributes** listed below can be read. NEVER change these attributes.
    
    Attributes:
        birth : integer : cycle when front was created.
        death : integer : cycle when front was retracted, default: -1 (not retracted).
        end : ``Point`` : specifying end coordinate of cylinder.
        order : integer : centripetal order of branching, soma has order 0.
        orig : ``Point`` : origin coordinate of cylinder or center of sphere.
        num_children : integer : number of children in the tree structure.
        path_length : float : distance in µm to soma center along the tree structure.
        radius : float : radius in µm of the cylinder or sphere.
        swc_type : integer : SWC_type as specified in Cannon et al. (optional).
    """
    # private fields:
    # _nid: first index in fronts array, same for all fronts of a neuron
    # _fid: second index in fronts array, unique to each front of a neuron
    # _sid: soma second index in fronts array, same for all fronts of a neuron
    #       for soma index into neurons array (_neid)
    # _pid: parent of this front: second index in fronts array
    # _cid: depends on num_children: 0: _cid is 0
    #                                1: _cid is child second index in fronts array
    #                                >1: _cid is index into children array
    # _gid: index into gids array, negative: gids locked by Admin
    # _dbid : id used by front_data in database, used for fast database updating
    # _flags: bit-wise flags, see _flag encoding comments
    
    _fields_ = [('orig', Point), ('end', Point), ('name', c_char * BNAMELENGTH), \
                ('radius', c_double), ('path_length', c_double), \
                ('num_children',c_short), ('swc_type', c_short), \
                ('order', c_short), ('birth', c_short), ('death', c_short), \
                ('_nid', c_short), ('_fid', c_int),  ('_sid', c_int), \
                ('_pid', c_int), ('_cid', c_int), ('_gid', c_int), \
                ('_dbid', c_int), ('_flags', c_short)]
                
    # Fronts and subclasses do not use the __init__ method: an existing
    #   self._fronts shared array entry is updated by Constellation._enter_front
    
    def __str__(self):
        if self.is_migrating():
            type = "Migrating front "
        elif self.is_growing():
            type = "Growing front "
        elif self.is_active():
            type = "Active front "
        elif self.is_retracted():
            type = "Retracted front (at " + str(self.death) + ") "
        else:
            type = "Inactive front "
        text = type + str(self._fid) + " (" + str(self._nid) + "): "
        if self.is_cylinder():
            text = text +  str(self.orig) + " " + str(self.end)
        else:
            if self.is_migrating():
                text = text +  "migrating sphere " + str(self.orig)
            else:
                text = text +  "sphere " + str(self.orig)
        text = text + " radius: {:4.2f}".format(self.radius) + \
                " path_length: {:4.2f}".format(self.path_length) + \
                " swc: " + str(self.swc_type) + " order: " + str(self.order)
        if self._pid > 0:
            text = text + ", parent: " + str(self._pid) + " with " + \
                str(self.num_children) + " children, soma " + str(self._sid)
        else:
            text = text + ", no parent with " + str(self.num_children) +\
                " children"
        return text + ", made at " + str(self.birth) + \
                " (" + str(self._flags) + " " + str(self._gid) + ")"
    
    # equality of two Fronts:
    def __eq__(self, other):
        if not isinstance(other,Front):
            if other == None: # self is always a Front in this method
                return False
            raise TypeError("other not a Front")
        return self._nid == other._nid and self._fid == other._fid

    # non equality of two Fronts
    def __ne__(self, other):
        if not isinstance(other,Front):
            if other == None: # self is always a Front in this method
                return True
            raise TypeError("other not a Front")
            return
        return self._nid != other._nid or self._fid != other._fid

    # turn FrontId into unique string
    def _key(self):
        return str(self._nid) + "_" + str(self._fid)

    # compute point of surface of spherical parent as origin of cylinder
    def _sphere_surf(self,new_pos,offset=0.0):
        vec = new_pos - self.orig
        surf_pos = self.orig + (vec.norm() * (self.radius + offset))
        return surf_pos
    
    # for debugging: print child structure
    def _print_children(self,constellation):
        print ("num_children:",self.num_children,self._cid)
        if self.num_children == 0:
            return
        if self.num_children == 1:
            print ("  child 1 :",self._cid,"( swc",constellation._fronts[self._nid][self._cid].swc_type,") in parent")
            return
        num = 1
        child_link = self._cid
        while True:
            child = constellation._children[child_link]
            print ("  child",num,":",child._cid,"( swc",constellation._fronts[self._nid][child._cid].swc_type,") in children",child_link)
            child_link = child.next
            if child_link == 0:
                break

    def nds_class(self):
        """ Return class name.
         
        Returns
        -------
        Class name of Front subclass : string
        """
        name = str(self.__class__)
        return _strip_neuron_type(name)

    def manage_front(self,constellation):
        """ Main front method.
        
        Must be replaced by a method in the subclass, not doing so generates an error.
        This method is called each cycle for active fronts so that different growth, migration, etc. methods can be used.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        """
        raise BugError("manage_front","not defined for" + str(self))
  
    def add_child(self,constellation,coordinate,radius=0.,branch_name="",\
                    swc_type=0,cylinder=True):
        """ Main growth method. Instantiate a new ``Front`` of the same subclass as *self* that becomes a child of *self*.
            
        By default a cylindrical front with *orig* equal to *self.end* (if *self* is also cylindrical, otherwise point on spherical surface) and *end* equal to *coordinate* is created. If ``cylinder==False`` a spherical front with *orig* equal to *coordinate* is created.
        Collision and inside simulation volume checks are performed, if these fail a CollisionError, InsideParentError, VolumeError or GridCompetitionError exception is thrown.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        coordinate : ``Point`` : *end* (cylinder, default) or *orig* (sphere) coordinate of new child front.
        Optional:
        branch_name : string of maximum 20 characters : optional branch_name, default: None.
        cylinder : boolean : make cylindrical (True) or spherical child (False), default True.
        radius : float : radius in µm of the cylinder or sphere, default same radius as *self*.
        swc_type : integer : SWC_type as specified in Cannon et al., default same as *self* or 3 (dendrite) if *self* is a soma.

        Returns
        -------
        the new child or an exception : Front subclass or CollisionError, GridCompetitionError, InsideParentError, NotSelfError, ValueError or VolumeError
        """
        if constellation.verbose >= 3:
            print (constellation.my_id,"add_child",self._fid,coordinate)
        # check for errors
        #   check whether self is calling this
        if constellation._automatic and (self != constellation._manage):
            #print (constellation.my_id,self._fid,"add_child NotSelfError")
            raise NotSelfError("add_child")
        #   check whether coordinate inside simulation volume
        result = coordinate.out_volume(constellation)
        if result != 0.:
            #print (constellation.my_id,self._fid,"add_child VolumeError")
            raise VolumeError(result)
        #   check whether coordinate inside parent
        if self._point_in_self(coordinate):
            #print ("add_child InsideParentError",self, coordinate)
            #print (constellation.my_id,self._fid,"add_child InsideParentError")
            raise InsideParentError()
        if constellation._automatic and (constellation.verbose > 1) and \
                    (not self.is_growing()) and (not self.is_migrating()):
            if constellation.verbose >= 2:
                print (Fore.MAGENTA + "Warning: add_child called by non-growing front, this makes GridCompetitionErrors more likely:",self,constellation.my_id,coordinate,Fore.RESET)
        # compute radius of front
        if radius == 0.: # use parent radius
            rad = self.radius
        else:
            rad = radius
        #   check whether valid point: will raise CollisionError if not
        if cylinder:
            # compute radius for collision checking
            if (swc_type == 12) or ((swc_type == 0) and (self.swc_type == 12)):
                # filipodium, check whether it belongs to migrating soma
                soma = self.get_soma(constellation) # self could be the soma
                if soma.has_migrated(): # yes -> use its radius for collision check
                    min_rad = soma.radius
                else:
                    min_rad = rad
            else:
                min_rad = rad
            gids = constellation._get_gids(self.end,coordinate,0.)
            constellation._test_collision(self.end,coordinate,min_rad,self,gids)
        else:
            gids = constellation._get_gids(coordinate,None,rad)
            constellation._test_collision(None,coordinate,rad,self,gids)
        # enter front into the shared arrays and communicate to Admin
        child = constellation._enter_front(self,cylinder,coordinate,rad,\
                                                branch_name,swc_type)
        # update the grid: this has been locked by _test_collision
        id = child.get_id()
        my_id = constellation.my_id
        for gid in gids:
            constellation._grid_set(gid,id)
            # unlock it after use
            constellation._grid_wlock[gid] = 0
        child._store_gids(constellation,gids)
        # update parent
        constellation._add_child(self,child._fid)
        self._clear_child_retracted()
        constellation._newfs.append(child) # store for this manage_front call
        # increase order if needed in all children that were generated this cycle
        if (child.order == self.order) and (self.num_children > 1): # branch point
            new_order = self.order + 1
            # possibly iterative upgade of all children
            self._update_order(constellation,new_order)
        if constellation.verbose >= 6:
            print (constellation.my_id,"add_child success",self._fid,swc_type,coordinate)
        return child
        
    def add_branch(self,constellation,coord_list,radius=0.,branch_name="",\
                    swc_type=0,enable_all=False):
        """ Additional growth method. Similar to ``add_child`` but multiple cylindrical ``Front`` are made as an unbranched branch with as a root a child of *self*.

        Collision and inside simulation volume checks are performed, if these fail on the first point a CollisionError, InsideParentError, VolumeError or GridCompetitionError exception is thrown. For collision errors on later points no error is thrown, instead ``add_branch`` tries to solve GridCompetitionErrors by retrying or returns with less fronts made than points provided.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        coord_list : [``Point``,] : *end* of consecutive child fronts.
        Optional:
        branch_name : string of maximum 20 characters : optional branch_name, default: None.
        enable_all : boolean : all new ``Front`` made are enabled, otherwise only the terminal child is enabled and the rest are disabled, default: False.
        radius : float : radius in µm of the cylinder or sphere, default same radius as *self*.
        swc_type : integer : SWC_type as specified in Cannon et al., default same as *self* or 3 (dendrite) if *self* is a soma.

        Returns
        -------
        the new fronts as nds_list or an exception : [Front subclass,] or CollisionError, GridCompetitionError, InsideParentError, NotSelfError, ValueError or VolumeError
        """
        if constellation.verbose >= 3:
            print (constellation.my_id,"add_branch",self._fid,self.end,":",nds_list(coord_list),enable_all)
        # check for errors
        #   check whether self is calling this
        if constellation._automatic and (self != constellation._manage):
            raise NotSelfError("add_branch")
        if constellation._automatic and (constellation.verbose > 1) and \
                    (not self.is_growing()) and (not self.is_migrating()):
            if constellation.verbose >= 2:
                print (Fore.MAGENTA + "Warning: add_branch called by non-growing front, this makes GridCompetitionErrors more likely:",self,Fore.RESET)
        # compute radius of front
        if radius == 0.: # use parent radius
            rad = self.radius
        else:
            rad = radius
        # compute radius used for collision checking
        if (swc_type == 12) or ((swc_type == 0) and (self.swc_type == 12)):
            # filipodium, check whether it belongs to migrating soma
            soma = self.get_soma(constellation) # self could be the soma
            if soma.has_migrated(): # yes -> use its radius for collision check
                min_rad = soma.radius
            else:
                min_rad = rad
        else:
            min_rad = rad
        # is this an arc?
        arc_used = None
        key = self._key()
        if key in constellation._used_arcs: # check for used arc
            for index in constellation._used_arcs[key]:
                arc = constellation._arcs[index]
                if constellation._arc_points[arc.index] == coord_list[0]:
                    arc_used = arc # assumes points list was not changed
        elif self.is_arc(): # check for arc continuations
            index = constellation._get_arc(self)
            if index >= 0:
                arc_used = constellation._arcs[index]
                if constellation._arc_points[arc_used.index+arc_used.next_point] \
                        != coord_list[0]:
                    raise BugError("add_branch","wrong continuation arc" + str(arc_used))
                    arc_used = None
        if constellation.verbose > 4:
            if not arc_used:
                print (Fore.RED,constellation.my_id,"add_branch no arc found",self._fid,self.end,nds_list(coord_list),Fore.RESET)
        new_fronts = nds_list() # list of all new fronts made
        parent = self
        my_id = constellation.my_id
        # now cycle through all the coordinates in coord_list
        for coordinate in coord_list:
            error = None
            #   check whether coordinate inside simulation volume
            result = coordinate.out_volume(constellation)
            if result != 0.:
                if not new_fronts: # no fronts made yet -> return to user
                    raise VolumeError(result)
                else: # fronts made, exit 'normally'
                    break # out for coordinate
            #   check whether coordinate inside parent
            if parent._point_in_self(coordinate):
                if not new_fronts: # no fronts made yet -> return to user
                    raise InsideParentError()
                else: # fronts made, exit 'normally'
                    break # out for coordinate
            gids = constellation._get_gids(parent.end,coordinate,0.)
            if new_fronts: # previous fronts made -> do complete error checking 
                count = 0
                while True: # loop for GridCompetitionError
                    count += 1
                    if count > 100:
                        raise GridCompetitionError(cgid)
                    try: # first test for collision of soma
                        constellation._test_collision(parent.end,coordinate,\
                                                         min_rad,self,gids)
                        failed = False
                        break # out of while True
                    except GridCompetitionError as error:
                        cgid = error.gid
                        time.sleep(2*LOCKPAUSE) # competition with other processor
                        continue # while True
                    except CollisionError as error:
                        if error.only_first:
                            col_front = error.collider
                        else:
                            col_front = error.collider[0]
                        failed = True # do not make this point
                        break # out of while True
                    except Exception as error: # other, usually fatal exceptions
                        print ("add_branch unexpected",error)
                        failed = True # do not make this point
                        break # out of while True
                if failed:
                    break # for coordinate in coord_list
            else: # first front: let the user handle this
                if constellation.verbose >= 6:
                    print (constellation.my_id,"add_branch raw _test_collision")
                constellation._test_collision(parent.end,coordinate,min_rad,\
                                                                self,gids)
            #print (constellation.my_id,self._fid,swc_type,"add_child locked",gids)
            # enter front into the shared arrays and communicate to Admin
            child = constellation._enter_front(parent,True,coordinate,rad,\
                                                    branch_name,swc_type)
            if constellation.verbose >= 6:
                print (constellation.my_id,"add_branch made",child)
            # update the grid: this has been locked by _test_collision
            id = child.get_id()
            for gid in gids:
                constellation._grid_set(gid,id)
                # unlock it after use
                constellation._grid_wlock[gid] = 0
            child._store_gids(constellation,gids)
            # update parent
            if parent != self:
                constellation.lock(parent)
            else:
                self._clear_child_retracted()
            constellation._add_child(parent,child._fid)
            if parent != self:
                constellation.unlock(parent)
            if arc_used: # update arc info
                arc_used.next_point += 1
                if arc_used.next_point == arc_used.num_points: # finished
                    arc_used.complete = constellation.cycle
                    arc_used = None # rest of points do not belong to arc
            # update data structures
            new_fronts.append(child)
            constellation._newfs.append(child) # store for this manage_front call
            # increase order if needed in all children that were generated this cycle
            if (parent == self) and (child.order == self.order) and \
                                    (self.num_children > 1): # branch point
                new_order = self.order + 1
                # possibly iterative upgade of all children
                self._update_order(constellation,new_order)
            parent = child
        if new_fronts and (not enable_all): # keep only last one active
            for f in new_fronts[:-1]: # make inactive
                f._clear_active()
                f.clear_growing()
                if constellation.verbose >= 6:
                    print (constellation.my_id,"add_branch inactivated",f)
        if arc_used and (len(new_fronts) == len(coord_list)): 
            # complete requested arc made -> mark as arc (do not mark incomplete arcs)
            for f in new_fronts:
                f._set_arc()
            arc_used.last_fid = new_fronts[-1]._fid
        if constellation.verbose >= 6:
            print (constellation.my_id,"add_branch success",self._fid,swc_type,len(new_fronts))
        return new_fronts
    
    def migrate_soma(self,constellation,coordinate,filipod=False,\
                     trailing_axon=False):
        """ Growth method. Migrate *self*, a soma, to a new position.

        By default migration to *coordinate* and no children allowed.
        Alternative is to migrate along a filipod with `filipod==True`, replacing the single child filipod (swc_type 12) front (*coordinate* ignored). If `trailing_axon==True` an axon front is made to connect with single axon child (swc type 2). Maximum number of children allowed is two (a filipod and an axon).
        Collision and inside simulation volume checks are performed, if these fail a CollisionError, InsideParentError or VolumeError exception is thrown. Additional possible errors: BadChildError, NotSomaError.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        coordinate : ``Point`` : position to which soma migrates. Ignored if `filipod==True`.
        Optional:
        filipod : boolean : replace filipod child by migrating to a position where *filipod.end* falls on surface of soma, default False.
        trailing_axon : boolean : make a new axon front to connect with existing axon child, default False.

        Returns
        -------
        the new position or an exception : Point or BadChildError, NotSelfError, NotSomaError, CollisionError, GridCompetitionError, VolumeError
        """
        if constellation.verbose >= 3:
            print (constellation.my_id,"migrate_soma",self._fid,self.orig)
        # check for errors
        #   check whether self is calling this
        if constellation._automatic and (self != constellation._manage):
            raise NotSelfError("migrate_soma")
        #   check whether self is a soma
        if self.order != 0:
            raise NotSomaError()
        #   check whether coordinate inside simulation volume
        if not filipod:
            result = coordinate.out_volume(constellation)
            if result != 0.:
                raise VolumeError(result)
        if constellation._automatic and (constellation.verbose > 1) and \
                                            (not self.is_migrating()):
            if constellation.verbose >= 2:
                print (Fore.MAGENTA + "Warning: migrate_soma called by non-migrating front, this makes GridCompetitionErrors more likely:",self,Fore.RESET)
        fili = None # filipod front
        axon = None # axon front
        #   check whether correct number of children
        if not filipod and not trailing_axon:
            if self.num_children > 0:
                raise BadChildError("No children allowed")
        else:
            children = self.get_children(constellation)
            found_fili = False
            found_axon = False
            for child in children:
                if filipod and (child.swc_type == 12):
                    if not found_fili:
                        found_fili = True
                        fili = child
                        continue
                    else:
                        raise BadChildError("Two filipod children for " + str(self))
                if trailing_axon and (child.swc_type == 2):
                    if not found_axon:
                        found_axon = True
                        axon = child
                        continue
                    else:
                        raise BadChildError("Two axon children for " + str(self))
                raise BadChildError("Inappropriate child " + str(child) + " for " + str(self))
            if filipod:
                if not found_fili:
                    raise BadChildError("No filipod child for filipod==True")
                elif not fili.is_cylinder():
                    raise BadChildError("Filipod child must be cylinder")
                elif fili.is_active():
                    raise ActiveChildError()
            if trailing_axon:
                if not found_axon:
                    raise BadChildError("No axon child for trailing_axon==True")
                elif not axon.is_cylinder():
                    raise BadChildError("Axon child must be cylinder")
        # prepare for future locking
        id = self.get_id()
        add_gids = [] # keep this list
        l_add_gids = [] # use this one for locking
        my_id = constellation.my_id
        # check migration status
        if not self.is_migrating(): # need to activate migration
            self.set_migrating()
        old_gids = self._get_gids(constellation) # get original gids
        try:
            if not filipod:
                # check whether valid point: will raise CollisionError if not
                new_gids = constellation._get_gids(coordinate,None,self.radius)
                gids = set(old_gids + new_gids)
                # write locks gids fronts
                constellation._test_collision(None,coordinate,self.radius,self,gids)
            else: # compute coordinate at filipod, raise CollisionError if it fails
                # write locks old_gids + new_gids fronts
                coordinate, new_gids , tgids1 = \
                            constellation._move_to_fili(self,fili,axon,old_gids)
                if not trailing_axon:
                    gids = set(old_gids + new_gids + tgids1)
        except GridCompetitionError as error:
            raise GridCompetitionError(error.gid)
        # now perform move
        self.orig = self.end = coordinate
        self._set_moved_now()
        self._set_migrated()
        # make axon before we remove soma from old grid space
        if trailing_axon: # create a new trailing axon and perform all updates
            tgids2 = constellation._trailing_axon(self,axon)
            gids = set(old_gids + new_gids + tgids1 + tgids2)
        ### now update grid for move if needed ###
        changed = False
        for gid in old_gids:
            if gid not in new_gids:
                constellation._grid_remove(gid,id)
                changed = True
            # else: gid was set previously in grid, no need to update
            #constellation._grid_wlock[gid] = 0 # unlock
        for gid in new_gids: # set new grid points
            if gid not in old_gids:
                constellation._grid_set(gid,id)
                changed = True
            #constellation._grid_wlock[gid] = 0 # unlock
        ### end grid update, not yet unlocked ###
        if changed: # store updated gids
            self._store_gids(constellation,new_gids)
        # unlock all the locked fronts
        for gid in gids:
            if constellation._grid_wlock[gid] == my_id:
                constellation._grid_wlock[gid] = 0
            else:
                raise BugError("migrate_soma","lock error " + str(constellation.my_id) + " " + str(gid) + " " +  str(constellation._grid_wlock[gid]))
        # This needs to be done after gids unlocked, otherwise it can cause
        #   a reciprocal block.
        if filipod: # only remove after we have updated grid for soma move
            constellation._delete_front(fili,filipod=True)
        if constellation.verbose >= 6:
            print (constellation.my_id,"migrated",self,coordinate)
        return coordinate

    def retract(self,constellation):
        """ Growth method. Retract *self*: the front is removed from the simulation.

        This can be called only if the front has no children, otherwise a TypeError is raised. The irreversible retraction is done at the end of the current cycle. Next cycle will have ``self.has_child_retracted()==True``.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        """
        if constellation.verbose >= 3:
            print (constellation.my_id,"retract",self._fid)
        if self.num_children > 0:
            raise TypeError("retract not possible when children are present")
        if constellation._n_newAID >= constellation._new_range:
            raise OverflowError("_new_AIDs","max_active")
        aid = ActiveFrontID(self.get_id(),b'd')
        constellation._new_AIDs[constellation._n_start+constellation._n_newAID] = aid
        constellation._n_newAID += 1


    def retract_branch(self,constellation,child):
        """ Growth method. Retract *child* and all its (grand)children: the fronts are removed from the simulation.

        This can be called on any child of *self* and is robust to any status of the child and its (grand)children. The irreversible retraction is done at the end of the current cycle. Next cycle will have ``self.has_child_retracted()==True``.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        child : ``Front`` : the root of the branch to be retracted.
        """
        if constellation.verbose >= 3:
            print (constellation.my_id,constellation.cycle,"retract_branch",child._fid)
        # check whether child exists
        children = self.get_children(constellation)
        found = False
        for c in children:
            if c == child:
                found = True
                break
        if not found:
            raise TypeError("retract_branch expects a child of *self*")
        # send instructions to admin: admin will take care of this
        if constellation._n_newAID >= constellation._new_range:
            raise OverflowError("_new_AIDs","max_active")
        aid = ActiveFrontID(child.get_id(),b'd')
        constellation._new_AIDs[constellation._n_start+constellation._n_newAID] = aid
        constellation._n_newAID += 1
        
    # return ID of front
    def get_id(self):
        """ Attribute method that returns the ``ID`` of self.

        Returns
        -------
        self's unique ID : ID
        """
        return ID(self._nid,self._fid)
    
    # return DataID of front
    def get_dataid(self,constellation):
        """ Attribute method that returns the ``DataID`` of self.

        Returns
        -------
        self's data id : DataID
        """
        neuron_id = constellation._fronts[self._nid][self._sid]._sid
        return DataID(neuron_id,self._fid)
    
    def get_neuron(self,constellation):
        """ Attribute method that returns the neuron self belongs to.

        Parameters
        ----------
        constellation : ``Constellation`` object.

        Returns
        -------
        self's neuron : Neuron
        """
        if self.order == 0: # soma
            neuron_id = self._sid
        else: # other front
            neuron_id = constellation._fronts[self._nid][self._sid]._sid
        return constellation._neurons[neuron_id]

    def get_neuron_type(self,constellation,index=True):
        """ Attribute method that returns the front subclass of the neuron.
        
        The method can return either an index into the *neuron_types* list that was used for ``Admin_agent`` initialization, with value 1 for the first entry, or the name of the ``Front`` subclass. 

        Parameters
        ----------
        constellation : ``Constellation`` object.
        Optional :
        index : boolean : return index starting with 1 into neuron_types, default: True.

        Returns
        -------
        self's neuron type: integer or string
        """
        if index:
            return self._nid
        else: # other front
            return constellation._neuron_types[self._nid]

    # return neuron_name: is stored in the Neuron
    def get_neuron_name(self,constellation):
        """ Attribute method that returns the name of the neuron self belongs to.

        Parameters
        ----------
        constellation : ``Constellation`` object.

        Returns
        -------
        self's neuron name : string
        """
        if self.order == 0: # soma
            neuron_id =self._sid
        else: # other front
            neuron_id = constellation._fronts[self._nid][self._sid]._sid
        return repr(constellation._neurons[neuron_id].neuron_name)[2:-1]

    # branch_name: does not exist for soma -> generate it
    def get_branch_name(self):
        """ Attribute method that returns the optional branch name of self.

        Returns
        -------
        self's branch name : string
        """
        if self.order == 0: # soma
            return swc_name(self.swc_type)
        else: # other front
            return repr(self.name)[2:-1]

    def set_branch_name(self,name):
        """ Attribute method that sets the optional branch name of self.
        
        Cannot be applied to soma, which is always called "soma".

        Parameters
        ----------
        name : string : new branch name
        """
        if self.order == 0: # soma
            return
        else: # other front
            self.name = name[:BNAMELENGTH].encode('utf-8')

    # returns parent front or None
    def get_parent(self,constellation,returnID=False,printing=False):
        """ Tree method that returns the parent of self.
        
        This method by default returns ``Front`` or None, but can also return an ``ID``.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        Optional :
        printing: boolean : also print soma, output depends on returnID, default: False
        returnID : boolean : return ``ID``, default: False.

        Returns
        -------
        self's parent, depending on returnID parameter : Front or ID or None
        """
        pid = self._pid
        if pid == -1:
            if printing:
                print ("Parent of",self_str,": none")
            return None
        else:
            if printing:
                print ("Parent of",self_str,":")
            if returnID:
                if printing: 
                    print ("  ",ID(self._nid,pid))
                return ID(self._nid,pid)
            else:
                if printing: 
                    print ("  ",constellation._fronts[self._nid][pid])
                return constellation._fronts[self._nid][pid]

    # is self the parent of front?
    def is_parent(self,front):
        """ Tree method that confirms whether self is the parent of front.
        
        Parameters
        ----------
        front : ``Front`` object.
        
        Returns
        -------
        is self the parent of front? : boolean
        """
        return (self._nid == front._nid) and (front._pid == self._fid)

    # is self grand_x_n parent of front?
    def is_ancestor(self,constellation,front):
        """ Tree method that confirms whether self is a (distant) parent of front, e.g. grandparent, grand-grandparent, etc.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        front : ``Front`` object.
        
        Returns
        -------
        is self an ancestor of front? : boolean
        """
        if self._nid != front._nid:
            return False # different neuron type
        parent = front.get_parent(constellation)
        while parent:
            if self == parent:
                return True
            parent = parent.get_parent(constellation)
        return False

    # is self a child of front?
    def is_child(self,front):
        """ Tree method that confirms whether self is a child of front.
        
        Parameters
        ----------
        front : ``Front`` object.
        
        Returns
        -------
        is self a child of front? : boolean
        """
        return (self._nid == front._nid) and (self._pid == front._fid)
    
    # returns a list of all children of front, can be empty
    def get_children(self,constellation,returnID=False,swc_type=0,printing=False):
        """ Tree method that returns a list of all children of self.
        
        This method by default returns a list of ``Front``, but can also return a list of ``ID``. The children returned can be selected by swc_type.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        Optional :
        printing: boolean : also print list of children, output depends on returnID, default: False
        returnID : boolean : return list of ``ID``, default: False.
        swc_type : integer : only return children with specified swc_type, default: return all children.
 
        Returns
        -------
        Depending on returnID parameter nds_list of Front or nds_list of ID : list
        """
        if self.num_children == 0: # no child
            return []
        if printing:
            if returnID:
                self_str = str(ID(self._nid,self._fid))
            else: 
                self_str = str(self)
            print ("Children of",self_str,":")
        count = 0
        # if this is not called for the front that is doing manage_front on this processor, it may actually be calling constellation._add_child on
        #   another processor causing errors because of a partial update: in that case just repeat as this will be solved by time.
        while True:
            if self.num_children == 1:
                try:
                    child = constellation._fronts[self._nid][self._cid]
                except:
                    if count < 10:
                        count += 1
                        continue # try again
                    else:
                        raise BugError("get_children","try zero child at front " + str(constellation.cycle) + " " + str(self))
                if child._nid == 0:
                    if count < 5:
                        count += 1
                        continue # try again
                    else:
                        raise BugError("get_children","zero child at front " + str(constellation.cycle) + " " + str(self) + " " + str(child))
                if child.is_retracted():
                    raise BugError("get_children","retracted child " + str(child))
                if (swc_type == 0) or (child.swc_type == swc_type):
                    if returnID:
                        if printing:
                            print ("  #1:",ID(self._nid,self._cid))
                        return nds_list([ID(self._nid,self._cid)])
                    else:
                        if printing:
                            print ("  #1:",child)
                        return nds_list([child])
                else:
                    return []
            else:
                children = constellation._children
                child_link = self._cid # start of linked list
                result = nds_list()
                n = 1
                failed = False
                while True:
                    try:
                        child = constellation._fronts[self._nid][children[child_link]._cid]
                    except:
                        if count < 5:
                            count += 1
                            failed = True
                            break # leave inner loop
                        else:
                            print (self._nid,self._cid)
                            raise BugError("get_children","try zero child at " + str(constellation.my_id) + " " + str(constellation.cycle) + " " + str(child_link) + " " + str(self))
                    if child._nid == 0:
                        if count < 5:
                            count += 1
                            failed = True
                            break # leave inner loop
                        else:
                            print (self._nid,self._cid,n)
                            raise BugError("get_children","zero child at " + str(constellation.my_id) + " " + str(constellation.cycle) + " " + str(child_link) + " " + str(self) + " " + str(child))
                    if child.is_retracted():
                        raise BugError("get_children","retracted child " + str(child))
                    if (swc_type == 0) or (child.swc_type == swc_type):
                        if returnID:
                            if printing:
                                print ("  #" + str(n) + ":",\
                                            ID(self._nid,children[child_link]._cid))
                            result.append(ID(self._nid,children[child_link]._cid))
                        else:
                            if printing:
                                print ("  #" + str(n) + ":",child)
                            result.append(child)
                        n += 1
                    child_link = children[child_link].next
                    if child_link == 0: # end of linked list
                        break # out of while True
                if failed:
                    continue # try again
                if (swc_type == 0) and (len(result) != self.num_children):
                    raise BugError("get_children","num_children wrong value")
                return result

    # returns soma front or None
    def get_soma(self,constellation,returnID=False,printing=False):
        """ Tree method that returns the soma of the neuron self belongs to.
        
        This method by default returns ``Front``, but can also return an ``ID``.

        Parameters
        ----------
        constellation : ``Constellation`` object.
        Optional :
        printing: boolean : also print soma, output depends on returnID, default: False
        returnID : boolean : return ``ID``, default: False.

        Returns
        -------
        self's soma, depending on returnID parameter : Front or ID
        """
        if printing:
            if returnID:
                self_str = str(ID(self._nid,self._fid))
            else: 
                self_str = str(self)
        if self.order == 0: # soma
            if printing:
                print ("Soma of",self_str,": none (is a soma)")
            if returnID:
                return ID(self._nid,self._fid)
            else:
                return self
        else:
            if printing:
                print ("Soma of",self_str,":")
            if returnID:
                if printing: 
                    print ("  ",ID(self._nid,self._sid))
                return ID(self._nid,self._sid)
            else:
                if printing: 
                    print ("  ",constellation._fronts[self._nid][self._sid])
                return constellation._fronts[self._nid][self._sid]

    # return entire neuron as a list based on any front belonging to it
    def get_neuron_fronts(self,constellation,returnID=False):
        """ Tree method that returns a list of all fronts of the entire neuron self belongs to.
        
        This method by default returns a list of ``Front``, but can also return a list of ``ID``.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        Optional :
        returnID : boolean : return list of ``ID``, default: False.
 
        Returns
        -------
        Depending on returnID parameter nds_list of [Front,] or [ID,] : list.
        """
        # get soma: root of the tree
        if self.order > 0:
            soma = constellation._fronts[self._nid][self._sid]
        else:
            soma = self
        # iteratively traverse all children
        if returnID:
            result = nds_list([ID(soma._nid,soma._fid)])
        else:
            result = nds_list([soma])
        if soma.num_children == 0:
            return result
        else:
            soma._add_children(constellation,result,returnID)
        return result
        
    def get_terminals(self,constellation,returnID=False):
        """ Returns a list of all terminal fronts for self.
        
        This method by default returns a list of ``Front``, but can also return a list of ``ID``.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        Optional :
        returnID : boolean : return list of ``ID``, default: False.
 
        Returns
        -------
        Depending on returnID parameter nds_list of [Front,] or [ID,] : list.
        """
        result = nds_list()
        self._get_terminals(constellation,result,returnID)
        return result
    
    # recursive routine that performs preorder tree traversal to make list of all terminals
    def _get_terminals(self,constellation,result,returnID):
        
        if self.num_children > 0: # not a terminal
            children = self.get_children(constellation)
            for child in children:
                child._get_terminals(constellation,result,returnID)
        else: # terminal
            if returnID:
                result.append(ID(self._nid,self._fid))
            else:
                result.append(self)
        return
        
    # return list of neighboring fronts within a given distance
    # distance: float: max distance, taken relative to path_length of calling front
    # branch_stop: boolean: stop at first branch point, both directions
    def get_neighbors(self,constellation,distance,branch_stop=False,\
                        returnID=False):
        """ Tree method that returns a list of fronts that are close to self on the neuron tree.
        
        Return all fronts that are within a given path_length distance of self, in both somatopetal and somatofugal directions, or (option) till next branch point. This method by default returns a list of ``Front``, but can also return a list of ``ID``.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        distance : float : maximum distance (inclusive) that returned fronts can be.
        Optional :
        branch_stop : boolean : stop at first branch point encountered, default: False.
        returnID : boolean : return list of ``ID``, default: False.
 
        Returns
        -------
        Depending on returnID parameter nds_list of [Front,] or [ID,] : list.
        """
        if constellation.verbose >= 3:
            print ("get_neighbors",self._fid,distance)
        result = nds_list()
        start = self.path_length
        # first go toward soma
        parent = self.get_parent(constellation)
        while parent:
            if abs(parent.path_length - self.path_length) > distance:
                break # beyond search distance, break out of while
            if branch_stop and parent.num_children > 1: # reached a branch point
                break # break out of while
            if returnID:
                result.append(parent.get_id())
            else:
                result.append(parent)
            parent = parent.get_parent(constellation)
        self._add_children(constellation,result,returnID,\
                            by_distance=[self.path_length,distance,branch_stop])
        return result

    # recursive routine that performs preorder tree traversal to make list of all children
    def _add_children(self,constellation,result,returnID,by_distance=[]):
        if self._nid == 0:
            raise BugError("_add_children","zero child " + str(self))
        if (len(by_distance) > 0) and by_distance[2] and (self.num_children) > 1: # branch_stop == True
            return # reached a branch point
        if self.num_children > 0:
            children = self.get_children(constellation)
            for child in children:
                if len(by_distance) > 0:
                    if abs(child.path_length - by_distance[0]) > by_distance[1]:
                        return # beyond search distance
                if returnID:
                    result.append(ID(child._nid,child._fid))
                else:
                    result.append(child)
                child._add_children(constellation,result,returnID,by_distance)

    def count_descendants(self,constellation):
        """ Returns number of descendants, the total number of all children, grandchildren, grand-grandchildren, etc.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
 
        Returns
        -------
        Count of descendant fronts : integer.
        """
        result = []
        self._add_child_count(constellation,result)
        return len(result)
        
    def _add_child_count(self,constellation,result):
        if self.num_children > 0:
            children = self.get_children(constellation)
            for child in children:
                result.append(1)
                child._add_child_count(constellation,result)
    
    def enable(self,constellation,growing=False,migrating=False):
        """ Behavior method: make a front active: it calls ``manage_front`` each cycle, starting next cycle.
        
        It is safe to call this method on other fronts than *self* without locking. If the front is already active a Warning will be printed if verbose >= 2 and nothing will change.
         
        Parameters
        ----------
        constellation : ``Constellation`` object.
        Optional:
        growing : boolean : flag enabled front as growing, default False: not growing.
        migrating : boolean : flag enabled front as migrating, default False: not migrating.        
        """
        if self.is_active():
            if constellation.verbose >= 2:
                print (Fore.MAGENTA + "Warning: enable of already active front",self,Fore.RESET)
            return
        # status can be changed now as it will not affect scheduling
        self._set_active()
        if migrating:
            self.set_migrating()
        elif growing:
            self.set_growing()
        # check for attribute storage
        if self.order == 0: # a soma
            if self._does_storing():
                storing = True
            else:
                storing = False
        else: # check soma status
            soma = self.get_soma(constellation)
            if soma._does_storing():
                self._set_storing()
                storing = True
            else:
                storing = False
        # check whether called for front that is not already sent to Admin
        if storing or ((self != constellation._manage) and \
                (self not in constellation._newfs)): # set on other front
            # store info for Admin
            if constellation._n_newAID >= constellation._new_range:
                raise OverflowError("_new_AIDs","max_active")
            aid = ActiveFrontID(self.get_id(),b'a')
            constellation._new_AIDs[constellation._n_start +\
                                            constellation._n_newAID] = aid
            constellation._n_newAID += 1 # update index into new_AIDs

    def disable(self,constellation,till_cycle=0,till_cycle_g=0,till_cycle_m=0):
        """ Behavior method: make a front inactive: it no longer calls ``manage_front``, starting next cycle.
        
        Also clears is_growing() and is_migrating().  
        It is safe to call this method on other fronts than *self* without locking.  
        If the *till_cycle* parameter is used this will make the front active again on *till_cycle*. In addition the front can be set to growing with *till_cycle_g* or to migrating with *till_cycle_m*. Only one of the *till_cycle options may be used. If the front should be made both growing and migrating, use *till_cycle_g* and call ``self.set_migrating()`` when *manage_front* is called again.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        Optional:
        till_cycle : integer : transiently disable, at till_cycle will be active again, default 0: permanent disable.
        till_cycle_g : integer : transiently disable, at till_cycle will be growing and active, default 0: permanent disable.
        till_cycle_m : integer : transiently disable, at till_cycle will be migrating and active, default 0: permanent disable.
        """
        if not self.is_active():
            if constellation.verbose >= 2:
                print (Fore.MAGENTA + "Warning: disable of already inactive front",self,Fore.RESET)
            return
        till = 0
        key = b'i'
        if till_cycle < 0:
            raise ValueError("till_cycle"," >= 0")
        if till_cycle_g < 0:
            raise ValueError("till_cycle_g"," >= 0")
        if till_cycle_m < 0:
            raise ValueError("till_cycle_m"," >= 0")
        if till_cycle > 0:
            till = till_cycle
        if till_cycle_g > 0:
            if till > 0:
                raise ValueError("till_cycle_g","0 if 'till_cycle' is larger than 0")
            till = till_cycle_g
            key = b'g'
        if till_cycle_m > 0:
            if till > 0:
                raise ValueError("till_cycle_g","0 if 'till_cycle' or 'till_cycle_g' is larger than 0")
            till = till_cycle_m
            key = b'm'
        
        # disable storing unless soma
        if (self.order > 0) and self._does_storing():
            self._clear_storing()
        # check whether called for front that is not already sent to Admin
        if ((self == constellation._manage) or (self in constellation._newfs)) and (till == 0): 
            # front local to this processor this cycle -> change now
            self._clear_active()
            self.clear_growing()
            self.clear_migrating()
        else: # instruct Admin to change
            if constellation._n_newAID >= constellation._new_range:
                raise OverflowError("_new_AIDs","max_active")
            aid = ActiveFrontID(self.get_id(),key,till)
            constellation._new_AIDs[constellation._n_start +\
                                            constellation._n_newAID] = aid
            constellation._n_newAID += 1 # update index into new_AIDs
            
    def enable_parent(self,constellation,growing=False,migrating=False):
        """ Behavior method: make the parent of a front active: it calls ``manage_front`` each cycle, starting next cycle.
        
        If the parent is already active nothing will change.
                 
        Parameters
        ----------
        constellation : ``Constellation`` object.
        Optional:
        growing : boolean : flag enabled front as growing, default False: not growing.
        migrating : boolean : flag enabled front as migrating, default False: not migrating.        
        """
        pid = self._pid
        if pid == -1:
            front = self
        else:
            front = constellation._fronts[self._nid][pid]
        if front.is_active():
            return
        front.enable(constellation,growing=growing,migrating=migrating)

    # routines to manage different flag settings
    # _flag encoding:
    # bit 0: cylindrical
    # bit 1: active (enabled)
    # bit 2: growing
    # bit 3: migrating
    # bit 4: moved on last cycle (only for migrating somata)
    # bit 5: has migrated sometime
    # bit 6: moved this cycle (only for migrating somata), used to update bit 4
    # bit 7: not used
    # bit 8: retracted
    # bit 9: has a retracted child
    # bit 10: part of an arc
    # bit 11: store new attribute in database
    # bit 12: not used
    # bit 13: status1 for user
    # bit 14: status2 for user
    # bit 15: status3 for user

    ### _flag routines: some public, some private
    
    def _print_flags(self):
        text = ""
        for n in range(16):
            mask = 1 << n
            text += str(n) + ":" + str(self._flags & mask) + " "
        print (text)
        
    # is front a cylinder? Otherwise it is a sphere.
    def is_cylinder(self):
        """ Behavior method: is self a cylinder (True) or a sphere (False)?.
         
        Returns
        -------
        is *self* a cylinder? : boolean
        """
        mask = 1 << 0
        return (self._flags & mask) > 0

    def _set_cylinder(self):
        mask = 1 << 0
        self._flags = self._flags | mask

    # is front active (enabled)?
    def is_active(self):
        """ Behavior method: is self active (calling manage_front)?.
         
        Returns
        -------
        is self active? : boolean
        """
        mask = 1 << 1
        return (self._flags & mask) > 0

    def _set_active(self):
        mask = 1 << 1
        self._flags = self._flags | mask

    def _clear_active(self):
        if self.is_active():
            self._flags = self._flags - 2 # 2**1

    # is front growing?
    def is_growing(self):
        """ Behavior method: is self growing?.
         
        Returns
        -------
        is self growing? : boolean
        """
        mask = 1 << 2
        return (self._flags & mask) > 0

    def set_growing(self):
        """ Behavior method: signal self is growing. Makes front active.
        
            Only set if ``self.is_migrating() == False``
        """
        if not self.is_active():
            self._set_active()
        if not self.is_migrating():
            mask = 1 << 2
            self._flags = self._flags | mask

    def clear_growing(self):
        """ Behavior method: signal self stopped growing.
        """
        if self.is_growing():
            self._flags = self._flags - 4 # 2**2

    # is soma migrating (calls migrate_front)?
    def is_migrating(self):
        """ Behavior method: is self migrating now?.
         
        Returns
        -------
        is self migrating now? : boolean
        """
        mask = 1 << 3
        return (self._flags & mask) > 0

    def set_migrating(self):
        """ Behavior method: signal self is migrating. Makes front active.
        
            Clears self.is_growing()
        """
        if self.is_growing():
            self.clear_growing()
        if not self.is_active():
            self._set_active()
        mask = 1 << 3
        self._flags = self._flags | mask

    def clear_migrating(self):
        """ Behavior method: signal self stopped migrating.
        """
        if self.is_migrating():
            self._flags = self._flags - 8 # 2**3

    # has migrating front moved on previous cycle?
    def has_moved(self):
        """ Behavior method: has migrating self moved in last call to `manage_front`?.
        
        At start of `manage_front` call this reflects whether it moved during the previous cycle. Upon return from `manage_front` it is set to what happened on current cycle.
        
        Returns
        -------
        has migrating self moved? : boolean
        """
        mask = 1 << 4
        return (self._flags & mask) > 0

    def _set_moved(self):
        mask = 1 << 4
        self._flags = self._flags | mask

    def _clear_moved(self):
        if self.has_moved():
            self._flags = self._flags - 16 # 2**4

    # has front ever migrated?
    def has_migrated(self):
        """ Behavior method: has self ever migrated?.
        
        Returns
        -------
        has self ever migrated? : boolean
        """
        mask = 1 << 5
        return (self._flags & mask) > 0

    def _set_migrated(self):
        mask = 1 << 5
        self._flags = self._flags | mask

    # has migrating front moved on this cycle?
    def _has_moved_now(self):
        mask = 1 << 6
        return (self._flags & mask) > 0

    def _set_moved_now(self):
        mask = 1 << 6
        self._flags = self._flags | mask

    def _clear_moved_now(self):
        if self._has_moved_now():
            self._flags = self._flags - 64 # 2**6

    # is front retracted (deleted after birth)?
    def is_retracted(self):
        """ Behavior method: has self been retracted (deleted after birth)?.
         
        Returns
        -------
        has self been retracted? : boolean
        """
        mask = 1 << 8
        return (self._flags & mask) > 0

    def _set_retracted(self):
        mask = 1 << 8
        self._flags = self._flags | mask

    # has a child of this front been retracted?
    def has_child_retracted(self):
        """ Behavior method: has a child of self been retracted in the past?.
        
        This flag is reset to False when self grows a new child.
         
        Returns
        -------
        has a child of self been retracted? : boolean
        """
        mask = 1 << 9
        return (self._flags & mask) > 0

    def _set_child_retracted(self):
        mask = 1 << 9
        self._flags = self._flags | mask

    def _clear_child_retracted(self):
        if self.has_child_retracted():
            self._flags = self._flags - 512 # 2**9
            
    # is front part of an arc?
    def is_arc(self):
        """ Behavior method: is self part of an arc made by the ``arc_around`` method?
         
        Returns
        -------
        is self part of an arc? : boolean
        """
        mask = 1 << 10
        return (self._flags & mask) > 0

    def _set_arc(self):
        mask = 1 << 10
        self._flags = self._flags | mask

    def _clear_arc(self):
        if self.is_arc():
            self._flags = self._flags - 1024 # 2**10
            
    # does front have attributes to store?
    def _does_storing(self):
        mask = 1 << 11
        return (self._flags & mask) > 0

    def _set_storing(self):
        mask = 1 << 11
        self._flags = self._flags | mask

    def _clear_storing(self):
        if self._does_storing():
            self._flags = self._flags - 2048 # 2**11

    # use status1
    def is_status1(self):
        """ Behavior method: is self status1 set?.
         
        Returns
        -------
        is self status1 set? : boolean
        """
        mask = 1 << 13
        return (self._flags & mask) > 0

    def set_status1(self):
        """ Behavior method: set self status1 to True.
        """
        mask = 1 << 13
        self._flags = self._flags | mask

    def clear_status1(self):
        """ Behavior method: set self status1 to False.
        """        
        if self.is_status1():
            self._flags = self._flags - 8192 # 2**13
            
    # use status2
    def is_status2(self):
        """ Behavior method: is self status2 set?.
         
        Returns
        -------
        is self status2 set? : boolean
        """
        mask = 1 << 14
        return (self._flags & mask) > 0

    def set_status2(self):
        """ Behavior method: set self status2 to True.
        """
        mask = 1 << 14
        self._flags = self._flags | mask

    def clear_status2(self):
        """ Behavior method: set self status2 to False.
        """        
        if self.is_status2():
            self._flags = self._flags - 16384 # 2**14
            
    # use status3
    def is_status3(self):
        """ Behavior method: is self status3 set?.
         
        Returns
        -------
        is self status3 set? : boolean
        """
        mask = 1 << 15
        return (self._flags & mask) > 0

    def set_status3(self):
        """ Behavior method: set self status3 to True.
        """
        mask = 1 << 15
        self._flags = self._flags | mask
        
    def clear_status3(self):
        """ Behavior method: set self status3 to False.
        """        
        if self.is_status3():
            self._flags = self._flags - 32768 # 2**15
            
    def has_synapse(self):
        """ Has self a synaptic connection defined?. Always False for ``Front``.
            
        Returns
        -------
        False : boolean
        """
        return False

    def get_fronts(self,constellation,what="other",name="",type=0,max_cycle=-1,\
                    max_distance=100.,swc_types=[],sort=True,returnID=False):
        """ Search method: returns all fronts with a specific property within Euclidian max_distance (default 100 µm).

        This method by default returns a distance sorted list of ``Front``, but can also return a list of ``ID``. The *what* controls overall behavior of the method, this can be further refined by changing *max_distance* or specifying a subset of *swc_types* to return.
        
        Speed of method depends on *max_distance*: for *max_distance* <= *Admin_agent.grid_step* a different method is used that is less sensitive to number of neurons in the simulation. Otherwise this method's computation time scales with the number of neurons in the simulation.

        Parameters
        ----------
        constellation : ``Constellation`` object.
        Optional :
        what : string: string options:
         - 'self': get fronts belonging to same neuron as self, excluding all up to second order ancestors and descendents.
         - 'self+': get all fronts within max_distance belonging to self.
         - 'name': get fronts belonging to neurons with name (wildcard), not including same neuron as self.
         - 'other': get fronts that do not belong to self (default).
         - 'type': get fronts belonging to this type of neuron, not including same neuron as self.
        max_cycle : integer : ignore fronts born at this cycle or later, default -1: use all fronts.
        name : string : used for the what=='name' and what=='type' options.
        max_distance : float : maximum Euclidian distance, inclusive, default: 100 µm.
        swc_types: list : only include fronts with these swc_types.
        sort : boolean : sort the list by increasing distance, default: True.
        returnID : boolean : return list of ``ID``, default: False.

        Returns
        -------
        Depending on returnID parameter nds_list of tuples (Front, distance) or (ID, distance) : nds_list of tuples
        """
        if max_cycle > 0:
            test_cycle = True
        else:
            test_cycle = False
        result = nds_list() # list to be returned
        if what not in ['self','self+','name','other','type']:
            raise ValueError("get_fronts what","'self','self+','name','other' or 'type'")
        if type and name not in constellation._neuron_types:
            raise ValueError("get_fronts type",str(name) + " does not exist")

        volume = constellation._volume
        cycle = constellation.cycle
        ssoma = self.get_soma(constellation) # self.soma
        my_id = self.get_id()
        if what == 'type': # get corresponding nid
            nid = constellation._neuron_types.index(name) + 1
        else:
            check_self = what == 'self'
        if check_self: # make list of all ids to exclude
            exclude = []
            p = self.get_parent(constellation)
            if p:
                exclude.append(p.get_id()) # parent
                pchildren = p.get_children(constellation,returnID=True)
                exclude.extend(pchildren) # siblings
                gp = p.get_parent(constellation)
                if gp:
                    exclude.append(gp.get_id()) # grandparent
            children = self.get_children(constellation,returnID=True)
            exclude.extend(children) # children
            for id in children:
                child = constellation.front_by_id(id)
                gc = child.get_children(constellation,returnID=True)
                exclude.extend(gc) # grandchildren

        if (not check_self) and (max_distance <= volume.grid_step): 
            # try to use grid to search nearby fronts
            res_ids = [] # list of detected ids
            gids = [] # list of gids to get
            # get the gids needed
            gid0 = self.orig._grid(volume)
            gids.append(gid0)
            if self.is_cylinder():
                gid1 = self.end._grid(volume)
                gids.append(gid1)
                gid2 = self.mid()._grid(volume)
                if (gid2 != gid0) and (gid2 != gid1):
                    gids.append(gid2)
            # perform locked _grid_get
            ids = set()
            count = 0
            while (len(gids) > 0) or (count < 20):
                # loop through gids till all are processed
                count += 1
                for gid in gids:
                    if constellation._automatic: # default mode
                        if constellation._grid_rlock[gid] == 0: # get brief read-only lock
                            constellation._grlock_request[constellation.my_id] = gid
                            wait = 0.
                            while constellation._grid_rlock[gid] !=\
                                                            constellation.my_id:
                                time.sleep(LOCKPAUSE)
                                wait += LOCKPAUSE
                                if wait > 2.0:
                                    raise BugError("get_fronts","read-only waited for two seconds on " + str(constellation.my_id) + " for grid " + str(gid))
                        else:
                            continue # for loop
                    else: # interactive mode -> no lock_broker
                        constellation._grid_rlock[gid] = 1
                        wait = 0.
                    ids |= set(constellation._grid_get(gid))
                    constellation._grid_rlock[gid] = 0 # unlock it after use
                    constellation._grid_lock_wait += wait
                    gids.remove(gid)
                    break # for loop, changed gids
            grid_based = len(gids) == 0
        else:
            grid_based = False
            
        if grid_based: # succeeded in getting all ids
            for id in ids:
                if id == my_id:
                    continue
                if id in res_ids:
                    continue
                front = constellation.front_by_id(id)
                # eliminate all that are invalid
                dist = self.front_distance(front)
                if dist > max_distance: # too far
                    continue
                if swc_types and (front.swc_type not in swc_types): # wrong swc_type
                    continue
                #    check for each what
                soma = front.get_soma(constellation)
                if what == 'other':
                    if soma == ssoma:
                        continue
                elif what == 'name':
                    if soma == ssoma:
                        continue
                    if not front.get_neuron_name(constellation).startswith(name):
                        continue
                elif what == 'type':
                    if soma == ssoma:
                        continue
                    if front._nid != nid:
                        continue
                else: # 'self' and 'self+'
                    soma = front.get_soma(constellation)
                    if soma != ssoma: # different neuron
                        continue
                    if check_self and id in exclude: # is an ancestor/descendent
                        continue
                # store this one
                res_ids.append(id)
                if returnID:
                    result.append((id,dist))
                else:
                    result.append((front,dist))
        else: # use neuron tree based search
            # copy dictionary information to entities depending on what
            if (what == 'other') or (what == 'name') or (what == 'type'):
                check_self = False
                # first compile list of all relevant somata depending on what as starting point
                neurons = []
                for n in range(constellation._num_types):
                    for i in range(1,constellation._f_next_indices[n+1][1]):
                        soma = constellation._fronts[n+1][i]
                        if self._nid == 0:
                            raise BugError("get_fronts","zero soma at " + str(n) + " " + str(i) + " " + str(soma))
                        if soma.order != 0:
                            raise BugError("get_fronts","not a soma " + str(soma))
                        if soma == ssoma:
                            continue
                        if what == 'other':
                            neurons.append(soma)
                        elif what == 'name':
                            if soma.get_neuron_name(constellation).startswith(name):
                                neurons.append(soma)
                        else: # what == 'type'
                            if soma._nid == nid:
                                neurons.append(soma)
            else:  # 'self' and 'self+'
                neurons = [ssoma]

            # now check all fronts belonging to neurons and keep only those within distance and proper swc_type
            for soma in neurons:
                # generate list of all fronts
                ids = soma.get_neuron_fronts(constellation,returnID=True)
                for id in ids:
                    if id == my_id:
                        continue
                    if check_self and id in exclude: # is an ancestor/descendent
                        continue
                    front = constellation.front_by_id(id)
                    if front.is_retracted():
                        continue
                    if test_cycle and (front.birth >= max_cycle):
                        continue
                    dist = self.front_distance(front)
                    if dist > max_distance: # too far
                        continue
                    if swc_types and (front.swc_type not in swc_types): # wrong swc_type
                        continue
                    # store this one
                    if returnID:
                        result.append((id,dist))
                    else:
                        result.append((front,dist))
        if sort:
            return nds_list(sorted(result,key=itemgetter(1)))
        else:
            return result

    def get_substrates(self,constellation,name,max_distance=100,sort=True,\
                       returnID=False):
        """ Search method: returns all substrate with a specific name within Euclidian max_distance (default 100 µm).

        This method by default returns a distance sorted list of ``Substrate``, but can also return a list of ``ID``. *max_distance* can be changed as an optional parameter.

        Parameters
        ----------
        constellation : ``Constellation`` object.
        name : string : name of the ``Substrate`` (wildcard).
        Optional :
        max_distance : float : maximum Euclidian distance, inclusive, default: 100 µm.
        sort : boolean : sort the list by increasing distance, default: True.
        returnID : boolean : return list of ``ID``, default: False.

        Returns
        -------
        Depending on returnID parameter nds_list of tuples (substrate, distance) or (ID, distance) : nds_list of tuples
        """
        result = nds_list() # list to be returned
        substrate = constellation._substrate
        found = False
        for i in range(1,constellation._max_sub_names): # look for name
            if substrate[i].get_name().startswith(name):
                found = True
                sub = substrate[i]
                dist = self.front_distance(sub)
                if dist <= max_distance: # store
                    if returnID:
                        result.append((ID(0,i),dist))
                    else:
                        result.append((sub,dist))
                while sub._next > 0:
                    n = sub._next
                    sub = substrate[n]
                    dist = self.front_distance(sub)
                    if dist <= max_distance: # store
                        if returnID:
                            result.append((ID(0,n),dist))
                        else:
                            result.append((sub,dist))
        if not found:
            if constellation.verbose >= 2:
                print (Fore.MAGENTA + "Error in get_substrates: name" + str(name) + "does not exist, empty list returned",Fore.RESET)
            return result
        if sort:
            return nds_list(sorted(result,key=itemgetter(1)))
        else:
            return result

    # Checks whether making the cylindrical front point-self.end will trigger a collision
    def point_valid(self,constellation,point,cylinder=True,new_radius=None):
        """ Search method: checks whether given point is a valid location for a new front: inside the simulation volume and not causing any collision.
        
        Depending on the cylinder (default True) optional parameter this method either checks for collisions with a cylinder with as origin *self.end* and as end *point* or, if False, for an isolated sphere with *point* as its origin. It assumes the same *radius* as self unless a *new_radius* is specified.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        point : ``Point`` : location to be validated.
        Optional :
        cylinder : boolean : check for cylinder ending at *point* (True) or isolated sphere around point (False), default True.
        new_radius : float : use this value as radius of cylinder or sphere, default None (use *self.radius* instead).
        
        Returns
        -------
        An exception if point is not valid: CollisionError, InsideParentError or VolumeError
        """
        # check whether coordinate inside simulation volume
        result = point.out_volume(constellation)
        if result != 0.:
            raise VolumeError(result)
            return
        # check whether coordinate inside parent
        if self._point_in_self(point):
            raise InsideParentError()
            return
        if new_radius:
            radius = new_radius
        else:
            radius = self.radius
        if cylinder:
            gids = constellation._get_gids(self.end,point,0.)
            constellation._test_collision(self.end,point,radius,self,\
                                                            gids,lock=False)
        else:
            gids = constellation._get_gids(point,None,radius)
            constellation._test_collision(None,point,radius,self,\
                                                            gids,lock=False)

    # Randomly changes self.end by amount * radius so that new point is inside simulation volume
    # return point
    def wiggle_front(self,constellation,max_scale,min_scale=0.5):
        """ Point method: returns a random point based on current front position.
        
        Called when ``Front`` is colliding. Returns a random point scaled by max_scale * radius added to the proper position. The point is guaranteed to be inside the *simulation volume*.
        
        This method can be subclassed to refine the generation of random points.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        max_scale : float : scales the maximum amount of wiggling by radius.
        Optional :
        min_scale : float : sets a minimum to the amount of wiggling, default: 1/2 radius vector length.
        
        Returns
        -------
        random position : Point
        """
        if max_scale <= min_scale:
            if constellation.verbose >= 2:
                print (Fore.MAGENTA + "Warning: wiggle_front max_scale too small",max_scale,Fore.RESET)
            return self.proper_pos(constellation) # no noise applied
        level = max_scale * self.radius
        min_level = min_scale * self.radius
        count = 0
        while count < 100: # only at border of volume can count be high
            # generate random coordinate in range -level to +level
            noise = (2 * level) * np.random.random(3) - level
            # impose minimum size on noise
            if norm(noise) < min_level:
                continue
            # ADDITIONAL CHECKS ON NOISE CAN BE INSERTED HERE
            # add noise to position causing a collision
            point = self.proper_pos(constellation) + noise
            # check whether random point is inside simulation volume
            if point.out_volume(constellation) == 0.:
                return point
            count += 1
        return self.proper_pos(constellation) # failed, no noise applied

    # Try to solve a collision
    def solve_collision(self,constellation,point,error,new_radius=None,\
                        half_arc=True):
        """ Point method: try to resolve a collision.
        
        Can be called after ``add_child`` exempts with a *CollisionError* or  ``point_valid(constellation,point)`` returns False for a cylindrical *self*. If successful, returns a list of points in the same general growth direction that does not collide, otherwise returns [].
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        point : ``Point`` : the point that caused the collision.
        error : ``Exception`` : the *CollisionError* that we are trying to solve
        Optional :
        new_radius : float : use this value as radius for collision checking, default None.
        half_arc : boolean : if arc made it is 180 deg, else 90 deg arc, default True.
        
        Returns
        -------
        new positions as nds_list: [Point,] or []
        """
        if constellation.verbose >= 3:
            print ("Process",constellation.my_id,"cycle",constellation.cycle,"solve_collision",self._fid)
        if new_radius:
            radius = new_radius
        else:
            radius = self.radius
        try:
            if error.only_first:
                col_front = error.collider
            else:
                col_front = error.collider[0]
        except:
            raise TypeError("solve_collision expects a CollisionError")
        # check whether possibly continuation of arc solves problem
        if self.is_arc():
            if constellation.verbose >= 6:
                print ("arc detected by solve_collision",self._fid)
            points = self.arc_continue(constellation)
            valid = True
            prev_p = self.end
            for p in points: # check for collisions
                if constellation.verbose >= 6:
                    print ("arc point",self._fid,p)
                count = 0
                failed = False
                while True: # loop for GridCompetitionError
                    count += 1
                    if count > 100:
                        raise GridCompetitionError(cgid)
                    try: 
                        gids = constellation._get_gids(prev_p,p,0.)
                        constellation._test_collision(prev_p,p,radius,None,gids,\
                                                        lock=False)
                        break # out of while True        
                    except GridCompetitionError as error:
                        cgid = error.gid
                        time.sleep(2*LOCKPAUSE) # competition with other processor
                        continue # while True
                    except CollisionError as error:
                        failed = True
                        valid = False
                        if constellation.verbose >= 6:
                            print (p,"collides with",error.collider)
                        break # out of while True
                if failed:
                    break # out of for p in points
                prev_p = p
            if valid and points:
                return points
        # analyze type of colliding fronts: at present only the closest one
        sphere = None
        close_f = None
        if col_front.is_cylinder():
            close_f = col_front
        else: # is a sphere
            if col_front.radius > 2 * self.radius: # arc solutions work only for large spheres
                sphere = col_front
            else:
                close_f = col_front
        if constellation.verbose >= 6:
            if constellation.verbose >= 7:
                print (self)
            print (self._fid,point,"solve_collision",sphere,close_f,self.is_arc(),radius)
        if sphere:
            if half_arc:
                a = 180
            else:
                a = 90
            return self.arc_around(constellation,sphere,arc=a,arc_size=self.length(),\
                                        new_radius=new_radius)
        elif close_f:
            # simple implementation, does not handle extra collisions well
            if close_f.is_cylinder():
                # check whether cylinders are close to parallel
                if self.is_cylinder():
                    start = self.end
                    s_dir = point - start
                else:
                    start = self.orig
                    s_dir = point - start
                f_dir = close_f.end - close_f.orig
                c_angle = angle_two_dirs(s_dir, f_dir) # 0-180
                if constellation.verbose >= 6:
                    print (self._fid,"solve_collision",close_f._fid,"test  parallel",c_angle,s_dir,f_dir)
                # threshold was previously 15: failed for angle 23.5 and
                #   then messed up go_around
                if (c_angle < 30.) or (c_angle > 160.): # close to parallel
                    if c_angle > 90.: # f_dir pointing in wrong direction
                        f_dir = f_dir * -1.
                    if constellation.verbose >= 6:
                        print (self._fid,"solve_collision",close_f._fid,"close to parallel",close_f.front_distance(self))                    
                    offs = radius + close_f.radius
                    cur_dist = close_f.front_distance(self)
                    if cur_dist < offs: # self is already too close
                        # go to close point far enough from close_f
                        #   get closest point on close_f relative to point
                        dist,p = dist3D_point_to_cyl(point,close_f.orig,\
                                                    close_f.end,points=True)
                        no_dir = (point - p).norm()
                        new_p = p + no_dir * 1.1 * offs
                        result = new_p.out_volume(constellation)
                        if result != 0.:
                            return []
                        if self._point_in_self(new_p):
                            return []
                        if constellation.verbose >= 6:
                            print (self._fid,"solve_collision close parallel test",new_p,close_f.front_distance(new_p))
                        cur_dist = cur_dist - close_f.radius # we cannot do better
                        # test segment self.end to new_p
                        failed = False
                        count = 0
                        while True: # loop for GridCompetitionError
                            count += 1
                            if count > 100:
                                raise GridCompetitionError(cgid)
                            try: 
                                gids = constellation._get_gids(self.end,new_p,0.)
                                constellation._test_collision(self.end,new_p,\
                                                    radius,self,gids,lock=False)
                                break # out of while True
                            except GridCompetitionError as error:
                                cgid = error.gid
                                time.sleep(2*LOCKPAUSE) # competition with other processor
                                continue # while True
                            except CollisionError as error:
                                failed = True
                                if constellation.verbose >= 6:
                                    print (new_p,"collides with",error.collider)
                                break # out of while True
                        if not failed:  # self.end to new_p  valid
                            return nds_list([new_p])
                    else: # try to realign new front parallel to close_f
                        s_len = s_dir.length()
                        nf_dir = f_dir.norm()
                        # try different distances based on desired length
                        if s_len > 10.:
                            n_trials = 4
                        elif s_len > 4.:
                            n_trials = 2
                        else:
                            n_trials = 1
                        for n in range(n_trials):
                            new_p = start + nf_dir * s_len * (n_trials - n) / n_trials
                            result = new_p.out_volume(constellation)
                            if result != 0.:
                                continue
                            if self._point_in_self(new_p):
                                continue                            
                            if constellation.verbose >= 6:
                                print (self._fid,"solve_collision parallel test",new_p,close_f.front_distance(new_p))
                            # test segment self.end to new_p
                            failed = False
                            count = 0
                            while True: # loop for GridCompetitionError
                                count += 1
                                if count > 100:
                                    raise GridCompetitionError(cgid)
                                try: 
                                    gids = constellation._get_gids(self.end,\
                                                                    new_p,0.)
                                    constellation._test_collision(self.end,new_p,\
                                                    radius,self,gids,lock=False)
                                    break # out of while True
                                except GridCompetitionError as error:
                                    cgid = error.gid
                                    time.sleep(2*LOCKPAUSE) # competition with other processor
                                    continue # while True
                                except CollisionError as error:
                                    failed = True
                                    if constellation.verbose >= 6:
                                        print (new_p,"collides with",error.collider)
                                    break # out of while True
                            if not failed:  # self.end to new_p  valid
                                return nds_list([new_p])
                else: # not parallel, try to go around
                    return self.go_around(constellation,point,close_f,\
                                                            new_radius=new_radius)
            else: # close_f is a sphere: get alternate locations around sphere
                number = max(8,int(2 * close_f.radius))
                alt_points = self.alternate_locations(close_f.orig,\
                                    close_f.radius + 1.5*radius,number)
                if constellation.verbose >= 6:
                    print (self._fid,"solve_collision alternate_locations?",nds_list(alt_points))
                # test the points are return the one closest to point
                good_point = None
                for new_p in alt_points:
                    result = new_p.out_volume(constellation)
                    if result != 0.:
                        continue
                    if self._point_in_self(new_p):
                        continue
                    # test segment self.end to new_p
                    count = 0
                    while True: # loop for GridCompetitionError
                        count += 1
                        if count > 100:
                            raise GridCompetitionError(cgid)           
                        try: 
                            gids = constellation._get_gids(self.end,new_p,0.)
                            constellation._test_collision(self.end,new_p,\
                                                radius,self,gids,lock=False)
                            if not good_point:
                                good_point = new_p
                                good_distance = (new_p - point).length()
                            else:
                                dist = (new_p - point).length()
                                if dist < good_distance:
                                    good_point = new_p
                                    good_distance = dist
                            break # out of while True
                        except GridCompetitionError as error:
                            cgid = error.gid
                            time.sleep(2*LOCKPAUSE) # competition with other processor
                            continue # while True
                        except CollisionError as error:
                            failed = True
                            if constellation.verbose >= 6:
                                print (new_p,"collides with",error.collider)
                            break # out of while True
                if good_point:
                    return nds_list([good_point])
        if constellation.verbose >= 6:
            print (self._fid,"solve_collision returns []")
        return []
           
    # is self.orig (sphere) or self.end (cylinder) in simulation volume?
    # returns: boolean
    def in_volume(self,constellation):
        """ Search method: returns whether the *self* is inside the simulation volume (including its borders).
        
        For cylinder *self.orig* and *self.end* is tested, for sphere *self.orig*.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.

        Returns
        -------
        is inside simulation volume : boolean
        """
        if self.is_cylinder():
            return (self.orig.out_volume(constellation) == 0.) and \
                    (self.end.out_volume(constellation) == 0.)
        else:
            return (self.orig.out_volume(constellation) == 0.)

    # perform rotation of direction vectors
    # if useR == 1-3: use provided R value, rotate depending on useR and return nothing
    #    useR == -1: compute R value, rotate depending on useR and return nothing
    #    useR == 0: compute R, do not rotate and return useR, R
    # vecs is a list of np.array, will be modified if useR != 0
    def _rotate_vectors(self,point,vecs,useR,R):
        if useR <= 0:
            if point: # rotate around line between point and front
                if self.is_cylinder():
                    fdir = point - self.end
                    if fdir.length() == 0.:
                        fdir = point - self.orig
                else:
                    fdir = point - self.orig
            else: # rotate around axis of front cylinder
                fdir = self.end - self.orig
            if fdir.length() == 0.:
                print (Fore.RED + "Error: _rotate_vectors: zero reference vector",Fore.RESET)
                return 3, None
            ufdir0 = fdir.norm()
            ufdir = ufdir0.nparray()
            uvdir = np.array([0.,0.,1.])
            dot = np.dot(uvdir,ufdir)
            if abs(abs(dot) - 1.0) > 1.0e-15:  # need to rotate
                rot_axis = np.cross(uvdir,ufdir)
                rot_angle = np.arccos(dot)
                # rotate vectors by this angle
                R = expm(np.cross(np.eye(3), rot_axis/norm(rot_axis)*rot_angle))
                if useR == 0:
                    return 1, R
                useR = 1
            elif abs(dot + 1.0) < 1.0e-15: # need to rotate by 180 deg
                if useR == 0:
                    return 2, None
                useR = 2
            else:
                if useR == 0:
                    return 3, None
                useR = 3
        if useR == 1: # use R to rotate
            for index in range(len(vecs)):
                vecs[index] = np.dot(R,vecs[index])
        elif useR == 2: # need to rotate by 180 deg
            for index in range(len(vecs)):
                vecs[index][2] = -vecs[index][2]
        # no action required for useR == 3
        return # nothing

    def unit_heading_sample(self,mean=0.0,width=52.,max_angle=180):
        """ Point method: return a random unit vector in a cone centered around the direction of front.
        
        For cylindrical front a zero angle vectors points in the same direction as the front. For spherical fronts the direction is random. Vector is drawn from a normal distribution defined by *mean* ± *width* in degrees. The default values are for cat motor neurons.
        
        Parameters
        ----------
        mean : float > 0.: mean of the normal distribution for angle in degrees, default: 0.
        width : float > 0.: standard deviation of the normal distribution for angle, default: 55.
        max_angle : float > 0.: maximum angle relative to heading, cuts of the distribution,
                          max value is 180 degrees, default: 180 degrees.

        Returns
        -------
        unit length vector : Point
        """
        if (width < 0.) or (width > 90):
            print (Fore.RED + "Error: unit_heading_sample: width should be in range [0-90]",Fore.RESET)
            return
        if (mean < 0.) or (mean > 180):
            print (Fore.RED + "Error: unit_heading_sample: mean should be in range [0-180]",Fore.RESET)
            return
        if (max_angle < 0.) or (max_angle > 180):
            print (Fore.RED + "Error: unit_heading_sample: max_angle should be in range [0-180]",Fore.RESET)
            return
        angle = -1.
        while (angle < 0.0) or (angle > max_angle):
            angle = np.random.normal(mean,width) # take only positive part of distribution
        z = cos(angle * 0.017453292519943)  # convert to radians
        phi = np.random.random() * 6.283185307179586 # 2 * pi
        z_sqrt = sqrt(1.0 - z**2)
        x = z_sqrt * cos(phi)
        y = z_sqrt * sin(phi)
        if not self.is_cylinder():
            return Point(x,y,z)
        # compute the angle of heading with vertical: only for cylindrical
        vecs = [] # need list for _rotate_vectors
        vecs.append(np.array([x,y,z]))
        self._rotate_vectors(None,vecs,-1,None)
        return Point(vecs[0][0],vecs[0][1],vecs[0][2])

    def unit_branching_sample(self,number,mean=45.,width=33.,sep_mean=73.,sep_width=32.):
        """ Point method: generates a list with *number* random unit vectors.
        
        If the front is a cylinder the vectors will have a *mean* ± *width* angle relative to the heading (zero angle) of *self*. The vectors always have a minimal pairwise separation that is larger than a sample of *sep_mean* ± *sep_width*. The default values are for cat motor neurons.
        
        Parameters
        ----------
        number : integer in range 2-20 : number of vectors to return.
        mean : float > 0. : mean of the normal distribution for angle in degrees relative to heading of *self*, default: 45.
        width : float > 0. : standard deviation of the normal distribution for angle, default: 44.
        sep_mean : float > 0.: mean of the normal distribution for separation angle in degrees, default: 73.
        sep_width : float > 0.: standard deviation of the normal distribution for separation angle, default: 32.

        Returns
        -------
        nds_list of unit length vectors : [Point,]
        """
        if (width < 0.) or (width > 90):
            print (Fore.RED + "Error: unit_branching_sample: width should be in range [0-90]",Fore.RESET)
            return
        if (mean < 0.) or (mean > 180):
            print (Fore.RED + "Error: unit_branching_sample: mean should be in range [0-180]",Fore.RESET)
            return
        if (number < 2) or (number > 20):
            print (Fore.RED + "Error: unit_branching_sample: number should be in range [2-20]",Fore.RESET)
            return
        if (sep_mean < 0.) or (sep_mean > 170):
            print (Fore.RED + "Error: unit_branching_sample: sep_mean should be in range [0-170]",Fore.RESET)
            return
        if (sep_width < 0.) or (sep_width > 180):
            print (Fore.RED + "Error: unit_branching_sample: mean should be in range [0-180]",Fore.RESET)
            return
        if self.length() == 0:
            print (Fore.RED + "Error: unit_branching_sample: called by zero length front",Fore.RESET)
            return
        points = nds_list() # points to be returned
        for n in range(number):
            min_sep = np.random.normal(sep_mean,sep_width) # random minimal separation
            if min_sep < 0.0:
                min_sep = 0.0
            k = 0 # number of attempts
            while True:
                if self.is_cylinder(): # only for cylindrical: sample normal distribution
                    angle = -1.
                    # take only positive part of distribution
                    while angle < 0.0:
                        angle = np.random.normal(mean,width) * 0.017453292519943 # convert to radians
                else: # around sphere: sample full uniform distribution
                    angle = np.random.random() * 3.141592653589793 # pi
                z = cos(angle)
                phi = np.random.random() * 6.283185307179586 # 2 * pi
                z_sqrt = sqrt(1.0 - z**2)
                x = z_sqrt * cos(phi)
                y = z_sqrt * sin(phi)
                p = Point(x,y,z)
                if n == 0: # no need to compute minimum separation
                    break # out of while True
                else: # test separation with other points based on random separation
                    success = True
                    for p0 in points:
                        if angle_two_dirs(p,p0) < min_sep:
                            success = False
                            if k > 20: # difficult for this min_sep
                                min_sep -= 10.0  # reduce it
                            break # out of for...
                    k += 1
                    if success:
                        break # out of while True
            points.append(p) # store new vector
        # rotate them to heading of front if needed
        if self.is_cylinder(): # only for cylindrical
            # convert to numpy
            vecs = []
            for p in points:
                vecs.append(p.nparray())
            self._rotate_vectors(None,vecs,-1,None)
            # convert back to points
            points = []
            for vec in vecs:
                points.append(Point(vec[0],vec[1],vec[2]))
        return points
        
    def get_migration_history(self,constellation):
        """ Point method: available after import_simulation only. If front migrated, returns a two lists: list of *front origin coordinates* and list of corresponding *cycles*.
        
            If no data available, empty lists are returned.
        
        Returns
        -------
        Coordinate nds_list and cycle list : [Point,] , [integer, ]
        """
        if constellation._automatic:
            raise TypeError("migration_history can only be called in interactive mode")
        try:
            key = self._key()
            if key in constellation.mig_index:
                index = constellation.mig_index[key]
                return nds_list(constellation.mig_history[index][0]),\
                        constellation.mig_history[index][1]
            else:
                return [],[]
        except:
            return [],[]

    def alternate_locations(self,point,distance,number,random=False):
        """ Point method: generates a list with *number* alternate ``Point`` around *point*.
        
        The alternate points are located on a ring with *distance* as radius around *point*, orthogonal to the axis *self.end* - *point* (cylinder) or *self.orig* - *point* (sphere). Order is clockwise unless optional parameter *random* is True.
        
        Warning: this method does not test whether returned points are inside the simulation volume.
        
        Parameters
        ----------
        point : ``Point`` : center of ring of ``Point``.
        distance : float > 0.: radius of ring of ``Point``.
        number : integer > 0: number of ``Point`` generated. Actual number may be smaller if some of the points would be too close to *self*.
        Optional :
        random : boolean : randomize position of points and order of the list returned, default: False.

        Returns
        -------
        positions around point as nds_list: [Point,]
        """
        if (distance <= 0):
            print (Fore.RED + "Error in alternate_locations: distance should be larger than zero" + Fore.RESET)
            return []
        if (number < 1):
            print (Fore.RED + "Error in alternate_locations: number should be larger than zero" + Fore.RESET)
            return []
        if self.is_cylinder():
            fdir = point - self.end
            if fdir.length() == 0.:
                fdir = point - self.orig
        else:
            fdir = point - self.orig
        if fdir.length() == 0.:
            print (Fore.RED + "Error in alternate_locations: point generates zero length" + Fore.RESET)
            return []
        # compute the rotation matrix relative to heading
        useR, R = self._rotate_vectors(point,[],0,None)
        vecs = [] # initial points as np.arrays
        if random:
            offset = np.random.random() * 6.283185307179586
        else:
            offset = 0.0 # first point is to the right in horizontal plane
        delta = 6.283185307179586 / number  # 2 * pi
        # generate vecs on a circle in the X-Y plane around [0.,0.,0.]
        min_dist = 1.2 * self.radius # minimum distance for proposed point
        for n in range(number): # generate all vecs
            vec = np.array([distance * cos(offset),distance * sin(offset),0.])
            vecs.append(vec)
            offset += delta
        #  rotate them to heading
        self._rotate_vectors(point,vecs,useR,R)
        # generate final positions and check whether they are not too close
        if random:
            np.random.shuffle(vecs)
        points = nds_list() # final points to be returned
        for vec in vecs:
            new_point = point + Point(vec[0],vec[1],vec[2])
            dist = _point_front_dist(new_point,self)
            if dist >= min_dist:
                 points.append(new_point)
        return points
    
    def arc_around(self,constellation,target,arc=90.,arc_size=None,\
                   offset=0.0,new_radius=None):
        """ Point method: returns a list of ``Point`` forming an arc around *target* ``Front``.
        
        To grow around a much larger spherical front an arc needs to be made, this will tend to consists of several short fronts. This method produces a list of positions that can be used to generate these fronts with the ``add_branch`` method. 
        
        The method checks obstructing structures close to *target* to find the optimal orientation of the arc. The number of points returned is determined by two parameters: the *arc* will be computed for any size in the range 1 - 180 degrees, for that given arc the number of points will be decided by whichever is smallest: the length of *self*, which can be overridden by the length parameter *arc_size*, and the resolution needed to have fronts not intersect with the *target*. The arc distance from target center is given by ``1.5 * self.radius + target.radius + offset``.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        target : ``Front`` : spherical front to go around.
        Optional :
        arc : float >=1. and <= 180. : maximal arc size to be produced.
        arc_size : float > 0. : minimum cumulative length of lines connecting the points returned, default 10 µm.
        new_radius : float : use this value as radius of cylinder or sphere, default None (use *self.radius* instead).
        offset : float >= 0. : distance by which the point is offset from target membrane in addition to *self.radius*, default 0.0 µm.

        Returns
        -------
        nds_list of points or empty list : [Point,] or GridCompetitionError
        """
        if constellation.verbose >= 3:
            print ("Process",constellation.my_id,"cycle",constellation.cycle,"arc_around",self,target,arc)
        if target.is_cylinder():
            raise ValueError("target","spherical")
        if (arc < 1.) or (arc > 180.):
            raise ValueError("arc","1 - 180")
        if offset < 0.:
            raise ValueError("offset","larger than zero")
        if not arc_size: # take length of front
            arc_size = self.length()
        if arc_size < 1.: # if it is too small we generate ridiculously many points
            arc_size = 1.
        # radius to be used
        if new_radius:
            if new_radius <= 0.:
                raise ValueError("new_radius","larger than zero")
            radius = new_radius
        else:
            radius = self.radius
        # compute distances
        tot_offset = 2.5 * radius + offset # real offset used: use extra as safety buffer
        min_distance = target.radius + 1.5 * radius + offset # minimum distance allowed
        tar_distance = (self.end - target.orig).length()
        if tar_distance < min_distance: # front end already closer than min_distance
            return [] # impossible to make arc
        p_distance = target.radius + tot_offset # make points at this distance
        # tar_point will be the north pole in rotated space
        if tar_distance < p_distance: # front end close: use this as arc distance
            tar_point = self.end
            if p_distance < tar_distance:
                p_distance = tar_distance
            new_tar = False # start arc from first arc point
        else:
            tar_point = None
            new_tar = True # start arc from tar_point to be computed
        # number of subdivisions needed to get minimum distance
        n_points = angle_sub_divs(arc,p_distance,min_distance,constellation.verbose)
        if constellation.verbose >= 6:
            print ("arc_around n_points",self._fid,arc,p_distance,min_distance,n_points)
        # number of subdivisions needed based on arc_size
        #    0.017453292519943 = 2 * pi / 360
        a_points = int((arc * 0.017453292519943 * p_distance) / arc_size) + 1
        if constellation.verbose >= 6:
            print ("arc_around radii",self._fid,p_distance,target.radius,radius,tot_offset,n_points,a_points)
        # use larger of two values
        if a_points > n_points:
            n_points = a_points
        if n_points == 0:
            return []
        arc_step = arc / n_points # compute this for original n_points
        if n_points > ARCLENGTH:
            n_points = ARCLENGTH
        # compute surface point relative to parent front if needed
        if new_tar: # the closest point
            if self.is_cylinder(): # continue on front heading, not the closest point
                tar_point = target.sphere_intersect(self,offset=tot_offset)
            if not tar_point: # the closest point
                tar_point = target.surface_point_to(self.orig,offset=tot_offset)
            if (self._point_in_self(tar_point)) or \
                    ((tar_point - self.orig).length() < 0.1) or \
                    ((tar_point - self.end).length() < 0.1):
                # will generate InsideParentError
                new_tar = False # do not use this point
        if constellation.verbose >= 6:
            print (constellation.my_id,self.orig,self.end,"arc_around tar_point",tar_point,new_tar)
        # start with random phi for new arc
        # TODO: pick shortest angle if tar_point not at north pole
        #angle = np.random.random() * 360.
        # analyze space around sphere
        #   get all points around target relative to target.orig
        max_radius, pts = target._sphere_cloud(constellation,p_distance + \
                                                2 * radius,radius)
        if constellation.verbose >= 6:
            print ("arc_around _sphere_cloud",nds_list(pts))
        # get R so that tar_point is north pole and remap pts
        free = True
        useR, R, pps = _polar_cloud(tar_point - target.orig,pts,self._fid)
        if constellation.verbose >= 6:
            print ("arc_around _polar_cloud",self._fid,len(pps),useR,max_radius)
        if pps: # obstructing points present: 
            # convert minimum distance 1.25 * (radius + max_radius) to angle
            min_angle = 71.619724391352901 * (radius + max_radius) / target.radius # 1.25 * 360 / (2 * pi)
            # add min_angle to max arc:
            max_angle = arc + min_angle
            if max_angle > 180.:
                max_angle = 180.
            if constellation.verbose >= 6:
                print ("arc_around angles",self._fid,arc_step,min_angle,max_angle)
            # find angle that is furthest removed from all phi's in pps
            for p in pps: # check whether north pole is free
                if p.theta < min_angle: # is tar_point free?
                    free = False
            phis = []
            # collect phis
            if not free: # do not include north pole in check
                free_angle = min_angle * 0.7 # close to min_angle may collide
                for p in pps:
                    if (p.theta > free_angle) and (p.theta <= max_angle):
                        phis.append(p.phi)
            else: # north pole is free, check everything
                for p in pps:
                    if p.theta <= max_angle:
                        phis.append(p.phi)
            best = 0.
            # coarse search
            for angle in range(0,370,10):
                closest = 360.
                for phi in phis:
                    dist = abs(phi - angle)
                    if dist < closest:
                        closest = dist
                if closest > best:
                    best = closest
                    best_angle = angle
            # finer search
            for angle in range(best_angle - 5,best_angle + 5,1):
                closest = 360.
                for phi in phis:
                    dist = abs(phi - angle)
                    if dist < closest:
                        closest = dist
                if closest > best:
                    best = closest
                    best_angle = angle
            if best_angle < 0.:
                best_angle = 360. + best_angle
            if best_angle > 360.:
                best_angle = best_angle - 360.
        else: # random angle
            best_angle = np.random.random() * 360.

        # Compute arc subdivisions in polar coordinates
        # compute first point
        if new_tar: # tar_point is part of arc
            if free:
                pps = [Pol_point(p_distance, 0., best_angle)]
                theta = 0
            else:
                pps = [Pol_point(p_distance, min_angle, best_angle)]
                theta = min_angle
        else:
            pps = []
            theta = 0.
        if constellation.verbose >= 6:
            print ("arc_around angle",self._fid,best_angle,arc_step,useR)
        for i in range(n_points):
            theta += arc_step
            if theta > 180.: # wrap around
                theta = 360. - theta
                #angle -= 180.
            pp = Pol_point(p_distance, theta, best_angle)
            pps.append(pp)
        # convert to Point and rotate back to real north pole
        points = nds_list()
        if useR == 1:
            for pp in pps:
                p = pp.toPoint()
                rot_np = np.dot(R.T,p.nparray())
                if constellation.verbose >= 6:
                    print ("arc_around point",self._fid,pp,p,rot_np,target.orig + rot_np)
                ap = target.orig + rot_np
                if self._point_in_self(ap): # possible for initial point
                    continue
                points.append(ap)
        elif useR == 2:
            for pp in pps:
                pp.theta = 180. - pp.theta # flip theta
                ap = pp.toPoint() + target.orig
                if self._point_in_self(ap): # possible for initial point
                    continue
                points.append(ap)
        else: # useR == 3
            for pp in pps:
                ap = pp.toPoint() + target.orig
                if self._point_in_self(ap): # possible for initial point
                    continue
                points.append(ap)
        # test whether first point is legal
        count = 0
        gids = constellation._get_gids(self.end,points[0],0.)
        while True: # loop for GridCompetitionError
            count += 1
            if count > 100:
                raise GridCompetitionError(cgid)
            try: 
                constellation._test_collision(self.end,points[0],radius,\
                                                        self,gids,lock=False)
                break # out of while True
            except GridCompetitionError as error:
                cgid = error.gid
                time.sleep(2*LOCKPAUSE) # competition with other processor
                continue # while True
            except (CollisionError,InsideParentError,VolumeError):
                if constellation.verbose >= 6:
                    print ("arc_around failed on CollisionError",self._fid)
                return []
        # now compute how many points to return (arc size for this cycle)
        size = 0.
        count = 0
        if self.is_cylinder():
            prev_p = self.end
        else:
            prev_p = self.orig
        for p in points:
            size += (p - prev_p).length()
            count += 1
            if (size > arc_size) and (count > 1):
                break
            else:
                prev_p = p
        # store the arc
        index = constellation._store_arc(self,target,arc,count,points) # global
        key = self._key()
        if key in constellation._used_arcs: # local and temporary
            constellation._used_arcs[key].append(index)
        else:    
            constellation._used_arcs[key] = [index]
        if constellation.verbose >= 6:
            print ("arc_around returns",self._fid,count,n_points,nds_list(points))
        return nds_list(points[:count])
            
    def arc_continue(self,constellation,collision=False):
        """ Point method: returns a list of ``Point`` continuing an arc ending at self.end.
        
        Returns empty list if no previous arc is found. The method performs NO collision detection. The number of points returned is either the same as before or the remainder of the arc.
        
        If ``collission=True`` it will extend the previous arc if no points on it remain.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        Optional:
        collision : boolean : called by ``solve_collision`` method, default False.

        Returns
        -------
        nds_list of points or empty list : [Point,]
        """
        cycle = constellation.cycle
        if constellation.verbose >= 3:
            print (cycle,"arc_continue",self)
        if not self.is_arc():
            return []
        # find the Arc used:
        index = constellation._get_arc(self)
        if index < 0: # not found
            print (Fore.RED,constellation.my_id,"no arc found",self._fid,self.end,self.birth,Fore.RESET)
            return []
        arc_used = constellation._arcs[index]
        # get available points:
        if constellation.verbose >= 6:
            print (cycle,"arc_continue",arc_used)
        points = nds_list()
        if (arc_used.complete > 0): # no points left
            if not collision:
                return []
           # return two newly computed points, these will not be labeled as arc
            num_points = arc_used.num_points
            a_div = 1. / num_points
            p0 = constellation._arc_points[arc_used.index]
            p1 = constellation._arc_points[arc_used.index + num_points - 1]
            for n in range(1,3):
                p = arc_used.sphere.sphere_interpol(p0,p1,scale=1. + n * a_div)
                points.append(p)
        else:
            av_points = arc_used.num_points - arc_used.next_point
            # find how many points we need to return
            if av_points <= arc_used.count:
                n_points = av_points # finish the arc
            else: # base on previous use
                n_points = arc_used.count
            for n in range(n_points):
                points.append(constellation._arc_points\
                                  [arc_used.index + arc_used.next_point + n])
        if constellation.verbose >= 6:
            print ("arc_continue returns",self._fid,n_points,nds_list(points))
        return points

    def go_around(self,constellation,point,target,offset=0.0,new_radius=None):
        """ Point method: returns a list of ``Point`` going around a cylindrical *target* ``Front`` without making an arc.
         
        Parameters
        ----------
        constellation : ``Constellation`` object.
        point : ``Point`` : the point that caused the collision, used to compute desired growth direction.
        target : ``Front`` : front to go around.
        Optional :
        new_radius : float : use this value as radius of cylinder or sphere, default None (use *self.radius* instead).
        offset : float >= 0. : distance by which the point is offset from target membrane in addition to *self.radius*, default 0.0 µm.

        Returns
        -------
        nds_list of points or empty list : [Point,] or GridCompetitionError
        """
        if constellation.verbose >= 3:
            print (self._fid,"go_around",target._fid)
        if not target.is_cylinder():
            return []
        if offset < 0.:
            return []
        # radius to be used
        if new_radius:
            if new_radius <= 0.:
                return []
            radius = new_radius
        else:
            radius = self.radius
        # compute distances
        tot_offset = 2 * radius + offset # real offset used: use extra radius as safety buffer
        min_distance = target.radius + tot_offset # minimum distance used
        # get closest point on colliding front
        dist,p = self.front_distance(target,point=True)
        # vector towards colliding front and vector of desired growth
        if self.is_cylinder():
            s_end = self.end
        else:
            s_end = self.orig
        connect = p - s_end
        g_dir = point - s_end
        c_len = connect.length()
        g_len = g_dir.length()
        # compute correct length direction vector for final growth past colliding front
        if c_len < g_len * 0.5: # compute test length for last segment
            f_dir = g_dir.norm() * (g_len - c_len) # remaining growth length past colliding front
        else:
            f_dir = g_dir * 0.5 # take decent remaining length
        # get normal to target line and connect
        t_dir = target.end - target.orig
        normal = (t_dir).cross(connect)
        # give it the proper length
        norm_l = normal.norm() * min_distance
        # positioning controls
        pos_diff = 0 # incremental distance from p along target
        t_ndir = t_dir.norm() # normalized direction of colliding front
        t_len_2 = target.length() * 0.5 # maximum distance from p explored
        if t_len_2 > 5 * radius: # compute incremental step
            step = 2 * radius
        else:
            step = radius
        # using the normal, first find point next to colliding front from which further
        #   growth is possible, do this relative to closest point
        while pos_diff < t_len_2:
            # test two points: before and after p
            if constellation.verbose >= 6:
                print (self._fid,target._fid,"go_around",p,t_ndir*pos_diff,norm_l)
            for k in range(2):
                if k == 0:
                    tpos = p - t_ndir * pos_diff
                else:
                    tpos = p + t_ndir * pos_diff
                # test two points: one on each side of colliding front
                mult = 1.0 
                for l in range(4):
                    if l // 2 == 0:
                        t_point = tpos + (norm_l * mult)
                    else:
                        t_point = tpos - (norm_l * mult)
                        mult += 0.5
                    result = t_point.out_volume(constellation)
                    if result != 0.:
                        continue
                    if self._point_in_self(t_point):
                        continue
                    # test whether t_point is valid
                    if constellation.verbose >= 6:
                        print (self._fid,target._fid,"go_around testing t_point",t_point)
                    use_r_point = False # no extra point between s_end and t_point
                    # test segment s_end to t_point
                    failed = 0
                    count = 0
                    while True: # loop for GridCompetitionError
                        count += 1
                        if count > 100:
                            raise GridCompetitionError(cgid)
                        try: 
                            gids = constellation._get_gids(s_end,t_point,0.)
                            constellation._test_collision(s_end,t_point,radius,\
                                                        None,gids,lock=False)
                            break # out of while True
                        except GridCompetitionError as error:
                            cgid = error.gid
                            time.sleep(2*LOCKPAUSE) # competition with other processor
                            continue # while True
                        except CollisionError as error:
                            # test whether colliding with self or its children
                            if constellation.verbose >= 6:
                                print (t_point,"t_point collides with:",error.collider)
                            if target == error.collider:
                                failed = 1
                            else:
                                failed = 2
                            break # out of while True
                    if failed == 2:  # s_end to t_point not valid
                        continue
                    elif failed == 1: # collision with colliding front
                        # try to interpose extra point so colliding front
                        #   is approached vertically
                        if l < 2:
                            r_point = s_end + norm_l
                        else:
                            r_point = s_end - norm_l
                        result = r_point.out_volume(constellation)
                        if result != 0.:
                            continue
                        if self._point_in_self(r_point):
                            continue
                        # test segment s_end to r_point
                        failed = False
                        count = 0
                        while True: # loop for GridCompetitionError
                            count += 1
                            if count > 100:
                                raise GridCompetitionError(cgid)
                            try: 
                                gids = constellation._get_gids(s_end,r_point,0.)
                                constellation._test_collision(s_end,r_point,\
                                                    radius,None,gids,lock=False)
                                break # out of while True
                            except GridCompetitionError as error:
                                cgid = error.gid
                                time.sleep(2*LOCKPAUSE) # competition with other processor
                                continue # while True
                            except CollisionError as error:
                                if constellation.verbose >= 6:
                                    print (r_point,"r_point collides with:",error.collider)
                                failed = True
                                break
                        if failed:
                            continue # s_end to r_point not valid
                        # test segment r_point to t_point
                        failed = False
                        count = 0
                        while True: # loop for GridCompetitionError
                            count += 1
                            if count > 100:
                                raise GridCompetitionError(cgid)
                            try: 
                                gids = constellation._get_gids(r_point,t_point,0.)
                                constellation._test_collision(r_point,t_point,\
                                                    radius,None,gids,lock=False)
                                break # out of while True
                            except GridCompetitionError as error:
                                cgid = error.gid
                                time.sleep(2*LOCKPAUSE) # competition with other processor
                                continue # while True
                            except CollisionError as error:
                                if constellation.verbose >= 6:
                                    print (r_point,t_point,"r_point-t_point collides with:",error.collider)
                                failed = True
                                break
                        if failed:
                            continue # r_point to t_point not valid
                        use_r_point = True # r_point works
                    f_point = t_point + f_dir # final point of growth
                    result = f_point.out_volume(constellation)
                    if result != 0.:
                        continue
                    # test segment t_point to f_point
                    failed = False
                    count = 0
                    while True: # loop for GridCompetitionError
                        count += 1
                        if count > 100:
                            raise GridCompetitionError(cgid)
                        try: 
                            gids = constellation._get_gids(t_point,f_point,0.)
                            constellation._test_collision(t_point,f_point,\
                                                radius,None,gids,lock=False)
                            break # out of while True
                        except GridCompetitionError as error:
                            cgid = error.gid
                            time.sleep(2*LOCKPAUSE) # competition with other processor
                            continue # while True
                        except CollisionError as error:
                            if constellation.verbose > verb4:
                                if constellation.verbose >= 6:
                                    print (t_point,f_point,"t_point-f_point collides with:",error.collider)
                            failed = True
                            break
                    if failed:
                        continue # t_point to f_point not valid
                    if use_r_point:
                        if constellation.verbose >= 6:
                            print (self._fid,target._fid,"go_around valid r_point,t_point,f_point",r_point,t_point,f_point)
                        return nds_list([r_point,t_point,f_point])
                    else:
                        if constellation.verbose >= 6:
                            print (self._fid,target._fid,"go_around valid t_point,f_point",t_point,f_point)
                        return nds_list([t_point,f_point])
            pos_diff += step
        if constellation.verbose >= 6:
            print (self._fid,target._fid,"go_around failed, returns []")
        return [] # failed
                
    def surface_point_to(self,point,mid=True,offset=0.0,pos=None):
        """ Point method: returns ``Point`` on the membrane surface of *self* in the direction of *point*.
        
        If optional parameter *mid* is True this point is halfway along the length of the cylinder, else it is a random location. This can be overruled with *pos* to specify a specific position along the length: between 0.0 (*self.orig*) and 1.0 (*self.end*), ``pos=0.5`` corresponds to mid=True. For a sphere it is always the closest point on the surface. Optional parameter *offset* gets added to the radius of *self* and results in a point at distance offset from its membrane surface.
            
        Parameters
        ----------
        point : ``Point`` : gives direction relative to *self*.
        Optional :
        mid : boolean : point is in the center of the cylinder, else it is a
                      random location. For a sphere it is always the closest
                      point on the surface. Default: True.
        offset : float >= 0. : distance by which the point is offset from the membrane
                       along the line connecting to *point*, default 0.0 µm.
        pos : float 0.0 - 1.0 : overrule mid setting with relative position, default None.

        Returns
        -------
        location on or close to membrane surface of *self* : Point
        """
        if self.is_cylinder():
        # Code based on https://www.geometrictools.com/Documentation/DistanceToCircle3.pdf
            vec = self.end - self.orig
            vecn = vec.norm()
            if pos:
                if pos < 0.0:
                    pos = 0.0
                elif pos > 1.0:
                    pos = 1.0
                newp = self.orig + (vec * pos)
            elif mid:
                newp = self.orig + (vec * 0.5)
            else:
                newp = self.orig + (vec * np.random.random())
            vecp = point - newp
            # projection of xyz on plane of circle
            proj = vecp - (vecn * vecn.dot(vecp))
            lproj = proj.length() # length of this vector
            if lproj > 0.:
                return newp + (proj * ((self.radius + offset) / lproj))
            else: # point is on axis of cylinder: return fixed location on circle
                ovec = _orthogonal(vecn)
                return newp + (ovec * (self.radius + offset))
        else: # sphere
            return self._sphere_surf(point,offset=offset)

    def sphere_interpol(self,point1,point2,scale=0.5):
        """ Point method: returns a point on or close to the membrane surface of spherical *self* on the arc connecting *point1* and *point2*.
        
        Optional parameter *scale* changes location of ``Point``: from *point1* for ``scale=0`` to *point2* for ``scale=1``. If the points are not on the surface it is assumed that they have the same distance to *self.orig*.
            
        Parameters
        ----------
        point1 : ``Point`` : (close to) surface point.
        point2 : ``Point`` : (close to) surface point.
        Optional :
        scale : float : in range 0 - 1.0, default 0.5 (mid-point).

        Returns
        -------
        location on or close to membrane surface of *self* : Point
        """
        if self.is_cylinder():
            return point1
        # turn into points on unit sphere
        p1 = point1 - self.orig
        radius = p1.length() # true radius of points, assume the same for both
        p1 = p1.norm()
        p2 = (point2 - self.orig).norm()
        # angle between the two points
        omega = angle_two_dirs(p1,p2,degree=False)
        if omega == 0.: # p1 == p2
            return p2
        sin_omega = sin(omega)
        # use Slerp interpolation to compute the new point
        p0 = p1 * (sin((1 - scale) * omega) / sin_omega) + \
                p2 * (sin(scale * omega) / sin_omega)
        # map point onto sphere
        return self.orig + p0 * radius
        
    def sphere_intersect(self,front,offset=0.0):
        """ Point method: return closest point along the heading of front that intersects the membrane surface of spherical *self*.
        
        Parameters
        ----------
        front : ``Front`` : must be a cylinder.
        Optional :
        offset : float >= 0. : distance by which the point is offset from the membrane
                       along the line connecting to *front*, default 0.0 µm.

        Returns
        -------
        closest intersection point : Point or None
        """
        # Modified from http://paulbourke.net/geometry/circlesphere/sphere_line_intersection.py
        if not front.is_cylinder(): # does not work for two spheres
            return None
        radius = self.radius + offset
        # find closest point to sphere: this will become P1
        dist1 = (front.orig - self.orig).length()
        dist2 = (front.end - self.orig).length()
        """
        if dist2 < radius: # front is already closer than desired distance
            return front.end
        """
        if dist1 < dist2:
            P1 = front.orig
            dir = front.end - front.orig
        else:
            P1 = front.end
            dir = front.orig - front.end
        a = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z
        if a == 0.: # zero length front
            return None
        b = 2.0 * (dir.x * (P1.x - self.orig.x) + dir.y * (P1.y - self.orig.y) \
                   + dir.z * (P1.z - self.orig.z))
        c = self.orig.x * self.orig.x + self.orig.y * self.orig.y + \
            self.orig.z * self.orig.z + P1.x * P1.x + P1.y * P1.y + P1.z * P1.z \
            - 2.0 * (self.orig.x * P1.x + self.orig.y * P1.y + self.orig.z * P1.z) \
            - radius * radius
        i = b * b - 4.0 * a * c
        if i < 0.0:    # no intersection
            return None
        elif i == 0.0: # one intersection
            mu = -b / (2.0 * a)
        else: # two intersections
            # find closest: corresponds to smallest mu
            mu1 = (-b + sqrt(i)) / (2.0 * a)
            mu2 = (-b - sqrt(i)) / (2.0 * a)
            if mu1 < mu2:
                mu = mu2
            else:
                mu = mu1
        return Point(P1.x + mu * dir.x, P1.y + mu * dir.y, P1.z + mu * dir.z)
        
    def sphere_T_points(self,point1,point2):
        """ For sphere with center and two points on or close to its surface, generate two more points on circle perpendicular to arc made by point1 and point2, resulting in a T with the vertical defined by point1 and point2.
    
        Parameters
        ----------
        point1 : ``Point`` : (close to) surface point.
        point2 : ``Point`` : (close to) surface point.

        Returns
        -------
        nds_list of pair of points at same distance from point2 : [Point,Point]
        """
        if self.is_cylinder():
            return []
        radius = (point1 - self.orig).length() # true radius of points, assume the same for both
        short_side = point2 - point1
        normal = (point2 - self.orig).cross(short_side)
        scaled_normal = normal.norm() * short_side.length()
        pl = point2 - scaled_normal
        pr = point2 + scaled_normal
        # these will be too far out -> rescale
        pl = self.orig + (pl - self.orig).norm() * radius
        pr = self.orig + (pr - self.orig).norm() * radius
        return nds_list([pl, pr])
        
    # return mid point on cylinder axis or origin if sphere
    #  or any other point on cylinder axis if scale is specified
    def mid(self,scale=0.5):
        """ Point method: returns center point on the axis of the cylinder.
        
        Optional parameter *scale* changes location of ``Point``: from *self.orig* for ``scale=0`` to *self.end* for ``scale=1``. For spheres the origin is returned.
            
        Parameters
        ----------
        Optional :
        scale : float : in range 0 - 1.0, default 0.5 (mid-point).

        Returns
        -------
        location on axis of cylindrical *self* : Point
        """
        if self.is_cylinder():
            vec = self.end - self.orig
            return self.orig + vec * scale
        else:
            return self.orig

    # return length of cylinder or diameter of sphere
    def length(self):
        """ Size method: returns length of *self*.
        
        Length of cylinder or diameter of sphere.
            
        Returns
        -------
        length of *self* in µm : float
        """
        if self.is_cylinder():
            vec = self.end - self.orig
            return vec.length()
        else:
            return 2. * self.radius

    # return minimal distance to other Front or Substrate
    def front_distance(self,other,point=False):
        """ Size method: returns shortest distance between *self* and a ``Front``, ``Point`` or ``Substrate``.
        
        The distance is computed for the coordinates of the stuctures, not for their extent in space as defined by a radius.
        
        Parameters
        ----------
        other : ``Front``, ``Point`` or ``Substrate``.
        Optional:
        point : boolean : return also closest point on other, default=False

        Returns
        -------
        distance in µm : float
        if point=True; distance,point : float, Point
        """
        if self.is_cylinder():
            if other.is_cylinder():
                if point:
                    dist,p1,p2 = dist3D_cyl_to_cyl(self.orig,self.end,other,points=True)
                    return dist,p2
                else:
                    return dist3D_cyl_to_cyl(self.orig,self.end,other)
            else:
                if isinstance(other,Point):
                    dist = dist3D_point_to_cyl(other,self.orig,self.end)
                    if point:
                        return dist,other
                    else:
                        return dist
                else:
                    dist = dist3D_point_to_cyl(other.orig,self.orig,self.end)
                    if point:
                        return dist,other.orig
                    else:
                        return dist
        else:
            if other.is_cylinder():
                return dist3D_point_to_cyl(self.orig,other.orig,other.end,points=point)
            else:
                if isinstance(other,Point):
                    dist = (self.orig - other).length()
                    if point:
                        return dist,other
                    else:
                        return dist
                else:
                    dist = (self.orig - other.orig).length()
                    if point:
                        return dist,other.orig
                    else:
                        return dist

    # taper radius by given fraction but do not go below minimum
    # returns: float: new radius
    def taper(self,fraction,minimum=0.3):
        """ Size method: returns a tapered *radius*.
        
        Returns *self.radius* scaled by *fraction*, with value larger or equal to optional parameter *minimum*.
        
        Parameters
        ----------
        fraction : float : relative scaling of *radius*.
        Optional:
        mimimum : float : mimimal *radius*, default 0.3 µm.

        Returns
        -------
        new radius in µm : float
        """
        rad = self.radius * fraction
        if rad < minimum:
            rad = minimum
        return rad
    
    # update order of all children, can be called iteratively in case where
    #  first child was made by add_branch
    def _update_order(self,constellation,new_order):
        children = self.get_children(constellation)
        for c in children: # only update children grown this cycle
            if c.order != new_order:
                # interstitial growth will increase order only for oblique dendrite
                if c.birth == constellation.cycle:
                    if c.order != self.order:
                        raise BugError("_update_order","order wrong value")
                    c.order = new_order
                    c._update_order(constellation,new_order)
                
    # get grid ids that will need to be updated for this front
    # if _gid==0 a new list is made, otherwise the list is taken from self._gids
    # returns a list of gid (integers)
    def _get_gids(self,constellation):
        if self._gid:
            n = abs(self._gid)
            gids = []
            for i in range(n + 1, n + 1 + constellation._gids[n]):
                gids.append(constellation._gids[i])
            return gids
        else:
            if self.is_cylinder():
                return constellation._get_gids(self.orig,self.end,0.)
            else:
                return constellation._get_gids(self.orig,None,self.radius)

    # store grid ids in self._gids
    # self._gids entries:
    #   index 1: length of entry
    #   index 2: first gid
    #   index 3: second gid
    #   ...
    # front._gid points to index 1
    def _store_gids(self,constellation,gids):
        if constellation._gid_next >= constellation._gid_max:
            raise OverflowError("_gids","grid_extra")
        _gid = constellation._gid_next
        constellation._gids[_gid] = len(gids)
        constellation._gid_next += 1
        for gid in gids:
            if constellation._gid_next >= constellation._gid_max:
                raise OverflowError("_gids","grid_extra")
            constellation._gids[constellation._gid_next] = gid
            constellation._gid_next += 1
        self._gid = _gid

    # Method to analyze space occupation around a sphere.
    # Now also checks for volume borders: these are also encoded as points
    # Returns:
    # max_radius, [Point, ] : maximum front radius found, followed by a nds_list
    #         of points relative to self.orig, representing front coordinates 
    #         around the sphere within distance relative to origin.
    def _sphere_cloud(self,constellation,distance,resolution):
        if self.is_cylinder():
            return []
        volume = constellation._volume
        pts = nds_list() # list to be returned
        # check for volume borders
        sim_volume = volume.sim_volume
        if self.orig.x - sim_volume[0][0] < distance: # close to left border
            self._add_plane(pts,\
                    Point(sim_volume[0][0],self.orig.y,self.orig.z),\
                    self.radius,resolution,[0,1,1])
        elif sim_volume[1][0] - self.orig.x < distance: # close to right border
            self._add_plane(pts,\
                    Point(sim_volume[1][0],self.orig.y,self.orig.z),\
                    self.radius,resolution,[0,1,1])
        if self.orig.y - sim_volume[0][1] < distance: # close to left border
            self._add_plane(pts,\
                    Point(self.orig.x,sim_volume[0][1],self.orig.z),\
                    self.radius,resolution,[1,0,1])
        elif sim_volume[1][1] - self.orig.y < distance: # close to right border
            self._add_plane(pts,\
                    Point(self.orig.x,sim_volume[1][1],self.orig.z),\
                    self.radius,resolution,[1,0,1])
        if self.orig.z - sim_volume[0][2] < distance: # close to left border
            self._add_plane(pts,\
                    Point(self.orig.x,self.orig.y,sim_volume[0][2]),\
                    self.radius,resolution,[1,1,0])
        elif sim_volume[1][2] - self.orig.z < distance: # close to right border
            self._add_plane(pts,\
                    Point(self.orig.x,self.orig.y,sim_volume[1][2]),\
                    self.radius,resolution,[1,1,0])
       # check for structures close to sphere
        max_radius = 0.
        gid0 = self.orig._grid(volume)
        gids = [gid0] + constellation._get_sphere_gids(self.orig,distance)
        ids = [self.get_id()] # all processed ids
        my_id = constellation.my_id
        while len(gids) > 0: # loop through gids till all are processed
            for gid in gids: # loop over all grid points
                if constellation._automatic:
                    if constellation._grid_rlock[gid] == 0: # get brief read-only lock
                        constellation._grlock_request[my_id] = gid
                        wait = 0.
                        while constellation._grid_rlock[gid] != my_id:
                            time.sleep(LOCKPAUSE)
                            wait += LOCKPAUSE
                            if wait > 2.0:
                                raise BugError("_sphere_cloud","read-only waited for two seconds on " + str(my_id) + " for grid " + str(gid))
                    else:
                        continue # for loop
                else: # interactive mode
                    self._grid_rlock[gid] = 1
                new_ids = constellation._grid_get(gid)
                constellation._grid_rlock[gid] = 0 # unlock it after use
                constellation._grid_lock_wait += wait
                gids.remove(gid)
                for id in new_ids: # loop over all ids for this grid point
                    if id._nid == 0:
                        print (Fore.RED + "_sphere_cloud got zero ID " + str(gid) + Fore.RESET)
                        continue
                    if id not in ids: # not processed yet
                        ids.append(id)
                        front = constellation.front_by_id(id)
                        if front.is_cylinder(): # real distance
                            dist,p = self.front_distance(front,point=True)
                        else: # sphere: distance to surface
                            sur_p = front.surface_point_to(self.orig)
                            dist = self.front_distance(sur_p)
                        if dist <= distance: # within desired range
                            if front.is_cylinder():
                                # check front coordinates for distance
                                rel_o = front.orig - self.orig # relative to sphere origin
                                if rel_o.length() <= distance:
                                    #print ("_sphere_cloud rel_o",front.orig,rel_o)
                                    if constellation.verbose >= 7:
                                        print (self._fid,"_sphere_cloud adding o",id,dist,rel_o)
                                    pts.append(rel_o)
                                if front.radius > max_radius: # not for spheres
                                    max_radius = front.radius
                                rel_e = front.end - self.orig # relative to sphere origin
                                if rel_e.length() <= distance:
                                    if constellation.verbose >= 7:
                                        print (self._fid,"_sphere_cloud adding e",id,dist,rel_e)
                                    pts.append(rel_e)
                                # check whether closest point should be included
                                if ((p - front.orig).length() > resolution) and \
                                    ((p - front.end).length() > resolution):
                                    rel_p = p - self.orig
                                    if constellation.verbose >= 7:
                                        print (self._fid,"_sphere_cloud adding p",id,dist,p,rel_p)
                                    pts.append(rel_p)
                                elif front.length() > resolution: # check for mid point
                                    rel_m = front.mid() - self.orig
                                    if constellation.verbose >= 7:
                                        print (self._fid,"_sphere_cloud adding m",id,dist,rel_m,front.mid(),front.mid() - self.orig)
                                    pts.append(rel_m)
                            else: # for a sphere generate surface points <= distance
                                p_sur = (sur_p - front.orig).toPol_point() # convert to polar coordinates
                                r = p_sur.r # store the coordinates
                                theta = p_sur.theta
                                phi = p_sur.phi
                                sep_a = resolution * 57.295779513082321 / front.radius # convert into angle
                                # make list of phi to use and test maximum angle to generate
                                new_p = [phi]  # list to use
                                step_a = sep_a # incremental increase of phi
                                max_step = 180.
                                while step_a < max_step:
                                    # go left
                                    np = phi - step_a
                                    if np < 0.:
                                        np += 360.
                                    new_p.append(np)
                                    pp = Pol_point(r, theta, np) # make new polar point
                                    p = pp.toPoint() + front.orig # convert to cartesian
                                    pts.append(p - self.orig) # store it
                                    # go right
                                    np = phi + step_a
                                    if np >= 360.:
                                        np -= 360.
                                    new_p.append(np)
                                    pp = Pol_point(r, theta, np)
                                    p = pp.toPoint() + front.orig
                                    pts.append(p - self.orig) # store it
                                    # test distance
                                    dist = self.front_distance(p)
                                    if dist > distance: # reached max
                                        max_step = step_a
                                        break  # out of while step_a
                                    else:
                                        step_a += sep_a
                                    if constellation.verbose >= 7:
                                        print ("_sphere_cloud",self._fid,front._fid,phi,sep_a,np,dist,distance)
                                # generate theta on the fly
                                step_a = sep_a # incremental increase of theta
                                max_step = min(90., max_step)
                                while step_a < max_step:
                                    new_t = theta - step_a
                                    if new_t < 0.:
                                        new_t += 180.
                                    for np in new_p:
                                        pp = Pol_point(r, new_t, np)
                                        p = pp.toPoint() + front.orig
                                        pts.append(p - self.orig) # store it
                                    new_t = theta + step_a
                                    if new_t > 180.:
                                        new_t -= 180.
                                    for np in new_p:
                                        pp = Pol_point(r, new_t, np)
                                        p = pp.toPoint() + front.orig
                                        pts.append(p - self.orig) # store it
                                    step_a += sep_a
                break # for loop, changed gids
        return max_radius, pts
       
    # adds a points to list points on a plane of length * length with resolution around center in dimensions encoded in list dim
    def _add_plane(self,points,center,length,resolution,dim):
        points.append(center)
        nstep = length / resolution + 1
        for d in range(3): # check dimensions
            if dim[i] > 0:
                step = dimensions[d] * resolution
                for n in range(1,nstep):
                    p = center + step
                    points.append(p)
                    p = center - step
                    points.append(p)
        
    # are new_front and front related as siblings or nephews?
    # self is the parent of new_front
    # returns boolean: True if related
    def _is_related(self,constellation,front,trailing_axon):
        if self._nid != front._nid: # different neurons -> never related
            return False
        fid = self._fid
        # are they siblings: is self common parent?
        if fid == front._pid:
            return True
        # are self and front identical?
        if fid == front._fid:
            return True
        # allow migrating soma to collide with any of its filipod or trailing axon children
        if self.is_migrating() and (front._sid == self._fid):
            if front.swc_type == 12:
                return True
            # allow collision with recent trailing axon
            if trailing_axon and (front.swc_type == 2):
                parent = front.get_parent(constellation)
                if parent._pid == self._fid: # we are comparing to grandparent
                    return True
                elif constellation.cycle - front.birth < 5: # possible great-grandparent
                    return True
        elif self.swc_type == 12:
            if front.is_migrating() and (self._sid == front._fid):
                return True
        return False

    # Checks whether point is not inside self (the parent)
    # Returns boolean
    def _point_in_self(self,point):
        if self.is_cylinder():
            dist, nearest = \
                    dist3D_point_to_cyl(point,self.end,self.orig,points=True)
            if dist <= self.radius: # possibly inside cylinder
                # if nearest is self.orig or self.end then point is just outside of cylinder
                if (not (nearest == self.orig)) and (not (nearest == self.end)):
                    return True
        else: # spherical parent
            dist = (point - self.orig).length()
            if dist <= self.radius: # inside sphere
                return True
        return False

def average_fronts(front_list):
    """
    Compute average position over all fronts in front_list.

    Parameters
    ----------
    front_list: list: either list of fronts or list returned by get_fronts

    Returns
    -------
    average position: Point
    """

    gf_list = isinstance(front_list[0],tuple) # first detect type of list
    sum = Point(0.,0.,0.) # initialize to zero
    for item in front_list:
        if gf_list:  # item is (front, distance)
            front = item[0]
        else:
            front = item
        if front.is_cylinder():
            sum += front.end
        else:
            sum += front.orig
    return sum / len(front_list)

class SynFront(Front):
    """ Front class that implements synaptic connections.
    
    Current implementation allows only one synaptic connection per front, the front is  either pre- or postsynaptic. Neurons based on ``SynFront`` have a *firing_rate* (default: 1) and a *CV_ISI* (default 0.) defined that can be accessed with methods.
    
    This class is not meant to be instantiated, instead subclass it with a model specific ``manage_front`` method.
    
    Only the **attributes** listed below can be read in addition to ``Front`` attributes. NEVER change these attributes.

    Attributes:
        syn_input : float : current synaptic input, always 0. for presynaptic front, updated for active postsynaptic front.
    """
    
    _fields_ = Front._fields_ + [('syn_input', c_double), ('_yid', c_int)]
    
    def nds_class(self):
        """ Return class name.
         
        Returns
        -------
        Returns
        -------
        Class name of SynFront subclass : string
        """
        name = str(self.__class__)
        return name[name.rfind('.') + 1 : -2]

    # _yid: abs(_yid) > 0 is index to _synapses. Is negative for presynaptic 
    #        front, positive for postsynaptic one

    def add_synapse(self,constellation,other_front,weight,presynaptic=True):
        """ Make a synaptic connection with another front.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        other_front : ``SynFront`` : postsynaptic (default) or presynaptic front that is also part of this synapse.
        weight : float : synaptic weight. Positive for excitation, negative for inhibition.
        Optional:
        presynaptic : boolean : *self* is the presynaptic front, default True.
        """
        if self._yid != 0:
            raise SynapseError("add_synapse: self already has a synaptic connection")
        if not isinstance(other_front,SynFront):
            raise SynapseError("add_synapse: other_front is not a SynFront")
        if other_front._yid != 0:
            raise SynapseError("add_synapse: other_front already has a synaptic connection")
        if self.front_distance(other_front) > 5.:
            raise SynapseError("add_synapse: other_front is too far away")
        if self.is_migrating() or other_front.is_migrating():
            raise SynapseError("add_synapse: no synapse possible between migrating fronts")
        if constellation._n_newAID >= constellation._new_range:
            raise OverflowError("_new_AIDs","max_active")
        if constellation._syn_next >= constellation._syn_max:
            raise OverflowError("_synapses","max_synapse")
        if constellation.verbose >= 3:
            print ("Process",constellation.my_id,"cycle",constellation.cycle,"adding synapse",self._fid)
        # make synapse and update its attributes
        if presynaptic:
            syn = Synapse(self.get_id(),other_front.get_id(),weight)
            neuron = self.get_neuron(constellation)
        else:
            syn = Synapse(other_front.get_id(),self.get_id(),weight)
            neuron = other_front.get_neuron(constellation)
        # initialize
        syn._syn_update(neuron.firing_rate,neuron.CV_ISI,weight=weight)    # 
        # store synapse in synapses
        yid = constellation._syn_next
        constellation._synapses[yid] = syn
        # update _yid in both fronts and initialize syn_input in postsynaptic front
        constellation.lock(other_front) # lock it first
        if presynaptic:
            other_front._yid = yid # postsynaptic
            other_front.syn_input = syn._syn_input(constellation)
        else:
            other_front._yid = -yid # presynaptic
            other_front.syn_input = 0.
        constellation.unlock(other_front)
        if presynaptic:
            self._yid = -yid
        else: # postsynaptic
            self._yid = yid
            self.syn_input = syn._syn_input(constellation)
        constellation._syn_next += 1
        # store info for Admin so that database can be updated
        aid = ActiveFrontID(ID(0,yid),b'y')
        constellation._new_AIDs[constellation._n_start +\
                                        constellation._n_newAID] = aid
        constellation._n_newAID += 1 # update index into new_AIDs
        return True

    def remove_synapse(self,constellation):
        """ Removes the synaptic connection with another front.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        """
        if self._yid == 0:
            raise SynapseError("remove_synapse: no synaptic connection present")
        if constellation.verbose >= 3:
            if constellation.my_id == 1:
                print ("Process",constellation.my_id,"cycle",constellation._proc.cycle,"deleting synapse",self._fid)
            else:
                print ("Process",constellation.my_id,"cycle",constellation.cycle,"deleting synapse",self._fid)
        syn = constellation._synapses[abs(self._yid)]
        if self._yid < 0: # presynaptic front
            other_front = constellation.front_by_id(syn.post_syn)
        else:
            other_front = constellation.front_by_id(syn.pre_syn)
        # remove synapse also in other front
        constellation.lock(other_front) # lock it first
        other_front._yid = 0
        other_front.syn_input = 0.
        result = constellation.unlock(other_front)
        # store info for Admin so that database can be updated
        if constellation.my_id > 1: # send info to admin
            if constellation._n_newAID >= constellation._new_range:
                raise OverflowError("_new_AIDs","max_active")
            aid = ActiveFrontID(ID(0,abs(self._yid)),b'z')
            constellation._new_AIDs[constellation._n_start +\
                                            constellation._n_newAID] = aid
            constellation._n_newAID += 1 # update index into new_AIDs
        self._yid = 0
        self.syn_input = 0.

    def has_synapse(self):
        """ Has self a synaptic connection defined?.
            
        Returns
        -------
        has self has a synaptic connection defined? : boolean
        """
        return self._yid != 0

    def get_synapse(self,constellation):
        """ Returns ``Synapse`` if defined.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        
        Returns
        -------
        Synapse or None: Synapse
        """
        if self._yid != 0:
            return constellation._synapses[abs(self._yid)]
        else:
            return None

    def get_branch_synapses(self,constellation):
        """ Returns a list containing all ``Synapse`` present on *self* and all its descendants.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        
        Returns
        -------
        List of Synapses: [Synapse,]
        """
        if self.has_synapse():
            syns = nds_list([self.get_synapse(constellation)])
        else:
            syns = nds_list()
        self._add_synapses(constellation,syns)
        return syns
    
    # recursive routine that performs preorder tree traversal to make list of all synapses
    def _add_synapses(self,constellation,result):
        if self.num_children > 0:
            children = self.get_children(constellation)
            for child in children:
                if child.has_synapse():
                    result.append(child.get_synapse(constellation))
                child._add_synapses(constellation,result)

    def is_presynaptic(self):
        """ Is self a presynaptic?. Returns False if no synapse defined.
            
        Returns
        -------
        is self presynaptic? : boolean
        """
        if self._yid < 0:
            return True
        else:
            return False

    def is_postsynaptic(self):
        """ Is self a postsynaptic?. Returns False if no synapse defined.
            
        Returns
        -------
        is self postsynaptic? : boolean
        """
        if self._yid > 0:
            return True
        else:
            return False

    # update internal attributes before manage_front is called
    def _pre_manage_front(self,constellation):
        if self._yid > 0: # postsynaptic synapse present: update syn_input
            syn = constellation._synapses[self._yid]
            self.syn_input = syn._syn_input(constellation)

# used to create linked list for children
#   next: index to next child in children
#   _cid: always index to fronts
class Child(Structure):
    _fields_ = [('next', c_int), ('_cid', c_int)]

    def __str__(self):
        return "Child: next " + str(self.next) + ", _cid " + str(self._cid)

# ID consists of _nid : neuron_type index or substrate for _nid == 0;
#    and _fid: fronts RawArray index
# For efficiency both are c_int, though _nid could be a c_short
class ID(Structure):
    """Class that defines the ID used to uniquely identify ``Front``, ``Neuron`` and ``Substrate`` instantiations.
    
    This class has no public attributes or methods and the private ones may change in future NeuroDevSim implementations.
    """
    _fields_ = [('_nid', c_int), ('_fid', c_int)]

    def __str__(self):
        if self._nid < 0:
            return "Neuron ID: index " + str(abs(self._nid))
        elif self._nid == 0:
            return "Substrate ID: index " + str(self._fid)
        else:
            return "Front ID: neuron type " + str(self._nid) + ", index " + str(self._fid)
    
    def _key(self): # turn FrontId into unique string
        return str(self._nid) + "_" + str(self._fid)
    
    def __hash__(self):
        return hash(self._key())

    # equality of two IDs
    def __eq__(self, other):
        if not isinstance(other,ID):
            raise TypeError("other not an ID")
        return self._nid == other._nid and self._fid == other._fid

    # less than of two IDs
    def __lt__(self, other):
        if not isinstance(other,ID):
            raise TypeError("other not an ID")
        if self._nid < other._nid:
            return True
        elif self._nid > other._nid:
            return False
        else:
            return self._fid < other._fid

    # less or equal of two IDs
    def __le__(self, other):
        if not isinstance(other,ID):
            raise TypeError("other not an ID")
        if self._nid < other._nid:
            return True
        elif self._nid > other._nid:
            return False
        else:
            return self._fid <= other._fid

    # greater than of two IDs
    def __gt__(self, other):
        if not isinstance(other,ID):
            raise TypeError("other not an ID")
        if self._nid > other._nid:
            return True
        elif self._nid < other._nid:
            return False
        else:
            return self._fid > other._fid

    # less or equal of two IDs
    def __ge__(self, other):
        if not isinstance(other,ID):
            raise TypeError("other not an ID")
        if self._nid > other._nid:
            return True
        elif self._nid < other._nid:
            return False
        else:
            return self._fid >= other._fid

empty_ID = ID(0,0)
negat_ID = ID(-1,0)

class DataID(Structure):
    """Class that describes ``Front`` and ``Neuron`` using the identifiers used in the sqlite database that stores the simulation results.
    
    This class can be used to print information about ``Front`` or ``Neuron`` that can be matched with the database entry. 

    The **attributes** are read-only, changing them will have no effect.

    Attributes:

        neuron_id : integer : neuron_id used in sqlite database.
        front_id : integer : front_id used in sqlite database, 0 if data is about a ``Neuron``.
    """
    
    _fields_ = [('neuron_id', c_int), ('front_id', c_int),]
    
    def __str__(self):
        if self.front_id > 0:
            return "Front data id: neuron_id " + str(self.neuron_id) + ", front_id " + str(self.front_id)
        else:
            return "Neuron data id: neuron_id " + str(self.neuron_id)
    
    def _key(self): # turn FrontId into unique string
        return str(self.neuron_id) + "_" + str(self.front_id)
    
    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        if not isinstance(other,DataID):
            raise TypeError("other not an DataID")
        return self.neuron_id == other.neuron_id and \
               self.front_id == other.front_id


class Substrate(Structure):
    """ Class implementing substrate.
    
    Attributes:
        name : string : name of the substrate.
        orig : Point : 3D coordinate with location.
        birth : integer : cycle when it was created.
        n_mol : integer : number of molecules when it was created.
        rate : float : production rate in molecules/cycle (optional).
        diff_c: float : 3D diffusion coefficient in µm^2/cycle (optional).
    """
    
    _fields_ = [ ('name', c_char * BNAMELENGTH), ('orig', Point), \
                 ('n_mol', c_long),('rate', c_double), ('diff_c', c_double),\
                 ('birth', c_short),('_fid', c_int), ('_nid', c_int),\
                 ('_next', c_short)]

    def __init__(self,name,orig,birth,n_mol,rate=0.0,diff_c=0.0):
        self.name = name[:BNAMELENGTH].encode('utf-8')
        self.orig = orig
        self.birth = birth
        self.n_mol = n_mol
        self.rate = rate
        self.diff_c = diff_c
        # _next is used to create a linked list of all substrate with same name

    def __str__(self):
        text = "Substrate: " + self.get_name() + ": " + str(self.orig) + " n_mol: " + \
                str(self.n_mol)
        if (self.rate > 0.) or (self.diff_c > 0.):
            return text + " rate: " + str(self.rate) + " diff_c: " + str(self.diff_c)
        else:
            return text

    # equality of two Substrate:
    def __eq__(self, other):
        if not isinstance(other,Substrate):
            raise TypeError("other not Substrate")
        if self._fid > 0: # stored substrate: compare indices
            return self._fid == other._fid
        else: # front has not been stored yet, compare main attributes
            return self.orig == other.orig and self.n_mol == other.n_mol and \
                   self.rate == other.rate and self.birth == other.birth \
                   and self.diff_c == other.diff_c

    # non equality of two Substrates
    def __ne__(self, other):
        if not isinstance(other,Substrate):
            raise TypeError("other not Substrate")
        if self._fid > 0: # stored substrate: compare indices
            return self._fid != other._fid
        else: # front has not been stored yet, compare main attributes
            return self.orig != other.orig or self.n_mol != other.n_mol or \
                   self.rate != other.rate or self.birth != other.birth \
                   or self.diff_c != other.diff_c

    def nds_class(self):
        """ Return class name.
         
        Returns
        -------
        Class name "Substrate" : string
        """
        return "Substrate"

    # return ID of substrate
    def get_id(self):
        """ Returns the ``ID`` of self.

        Returns
        -------
        self's unique ID : ``ID``.
        """
        return ID(0,self._fid)

    def get_name(self):
        """ Returns the name of the substrate.

        Returns
        -------
        self's name : string.
        """
        return repr(self.name)[2:-1]

    def is_cylinder(self):
        """ Always False.
         
        Returns
        -------
        False : boolean.
        """
        return False

def diff_gradient_to(point,substrates,cycle,size=1,what="largest"):
        """ Computes a stochastic number of substrate molecules at *point* and a unit direction vector towards the substrate from *point*.
        
        Stochasticity can be controlled by changing the *size* of the sampling volume.
        Diffusion equation used depends on property of substrate: if *rate* > 0.0: constant rate based, *n_mol* value ignored; else: instantaneous point source using *n_mol* as initial amount.
        
        Parameters
        -----------
        point : Point : location at which the concentration is computed.
        substrates : list of ``Substrate``: list of substrate or list returned by *get_substrates* method.
        cycle : integer : current simulation cycle or -1 to compute concentration at very long time after release (only for rate-based).
        Optional:
        size : integer > 0 : size of cubic sampling box in µm, larger values reduce stochasticity of result, default 1.
        what : string : either 'largest': computes concentration and direction to the nearest point in the list of substrate or 'average': computes summed concentration and mean of all directions, default 'largest'.
           
        Returns
        -------
        number of molecules at *point* and direction to substrate : integer, Point
        """
        vbox = size * size * size
        empty = Point(0.,0.,0.)
        if len(substrates) == 1:
            what = "1"
        else:
            if what == "average":
                vecs = []
                concs = 0.0 # sum of all
            else:
                largest = 0.0
                largest_vec = empty
        n_binomial = 0
        for item in substrates:
            if isinstance(item,tuple): # assume list returned by get_substrates
                sub = item[0]
            else:
                sub = item
            diff = sub.diff_c
            if diff <= 0.0:
                print (Fore.MAGENTA + "Error: substrate",sub.name,"has no diffusion information",Fore.RESET)
                conc = 0
                continue
            rate = sub.rate
            vec = sub.orig - point
            r = vec.length()
            if rate > 0.0: # constant rate based diffusion
                if cycle >= 0:
                    time = cycle - sub.birth
                    if time == 0:
                        return 0, empty
                else: # very long time after start release
                    time = 1000
                n_binomial += int(rate * time) # number of molecules in total volume
                if r == 0.0:
                    r = 0.0001 # prevent division by zero error
                # Reference: Crank J. "The Mathematics of Diffusion" 1975, eq. 3.5b p. 32
                # continuous producing point source in infinite medium
                conc = rate * erfc(r/(2.*sqrt(diff*time)))/(12.566370614359173*diff*r)
            else: # instantaneous point source diffusion
                n_mol = sub.n_mol
                time = cycle - sub.birth
                if time == 0:
                    return 0, empty
                n_binomial += n_mol # number of molecules in total volume
                dt = diff * time
                # Reference: Crank J. "The Mathematics of Diffusion" 1975, eq. 3.7 p. 29
                # instantaneous point source in infinite medium
                conc = n_mol * exp(-r*r/(4.*dt))/(8.*(3.141592653589793*dt)**1.5)
            if what == "largest":
                if conc > largest:
                    largest = conc
                    largest_vec = vec
            elif what == "average":
                concs += conc
                vecs.append(vec.nparray())
            elif what != "1":
                print (Fore.MAGENTA + "Warning: unknown 'what' in diff_gradient_to call",Fore.RESET)
                return 0, empty
        if what == "largest":
            vec = largest_vec
            conc = largest
        elif what == "average":
            avec = np.mean(vecs,axis=0)
            vec = Point(avec[0],avec[1],avec[2])
            conc = concs
        n_box = conc * vbox # turn concentration in number of molecules
        if n_binomial > 0:
            p_binomial = n_box/n_binomial
            if p_binomial < 1.0:
                n_mol = binom.rvs(n_binomial, p_binomial)
                if n_mol > 0:
                    return n_mol,vec.norm()
                else:
                    return 0,empty
        return int(n_box),vec

class Synapse(Structure):
    """ Class implementing synaptic connections.
    
    Synaptic strength is computed as ``presynaptic_neuron.firing_rate * synaptic.weight`` and is stochastic if ``presynaptic_neuron.CV_ISI > 0.``.
    Only the **attributes** listed below can be read. NEVER change these attributes.
    
    WARNING: *weight* updates are not stored in the database, use the *attrib_to_db* method to store.

    Attributes:
        pre_syn : ID : id of the presynaptic front.
        post_syn : ID : id of the postsynaptic front.
        weight : float : synaptic weight. Positive for excitation, negative for inhibition.
    """
    
    _fields_ = [('pre_syn', ID), ('post_syn', ID), ('weight', c_double),
                ('_mean', c_double), ('_std', c_double), ('_dbid', c_int)]

    def __str__(self):
        return "Synapse from " + str(self.pre_syn) + " to " + \
                str(self.post_syn) + " weight: " + str(self.weight) + \
                " (" + str(self._mean) + " " + str(self._std) + ")"

    def nds_class(self):
        """ Return class name.
         
        Returns
        -------
        Class name "Synapse" : string
        """
        return "Synapse"

    def set_weight(self,constellation,weight):
        """ Change the synaptic weight for existing synapse.
    
        Parameters
        ----------
        constellation : ``Constellation`` object.
        weight : float : new synaptic weight. Positive for excitation, negative for inhibition.
        """
        # get presynaptic neuron
        pre = constellation.front_by_id(self.pre_syn)
        neid = constellation._fronts[pre._nid][pre._sid]._sid
        neuron = constellation._neurons[neid]
        # update synaptic properties
        self._syn_update(neuron.firing_rate,neuron.CV_ISI,weight=weight)    

    def _syn_update(self,firing_rate,cv,weight=None):
        if weight:
            self.weight = weight
        self._mean = firing_rate * self.weight
        if cv > 0.:
            self._std = sqrt(cv * self._mean) # standard deviation based on variance from the CV
        else:
            self._std = 0.

    # return synaptic input value
    def _syn_input(self,constellation):
        if self._std > 0.: # need to compute syn_input every cycle
            if self._mean >= 0.: # excitation
                return max(0.,np.random.normal(self._mean,self._std))
            else: # inhibition
                return min(0.,np.random.normal(self._mean,self._std))
        else: # constant
            return self._mean
            
# Class provides a logical place for firing_rate and CV_ISI fields, which otherwise would need to exist in every front.
# Because soma and Neuron are created together in add_neurons soma _fid is at present identical to _neid, but for robustness and
#   to make import_simualtion easier they are stored separately. Soma encode _neid as it _sid.
            
class Neuron(Structure):
    """ Class for storing neuron level data.
    
    Only the **attributes** listed below can be read. NEVER change these attributes directly, use methods instead for those that can be changed. 
    
    WARNING: *firing_rate* and *CV_ISI* updates are not stored in the database, use the *attrib_to_db* method to store these.
         
    Attributes:
        soma_ID : ID : ID of the soma or root of the axon tree. 
        firing_rate : float >= 0. : firing rate of the neuron, used for synaptic connections, default: 1.
        CV_ISI : float >= 0. : Coefficient of Variation of the firing InterSpike Interval distribution, used for synaptic connections, default: 0.
        num_fronts : integer : number of fronts belonging to the neuron at end of previous cycle.
        num_retracted : integer : number of retracted fronts that belonged to the neuron at end of previous cycle.
        num_synapses : integer : number of synapses belonging to the neuron at end of previous cycle.
    """
    # private fields:
    # _neid: index into Neurons array.
    
    _fields_ = [('neuron_name', c_char * NNAMELENGTH), ('soma_ID', ID), \
                ('firing_rate', c_double), ('CV_ISI', c_double), \
                ('num_fronts', c_short), ('num_retracted', c_short),\
                ('num_synapses', c_short),('_neid', c_int)]

    def __str__(self):
        text = "Neuron " + repr(self.neuron_name)[2:-1] + " " + str(self._neid)\
                + ": rate " + str(self.firing_rate) + \
                ", CV " + str(self.CV_ISI) + ", num_fronts " + \
                str(self.num_fronts) + ", num_retracted " + \
                str(self.num_retracted) + ", num_synapses " + \
                str(self.num_synapses)
        return text
    
    # equality of two neurons
    def __eq__(self, other):
        if not isinstance(other,Neuron):
            if other == None: # self is always a Neuron in this method
                return False
            raise TypeError("other not a Neuron")
        return self._neid == other._neid

    # non equality of two neurons
    def __ne__(self, other):
        if not isinstance(other,Neuron):
            if other == None: # self is always a Neuron in this method
                return True
            raise TypeError("other not a Neuron")
        return self._neid != other._neid
        
    def nds_class(self):
        """ Return class name.
         
        Returns
        -------
        Class name "Neuron : string
        """
        return "Neuron"

    # return ID of neuron
    def get_id(self):
        """ Returns the ``ID`` of self.

        Returns
        -------
        self's unique ID : ``ID``.
        """
        return ID(-1,self._neid)

    # return DataID of neuron
    def get_dataid(self,constellation):
        """ Attribute method that returns the ``DataID`` of self.

        Returns
        -------
        self's data id : DataID
        """
        return DataID(self._neid,0)
        
    def get_name(self):
        """ Returns the name of the neuron.

        Returns
        -------
        self's neuron name : string
        """
        return repr(self.neuron_name)[2:-1]

    def get_base_name(self):
        """ Return name of the neuron without its _number_ at the end.
        
        Returns
        -------
        self's name without number : string
        """        
        # neuron_name is of form base_name + "_" + number + "_"
        name = repr(self.neuron_name)[2:-2] # delete final _
        return name[:name.rfind('_') - 1]
    
    def set_firing_rate(self,constellation,rate):
        """ Change the neuron firing rate: this will affect the entire neuron.
    
        Parameters
        ----------
        constellation : ``Constellation`` object.
        rate : float >= 0. : new firing rate.
        """
        if rate < 0.:
            raise ValueError("max_active",">= 0.")
        self.firing_rate = rate
        self._update_all_pre(constellation,rate,self.CV_ISI)

    def set_CV_ISI(self,constellation,CV_ISI):
        """ Change the neuron firing rate CV: this will affect the entire neuron.
        
        A *CV_ISI* larger than zero makes the postsynaptic *syn_input* a stochastic variable.
    
        Parameters
        ----------
        constellation : ``Constellation`` object.
        CV_ISI : float >= 0. : new firing rate CV.
        """
        if CV_ISI < 0.:
            raise ValueError("max_active",">= 0.")
        self.CV_ISI = CV_ISI
        self._update_all_pre(constellation,self.firing_rate,CV_ISI)
        
    def get_neuron_fronts(self,constellation,returnID=False):
        """ Returns a list of all fronts of the entire neuron.
        
        This method by default returns a list of ``Front``, but can also return a list of ``ID``.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        Optional :
        returnID : boolean : return list of ``ID``, default: False.
 
        Returns
        -------
        Depending on returnID parameter nds_list of [Front,] or of [ID,] : list.
        """
        # get soma: root of the tree
        soma = constellation._fronts[self.soma_ID._nid][self.soma_ID._fid]
        # iteratively traverse all children
        if returnID:
            result = nds_list([ID(soma._nid,soma._fid)])
        else:
            result = nds_list([soma])
        if soma.num_children == 0:
            return result
        else:
            soma._add_children(constellation,result,returnID)
        return result  
              
    def get_neuron_soma(self,constellation,returnID=False):
        """ Returns the soma of a neuron.
        
        This method by default returns a ``Front``, but can also return an ``ID``.
        
        Parameters
        ----------
        constellation : ``Constellation`` object.
        Optional :
        returnID : boolean : return ``ID``, default: False.
 
        Returns
        -------
        Depending on returnID parameter: Front or ID.
        """
        # get soma: root of the tree
        soma = constellation._fronts[self.soma_ID._nid][self.soma_ID._fid]
        # iteratively traverse all children
        if returnID:
            return ID(soma._nid,soma._fid)
        else:
            return soma
              
    # updates all synapses this neuron connects to presynaptically
    def _update_all_pre(self,constellation,fr,cv):
        fronts = self.get_neuron_fronts(constellation)
        for f in fronts:
            if f._yid < 0: # presynaptic synapse present
                syn = constellation._synapses[abs(f._yid)]
                syn._syn_update(fr,cv)

### Classes and methods: internal

## EXCEPTIONS ##
# Base class for exceptions
class Error(Exception):
    pass

# An active filipod child in migrate_soma
class ActiveChildError(Error):
    """ Filipod in ``migrate_soma`` call is active.
    """
    def __str__(self):
        return Fore.RED + "Active filipod child in migrate_soma method." + Fore.RESET

# A bad number of children in migrate_soma
# Attributes:
#    explanation : string : bug description
class BadChildError(Error):
    """ Too many children or children with wrong swc_type in ``migrate_soma`` call.
    
    Attributes:
        explanation : string : bug description
    """
    def __init__(self,explanation):
        self.explanation = explanation

    def __str__(self):
        return Fore.RED + self.explanation + " in migrate_soma method." + Fore.RESET

# A bug occurred
# Attributes:
#    method : string : name of method where bug occurred
#    explanation : string : bug description
class BugError(Error):
    """ Bug occurred during the simulation. NeuroDevSim contains a lot of code that checks for inconsistent behavior.
    
    Attributes:
        method : string : name of method where bug occurred
        explanation : string : bug description
    """
    def __init__(self,method,explanation):
        self.method = method
        self.explanation = explanation

    def __str__(self):
        return Fore.RED + "Bug in " + self.method + ": " + self.explanation + \
                "." + Fore.RESET

# Collision error: new front collides with existing one
# Attributes:
#    front : ``Front`` : first colliding front
class CollisionError(Error):
    """ New front collides with existing one.
    
    Value stored depends on local constellation.only_first_collision: either a single ``Front`` (True) of a list of ``Front`` (False) is stored in *collider*
    
    Attributes:
        only_first : boolean : copy of constellation.only_first_collision.
        collider : ``Front`` or [``Front``,]: colliding front(s).
        distance : float or [float,]: distance(s) to colliding front(s).
    """
    def __init__(self,only_first,collider,distance):
        self.only_first = only_first
        self.collider = collider
        self.distance = distance

    def __str__(self):
        if self.only_first:
            return Fore.RED + "New front collides with " + str(self.collider) + \
                ", distance  {:4.2f}".format(self.distance) + "." + Fore.RESET
        else:
            return Fore.RED + "New front collides with " + str(len(self.collider)) + \
                " other fronts." + Fore.RESET

# Grid competition error: several processes trying to lock same grid point
class GridCompetitionError(Error):
    """ Process tries to lock an already locked grid point.
    
    This error occurs during collision detection and indicates a great likelihood of future collision. Different from ``CollisionError`` the colliding front is unknown.
    
    Attributes:
        gid : integer : grid index that is not available
    """
    def __init__(self,gid):
        self.gid = gid

    def __str__(self):
        return Fore.RED + "Grid competition error for grid id " + str(self.gid) +  "." + Fore.RESET

# Coordinate inside parent error
class InsideParentError(Error):
    """ Coordinate is inside the prospective parent front."""
    def __str__(self):
        return Fore.RED + "Point is inside prospective parent front." + Fore.RESET

# A lock fails
# Attributes:
#    type : string : type of lock
#    id : ID or int : identifier
class LockError(Error):
    """ Front or Substrate locking failed.
    
    Attributes:
        type : string : type of lock
        id : ID : identifier of front or substrate
        proc: integer : number of process that tries to lock
        lproc: integer : number of process that has the lock
    """
    def __init__(self,type_string,id,proc,lproc):
        self.type = type_string
        self.id = id
        self.proc = proc
        self.lproc = lproc

    def __str__(self):
        return Fore.RED + str(self.type) + " " + str(self.id) + \
                " lock failed for process " + str(self.proc) + \
                " , locked by process " + str(self.lproc) + \
                " (increase time_out)." + Fore.RESET

# Method can only be called by self error
class NotSelfError(Error):
    """ A method call was attempted for a front different than *self*. This method only allow calls from self to preserve model integrity.
    
    Attributes:
        method : string : name of offending method
    """
    def __init__(self,method):
        self.method = method

    def __str__(self):
        return Fore.RED + self.method + " can only be called by self." + Fore.RESET

# Migrating front should be a soma
class NotSomaError(Error):
    """ Only somata can migrate."""
    def __str__(self):
        return Fore.RED + "migrate_soma called by front that is not a soma." +\
                Fore.RESET

# A shared array overflows
# Attributes:
#    array : string : name of overflowing array
#    attribute : string : name of Admin_agent attribute to increase
class OverflowError(Error):
    """ Overflow of a shared array occurred.
    
    Attributes:
        array : string : name of overflowing array
        attribute : string : name of Admin_agent attribute to increase
    """
    def __init__(self,array,attribute):
        self.array = array
        self.attribute = attribute

    def __str__(self):
        return Fore.RED + self.array + " array overflow (increase Admin_agent " + \
                self.attribute + " attribute)." + Fore.RESET

# Synapse error
# Attributes:
#    message : string : description of error
class SynapseError(Error):
    """ Problem adding or removing synapse.
    
    Attributes:
        explanation : string : specific problem.
    """
    def __init__(self,explanation):
        self.explanation = explanation

    def __str__(self):
        return Fore.RED + "Synapse error in " + self.explanation + "." + Fore.RESET

# Type error
# Attributes:
#    message : string : description of error
class TypeError(Error):
    """ A method parameter has an inappropriate type.
    
    Attributes:
        message : string : name of the parameter.
    """
    def __init__(self,message):
        self.message = message

    def __str__(self):
        return Fore.RED + "Type error: " + self.message + "." + Fore.RESET

# Value error
# Attributes:
#    value : string : name of the parameter
#    range : string : allowed range formulated after "should be"
class ValueError(Error):
    """ A method parameter has an inappropriate value.
    
    Attributes:
        value : string : name of the parameter.
        range : string : description of allowed range.
    """
    def __init__(self,value,range):
        self.value = value
        self.range = range

    def __str__(self):
        return Fore.RED + "Inappropriate value for " + self.value + \
                ": should be " + self.range + "." + Fore.RESET

# Coordinate outside simulation volume error: new front is outside simulation volume
# Attributes:
#    coord : float : the first coordinate that is outside the volume
class VolumeError(Error):
    """ Point is outside the simulation volume.
    
    Attributes:
        coord : float : first coordinate outside of simulation volume
    """
    def __init__(self,coord):
        self.coord = coord

    def __str__(self):
        return Fore.RED + "Point is outside the simulation volume for coordinate "\
                + "{:4.2f}".format(self.coord) + "." + Fore.RESET


#### General routines
SMALL_NUM =  0.00000001
# compute minimal distance between two cylinders with first represented by points
# Segment1=(p0,p1), Segment2=(p2,p3)
# c++ from http://geomalgorithms.com/a07-_distance.html#dist3D_Segment_to_Segment%28%29
def dist3D_cyl_to_cyl(orig,end,front,points=False):
    """ Mathematical method to compute minimal distance between two cylinders. Can return the closest points on each cylinder.

    c++ from http://geomalgorithms.com/

    Parameters
    ----------
    orig : ``Point`` : origin of first cylinder.
    end : ``Point`` : end of first cylinder.
    front : ``Front`` : second cylinder
    Optional :
    points : boolean : return also closest points, default: False.

    Returns
    -------
    Depending on points parameter distance or distance and two points : float or float,Point,Point.
    """    
    u = end - orig              # Vector   u = S1.P1 - S1.P0
    v = front.end - front.orig  # Vector   v = S2.P1 - S2.P0
    w = orig - front.orig       # Vector   w = S1.P0 - S2.P0
    a = u.dot(u)      # always >= 0
    b = u.dot(v)
    c = v.dot(v)      # always >= 0
    d = u.dot(w)
    e = v.dot(w)
    D = a*c - b*b     # always >= 0
    sD = D            # sc = sN / sD, default sD = D >= 0
    tD = D            # tc = tN / tD, default tD = D >= 0
    # compute the line parameters of the two closest points
    if D < SMALL_NUM:  # the lines are almost parallel
        sN = 0.0         # force using point P0 on segment S1
        sD = 1.0         # to prevent possible division by 0.0 later
        tN = e
        tD = c
    else:              # get the closest points on the infinite lines
        sN = (b*e - c*d)
        tN = (a*e - b*d)
        if sN < 0.0:        # sc < 0 => the s=0 edge is visible
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:       # sc > 1  => the s=1 edge is visible
            sN = sD
            tN = e + b
            tD = c
    if tN < 0.0:            # tc < 0 => the t=0 edge is visible
        tN = 0.0
        # recompute sc for this edge
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:           # tc > 1  => the t=1 edge is visible
        tN = tD
        # recompute sc for this edge
        if (-d + b) < 0.0:
            sN = 0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = -d + b
            sD = a
    # finally do the division to get sc and tc
    # sc = (abs(sN) < SMALL_NUM ? 0.0 : sN / sD)
    if np.abs(sN) < SMALL_NUM:
        sc = 0.0
    else:
        sc = sN / sD
    #tc = (abs(tN) < SMALL_NUM ? 0.0 : tN / tD)
    if np.abs(tN) < SMALL_NUM:
        tc = 0.0
    else:
        tc = tN/tD
    # get the difference of the two closest points
    dP = w + (u * sc) - (v * tc)  # =  S1(sc) - S2(tc)
    if points:
        return dP.length(), orig + (u * sc), front.orig + (v * tc)
    else:
        return dP.length()   # return the closest distance

# compute minimal distance between a point and a cylinder
# P, Segment=(p0,p1)
# c++ from http://geomalgorithms.com/a02-_lines.html#dist_Point_to_Segment()
def dist3D_point_to_cyl(point,orig,end,points=False):
    """ Mathematical method to compute minimal distance a point and a cylinder. Can return the closest point on the cylinder.

    c++ from http://geomalgorithms.com

    Parameters
    ----------
    point : ``Point``
    orig : ``Point`` : origin of cylinder.
    end : ``Point`` : end of cylinder.
    Optional :
    points : boolean : return also closest point, default: False.

    Returns
    -------
    Depending on points parameter distance or distance and a points : float or float,Point.
    """      
    v = end - orig  # Vector v = S.P1 - S.P0
    w = point - orig  # Vector w = P - S.P0
    c1 = w.dot(v)
    if c1 <= 0.0:
        dP = point - orig
        if points:
            return dP.length(),orig
        else:
            return dP.length()
    c2 = v.dot(v)
    if c2 <= c1:
        dP = point - end
        if points:
            return dP.length(),end
        else:
            return dP.length()
    b = c1 / c2
    Pb = orig + (v * b)
    dP = point - Pb
    if points:
        return dP.length(),Pb
    else:
        return dP.length()

# returns distance of point to Front, Substrate or Point
def _point_front_dist(point,object):
    if object.is_cylinder():
        return dist3D_point_to_cyl(point,object.orig,object.end)
    else: # spherical Front or Substrate or Point
        if isinstance(object,Point):
            return (point - object).length()
        else:
            return (point - object.orig).length()

# returns distance of cylinder to Front, Substrate or Point
# if valid=True checks are made on whether this is a reliable distance for validity checking,
#   if not the distance is returned as a negative value to indicate a problem.
def _cylinder_front_dist(orig,end,object,valid=False):
    if object.is_cylinder():
        if valid:
            D1,p1,p2 = dist3D_cyl_to_cyl(orig,end,object,points=True)
            if orig == p1: # distance is measured to origin of first cylinder
                return -D1 # cannot be used for collision detection
            else:
                return D1
        else:
            return dist3D_cyl_to_cyl(orig,end,object)
    else: # spherical Front or Substrate or Point
        if isinstance(object,Point):
            return dist3D_point_to_cyl(object,orig,end)
        else:
            return dist3D_point_to_cyl(object.orig,orig,end)

# return point orthogonal to p
def _orthogonal(p):
    if p.y == 0. and p.z == 0:
        return p.cross(Point(0., 1., 0.))
    else:
        return p.cross(Point(1., 0., 0.))

# Method to convert a list of points around a sphere into polar coordinates
#   with a specified north pole (Point) on the sphere.
# Returns:
# useR, R, [Pol_point,] : integer how to use R, rotation matrix, nds_list of polar points.
def _polar_cloud(north,points,fid):
    rad = north.length() # radius of implicit sphere
    if rad == 0.:
        return 0, 0, []
    pps = nds_list() # list to be returned
    # compute rotation matrix
    old_north = np.array([0., 0., 1.])
    n_north = north.norm()
    new_north = n_north.nparray()
    dot = np.dot(old_north,new_north)
    if abs(abs(dot) - 1.0) > 1.0e-15:  # need to rotate and convert to polar
        rot_axis = np.cross(new_north,old_north)
        rot_angle = np.arccos(dot)
        # rotate vectors by this angle
        R = expm(np.cross(np.eye(3), rot_axis / norm(rot_axis) * rot_angle))
        prev_phi = -10. # impossible value
        for p in points:
            rot_np = np.dot(R,p.nparray())
            rot_p = Point(rot_np[0],rot_np[1],rot_np[2])
            rot_pp = rot_p.toPol_point()
            if rot_pp.phi == prev_phi: # ignore points with same phi
                continue
            pps.append(rot_pp)
            prev_phi = rot_pp.phi
        useR = 1
    elif abs(dot + 1.0) < 1.0e-15: # need to rotate by 180 deg and convert to polar
        for p in points:
            pp = p.toPol_point() # convert to polar
            pp.theta = 180. - pp.theta # flip theta
            pps.append(pp)
        useR = 2
        R = None
    else: # no rotation required -> convert to polar
        for p in points:
            pps.append(p.toPol_point())
        useR = 3
        R = None
    return useR, R, pps

# convert key back to ID
def _key_to_ID(key): 
    index = key.rfind('_')
    if index == -1: # not a key
        return ID(0,0)
    else:
        fid = int(key[index + 1:])
        if fid == 0: # a Neuron
            return ID(-int(key[:index]),0)
        else: # Front or Substrate
            return ID(int(key[:index]),int(key[index + 1:]))

# convert key back to DataID
def _key_to_DataID(key): 
    index = key.rfind('_')
    if index == -1: # not a key
        return DataID(0,0)
    else:
        fid = int(key[index + 1:])
        if fid == 0: # a Neuron
            return DataID(int(key[:index]),0)
        else: # Front or Substrate
            return DataID(int(key[:index]),int(key[index + 1:]))

# returns neuron_type name with "class '__main__. '> removed
def _strip_neuron_type(nt_name):
    name = str(nt_name)
    if name[-1] == '>': # contains "class '__main__. '>"
        return name[name.rfind('.') + 1 : -2] # extract original name
    else:
        return name

### Arcs
# stores information about arc generated by Front.arc_around
# fields:
#   cycle : constellaton.cycle
#   complete : 0, cycle when all entries has been used
#   caller : ID of Front that made the arc
#   sphere : ID of sphere Front around which the arc goes
#   arc_angle : angle covered in degrees
#   num_points : number of points stored for complete arc
#   start : starting index in self._arc_points shared array
#   count : number of points to return each cycle
#   next_point : index of next point to use, relative to index
# methods are in Constellation because they need to access both 
#   self._arcs and self._arc_points shared arrays
class Arc(Structure):
    _fields_ = [('cycle', c_int), ('complete', c_int), ('neuron_id', c_int),\
                ('caller_fid', c_int),('last_fid', c_int), ('sphere', ID),\
                ('arc_angle', c_double), ('num_points', c_int), \
                ('index', c_int), ('count', c_short), ('next_point', c_short)]
            
    def __str__(self):
        return "Arc " + str(self.arc_angle) + " deg made " + str(self.cycle) + \
               " by neuron " + str(self.neuron_id) + " front " +\
               str(self.caller_fid) + " last used by " + str(self.last_fid)+\
               ": " + str(self.num_points) +" points, next " +\
               str(self.next_point) + ", completed on " + str(self.complete)


### Constellation data structures
# Volume: read only data about simulation volume and grid, initialization method only
class Volume(object):

    def __init__(self,sim_volume,grid_step):
        if not isinstance(sim_volume,list):
            raise TypeError("sim_volume should be a list")
        if len(sim_volume) == 3: # only single coordinate provided.
            self.sim_volume = [[0.,0.,0.],sim_volume]
        elif len(sim_volume) == 2: # two coordinates provided.
            for i in range(2):
                if len(sim_volume[i]) != 3:
                    raise TypeError("sim_volume should be a list containing 2 coordinate lists of length 3")
            self.sim_volume = sim_volume
        else:
            raise TypeError("sim_volume should be a coordinate list of length 3 or a list containing 2 coordinate lists of length 3")
        # compute sizes of volume for grid
        self.sim_sizes = [] # float entry for each coordinate
        self.grid_sizes = [] # int entry for each coordinate
        self.grid_max = 1 # total size of grid
        for i in range(3):
            s_size = ceil(self.sim_volume[1][i] - self.sim_volume[0][i])
            if s_size <= 0.0:
                raise ValueError("simulation volume size along axis " + str(i),"larger than zero")
            self.sim_sizes.append(s_size)
            g_size = int(s_size) // int(grid_step) + 1
            self.grid_sizes.append(g_size)
            self.grid_max *= g_size
        self.grid_max += 1 # grid[0] not used
        self.grid_step = grid_step
        self.half_grid_step = grid_step * 0.5
        #   multipliers for _grid method
        grid_size1 = self.grid_sizes[2]
        grid_size2 = self.grid_sizes[1] * self.grid_sizes[2]
        self.grid_div = 1. / grid_step
        # all 27 directions as difference in index to grid:
        # order: 7 cardinal first, then lower/higher z plane cardinal, then each plane diagonal
        self._sdirections = [0, -grid_size2, -grid_size1, -1, grid_size2, \
                grid_size1, 1, -grid_size2 -1, -grid_size1 -1,grid_size2 -1 ,\
                grid_size1 -1, -grid_size2 +1, -grid_size1 +1, grid_size2 +1,\
                grid_size1 +1,-grid_size2 - grid_size1,-grid_size2 + grid_size1,\
                grid_size2 - grid_size1, grid_size2 + grid_size1, \
                -grid_size2 - grid_size1 - 1, -grid_size2 + grid_size1 - 1, \
                grid_size2 - grid_size1 - 1, grid_size2 + grid_size1 - 1, \
                -grid_size2 - grid_size1 + 1, -grid_size2 + grid_size1 + 1, \
                grid_size2 - grid_size1 + 1, grid_size2 + grid_size1 + 1]
        self.grid_size1 = grid_size1
        self.grid_size2 = grid_size2

# Constellation: contains the constellation
class Constellation(object):
    """Class that provides other classes access to NeuroDevSim shared memory. Passed to many ``Front`` methods and provides some unique methods.
    
    Most of this class is for internal use only.
    A few **attributes** listed below can be read. NEVER change these attributes except for *only_first_collision* which can be changed.
    
    Attributes:
        cycle : integer : current simulation cycle.
        only_first_collision : boolean : stop checking for collision after first one is found, otherwise continue till all collisions have been detected, this **can be changed** by the user but will only affect the constellation of a single processor, default: True.
        sim_volume : list of 2 lists : the simulation volume as specified during **Admin_agent** instantiation.
        verbose : integer : the verbosity level as specified during **Admin_agent** instantiation.
    """

    def __init__(self,proc,master_constell):
        # PUBLIC attributes
        self.cycle = 0
        self.only_first_collision = True
        self.verbose = proc.verbose
        self.sim_volume = proc._volume.sim_volume
        self.my_id = my_id = proc.my_id
        # PRIVATE attributes
        self._proc = proc # links back to Proc_agent or Admin_agent
        self._automatic = True
        # shared arrays
        self._volume = master_constell[34]
        self._neurons = master_constell[41]
        self._fronts = master_constell[0]
        self._num_types = master_constell[33]
        self._num_types_1 = self._num_types + 1
        # processor specific ranges in fronts shared array
        self._f_next_indices = master_constell[3]
        self._f_max_indices = master_constell[12]
        self._front_lock = master_constell[6]
        #self._actives = master_constell[1]
        act_range = master_constell[13]
        self._new_range = int(act_range * 1.5)
        # processor specific range in new_AIDs shared array
        self._n_start = (my_id - 2) * self._new_range # starting index into new_AIDs
        self._new_AIDs = master_constell[2] # all needed
        self._n_newAID = 0 # number of new fronts on this process this cycle
        self._children = master_constell[4]
        child_range = master_constell[14]
        # processor specific range in children shared array
        self._c_next = (my_id - 1) * child_range # index into children, will increase
        self._c_max = my_id * child_range # max index into children
        self._neuron_types = master_constell[17]
        self._max_arcs = master_constell[35]
        self._tot_arcs = master_constell[39]
        max_arc_points = master_constell[36]
        if my_id > 1:
            self._arcs = master_constell[37]
            self._a_start = (my_id - 2) * self._max_arcs
            # compute fixed indices into self._arc_points
            for n in range(self._max_arcs):
                self._arcs[self._a_start + n].index = (self._a_start + n) * ARCLENGTH
            self._empty_arcs = [True] * self._max_arcs # available slots in self._arcs
            self._arc_points = master_constell[38]
        self._substrate = master_constell[20]
        self._max_sub_names = master_constell[21]
        max_sub_admin = master_constell[22]
        sub_range = master_constell[23]
        self._s_next = max_sub_admin + (my_id - 1) * sub_range
        self._s_max = self._s_next + sub_range
        self._synapses = master_constell[30]
        syn_range = master_constell[31]
        # processor specific range in synapses shared array
        if my_id == 1:
            self._syn_next = 1 # zero not used
            self._syn_max = syn_range
        else:
            self._syn_next = (my_id - 1) * syn_range + 1
            self._syn_max = self._syn_next + syn_range
        self._grid = master_constell[7]
        self._grid_wlock = master_constell[8]
        self._grid_rlock = master_constell[32]
        self._gwlock_request = master_constell[25]
        self._grlock_request = master_constell[27]
        self._grid_extra = master_constell[10]
        self._extra_range = master_constell[11]
        self._extra_size = master_constell[15]
        self._extra_size_1 = self._extra_size - 1
        self._max_extra = master_constell[16]
        # processor specific range in grid shared array, includes admin space
        if my_id == 1:
            self._g_next = 1
        else:
            self._g_next = (my_id - 1) * self._extra_range + 1
        self._g_max = self._g_next + self._extra_range
        gids_range = master_constell[18]
        self._gids = master_constell[19]
        # processor specific range in gids shared array
        if my_id == 1:
            self._gid_next = 1 # zero not used
        else:
            self._gid_next = (my_id - 1)* gids_range + 1
        self._gid_max = self._gid_next + gids_range
        # timing counters
        self._front_lock_wait = 0. # total time waiting for fronts to unlock
        self._grid_lock_wait = 0. # total time waiting for grid to unlock
        # local data
        self._collisions = {} # collisions by key front._key()
        self._manage = None # copy of self for manage_front call, used to check
        self._newfs = [] # new fronts for this manage_front call
        self._deleted_gids = [] # gids that _delete_front could not delete
        self._used_arcs = {} # dict by front._key() of [index,] into self._arcs
    
    # return front corresponding to ID
    def front_by_id(self,id):
        """ Returns ``Front`` with given ``ID``.

        Parameters
        ----------
        id : ``ID``.

        Returns
        -------
        front : Front or None
        """
        if not isinstance(id,ID) or (id._nid <= 0):
            raise TypeError("id should be a front ID: " + str(id))
            return None
        else:
            front = self._fronts[id._nid][id._fid]
            # check whether front has been initialized
            count = 0
            # fix weird (cache? predictive branching?) bug where an initialized front is initially returned as empty
            while front._sid == 0:
                if count > 10:
                    return None
                time.sleep(LOCKPAUSE)
                front = self._fronts[id._nid][id._fid]
                count += 1
            return front

    # return neuron corresponding to ID
    def neuron_by_id(self,id):
        """ Returns ``Neuron`` with given ``ID``.

        Parameters
        ----------
        id : ``ID``.

        Returns
        -------
        neuron : Neuron or None
        """
        if not isinstance(id,ID) or (id._nid >= 0):
            raise TypeError("id should be a neuron ID: " + str(id))
        else:
            neuron = self._neurons[id._fid]
            # check whether neuron has been initialized
            if neuron.soma_ID._nid == 0:
                return None
            return neuron
            
    # returns index 1 -len(neuron_types)
    def neuron_type_index(self,front_class):
        """ Returns index of a *front_class*, this index is in range 1 - len(neuron_types).
        
        Method will cause a TypeError if *front_class* does not exist.

        Parameters
        ----------
        front_class : string : one of the front_class defined in the neuron_types list for ``Admin_agent``.

        Returns
        -------
        type_index : integer
        """
        try:
            type_index = self._neuron_types.index(front_class) + 1
            return type_index
        except:
            raise TypeError("undeclared front_class: " + str(front_class))
            return

    # return neuron(s) with corresponding name
    def neurons_by_name(self,name,type_index=None):
        """ Returns all neurons with given *name* (wildcard).

        Parameters
        ----------
        name : string : all neurons that start with name will be returned.
        Optional :
        type_index : integer : index returned by ``neuron_type_index``, only search for neurons of this *front_class* (faster), default: None. 

        Returns
        -------
        nds_list of neurons : [Neuron] or []
        """
        neurons = nds_list()
        if type_index:
            if (type_index < 1) or (type_index > len(self._neuron_types)):
                raise ValueError("type_index","range: 1 - " + str(len(self._neuron_types)))
            return
        for n in range(1,constellation._num_types_1):
            if type_index and (n != type_index ):
                continue
            for i in range(1,self._f_next_indices[n][1]):
                soma = self._fronts[n][i]
                if soma.order != 0:
                    raise BugError("neuron_by_name","not a soma " + str(soma))
                if soma.get_neuron_name(self).startswith(name):
                    neurons.append(self._neurons[soma._sid])
        return neurons

    # return alls neuron(s) with corresponding front_class
    def neurons_by_front_class(self,front_class):
        """ Returns all neurons with given *front_class*.

        Parameters
        ----------
        front_class : string : one of the front_class defined in the neuron_types list for ``Admin_agent``.

        Returns
        -------
        nds_list of neurons : [Neuron] or []
        """
        neurons = nds_list()
        type_index = self.neuron_type_index(front_class)
        for i in range(1,self._f_next_indices[type_index][1]):
            soma = self._fronts[type_index][i]
            if soma.order != 0:
                raise BugError("neurons_by_front_class","not a soma " + str(soma))
            neurons.append(self._neurons[soma._sid])
        return neurons

    # return alls neuron(s) with corresponding neuron_type index
    def neurons_by_type(self,type_index):
        """ Returns all neurons with given *type_index* (for front_class).

        Parameters
        ----------
        type_index : integer : index returned by ``neuron_type_index``. 

        Returns
        -------
        nds_list of neurons : [Neuron] or []
        """
        neurons = nds_list()
        for i in range(1,self._f_next_indices[type_index][1]):
            soma = self._fronts[type_index][i]
            if soma.order != 0:
                raise BugError("neurons_by_front_class","not a soma " + str(soma))
            neurons.append(self._neurons[soma._sid])
        return neurons

   # return substrate corresponding to substrate id (int)
    def substrate_by_id(self,id):
        """ Returns ``Substrate`` with given ``ID``.

        Parameters
        ----------
        id : ``ID``.

        Returns
        -------
        substrate : Substrate or None
        """
        if not isinstance(id,ID) or (id._nid != 0):
            raise TypeError("id should be a substrate ID: " + str(id))
        if id._fid > 0:
            return self._substrate[id._fid]
        else:
            return None
        
    def id_to_data(self,id):
        """ Returns the ``DataID`` for a given ``ID``.
        
        Parameters
        ----------
        id : ``ID``.

        Returns
        -------
        data id : DataID
        """
        if not isinstance(id,ID) or (id._nid == 0):
            raise TypeError("id should be a front or neuron ID: " + str(id))
        if id._nid == -1: # Neuron ID
            return DataID(id._fid,0)
        else: # front ID
            front = self._fronts[id._nid][id._fid]
            neuron_id = self._fronts[id._nid][front._sid]._sid
            return DataID(neuron_id,id._fid)
        
    def data_to_id(self,data):
        """ Returns the ``ID`` for a given ``DataID``.
        
        Parameters
        ----------
        data id : ``DataID``.

        Returns
        -------
        id : ID
        """
        if not isinstance(data,DataID):
            raise TypeError("data should be a DataID: " + str(DataID))
        if data.front_id == 0:  # Neuron DataID
            return ID(-1,data.neuron_id)
        else: # Front DataID
            nid = self._neurons[data.neuron_id].soma_ID._nid
            return ID(nid,data.front_id)
        
    # process my_id requests lock of front or substrate object
    # either the object or its ID can be passed
    # time_out is max time in seconds that we wait before declaring failure
    # Python does not provide object based locks so we need our own solution
    # item can be a front, substrate or ID
    def lock(self,item,time_out=0.25):
        """ Locks memory access to ``Front`` or ``Substrate`` objects.
        
        This prevents simultaneous writing by several different processes to the same memory address, resulting in unpredictable outcomes. Should be used before changing any attribute that does not belong to *self*.
        
        If the object is already locked by another process this call will wait till it the other lock is either released or the *time_out* period in seconds has passed. Every successful **lock** call should be followed by an **unlock** call within the same method context. Do this as soon as possible to allow other processes to access the object when needed.

        Parameters
        ----------
        item : ``ID``, ``Front`` or ``Substrate``.
        Optional :
        time_out : float : time in seconds that lock waits for another process to unlock the item.
        """
        nid = item._nid
        fid = item._fid
        if self.verbose >= 3:
            print (self.my_id,"locking",item)
        wait = 0.0
        prev_locked = True
        while prev_locked:
            # check whether unlocked
            while wait < time_out:
                if self._front_lock[nid][fid] != 0: # is locked
                    if self._front_lock[nid][fid] == self.my_id: # by caller?
                        return # successful lock
                else:
                    prev_locked = False
                    break
                # wait before checking again
                if self.verbose >= 6:
                    print ("Locked by",self._front_lock[nid][fid],"waiting on",self.my_id,"for",item)
                time.sleep(LOCKPAUSE)
                wait += LOCKPAUSE
            if not prev_locked: # try to lock
                self._front_lock_wait += wait # keep track of total waiting time
                self._front_lock[nid][fid] = self.my_id
                time.sleep(1*LOCKPAUSE) # wait to see whether lock is stable
                if self._front_lock[nid][fid] == self.my_id: # by caller?
                    return # successful lock
                else: # overwritten by somebody else
                    prev_locked = True
            else: # time_out occured
                if nid == 0:
                    raise LockError("Substrate",item.get_id(),self.my_id,\
                                    self._front_lock[nid][fid])
                else:
                    raise LockError("Front",item.get_id(),self.my_id,\
                                    self._front_lock[nid][fid])
        raise BugError("lock","")

    # process my_id unlocks front or substrate object
    # either the object or its ID can be passed
    def unlock(self,item):
        """ Unlocks memory access to ``Front`` or ``Substrate`` object after a previous **lock** call.
        
        Should be called from inside the same method that called **lock** and as soon as possible. Not calling unlock may result in NeuroDevSim hanging or crashing.
        
        Parameters
        ----------
        item : ``ID``, ``Front`` or ``Substrate``.
        """
        nid = item._nid
        fid = item._fid
        if self.verbose >= 3:
            print (self.my_id,"unlocking",item)
        if self._front_lock[nid][fid] != self.my_id: # other process grabbed this
            if self._front_lock[nid][fid] == 0:
                if self.verbose >= 2:
                    print (Fore.MAGENTA + "Warning: trying to unlock object that is not locked",nid,fid,Fore.RESET)
                return
            else:
                raise BugError("unlock","lock overwritten by " + str(self._front_lock[nid][fid]))
        self._front_lock[nid][fid] = 0

    # add substrate, called from one of the front methods.
    #  updates self.proc data structures
    def add_substrate(self,substrate):
        """ Add substrate to the simulation.
        
        This method can be called by any ``Front`` method. Biological realism requires that fronts add substrate at nearby locations only, but this is not enforced.
        
        Parameters
        ----------
        substrate: ``Substrate`` or list of ``Substrate``
        """
        proc = self._proc
        if isinstance(substrate,Substrate):
            subs = [substrate]
        elif not isinstance(substrate,list):
            raise TypeError("substrate must be Substrate or [Substrate,]")
        else:
            subs = substrate
        if self.verbose >= 3:
            print (self.my_id, "add_substrate",subs[0].get_name())
        for sub in subs:
            if proc.s_next >= proc.s_max:
                raise OverflowError("substrate","max_substrate")
            if self._n_newAID >= self._new_range:
                raise OverflowError("_new_AIDs","max_active")

            sub._fid = proc.s_next
            proc.substrate[proc.s_next] = sub
            aid = ActiveFrontID(ID(0,proc.s_next),b's')
            proc.s_next += 1
            self._new_AIDs[self._n_start + self._n_newAID] = aid
            self._n_newAID += 1 # update index into new_AIDs
    
    ### Private constellation routines ###
    ### Front routines
    
    # Enter new front data into a self._fronts entry, this may be overwritten
    #   again if it fails collision or volume testing. Does not update parent.
    # Parameters:
    #   parent : ``Front`` or None : parent front.
    #   cylinder : boolean : is cylindrical.
    #   coordinate : ``Point`` : end (cylinder) or orig (sphere).
    #   radius : float : radius if > 0., otherwise parent radius.
    #   name : string : optional branch_name or neuron_name.
    #   swc_type : integer : swc_type if > 0, otherwise parent swc_type.
    #   nid : integer : must be provided if parent==None (add_neurons).
    # Returns the front or ValueError, BugError, OverflowError.
    def _enter_front(self,parent,cylinder,coordinate,radius,name,swc_type,\
                    nid=None,trailing=False):
        if self.verbose >= 7:
            print (self.my_id,"_enter_front",parent,cylinder,coordinate,swc_type,nid,trailing)
        if parent:
            nid = parent._nid # neuron index
        elif (not nid) or (nid <= 0) or (nid > len(self._neuron_types)):
            raise BugError("_enter_front","bad nid " + str(nid))
        fid = self._f_next_indices[nid][self.my_id]
        # error checking
        if fid == self._f_max_indices[nid][self.my_id]:
            raise OverflowError("_fronts","max_fronts")
        if radius <= 0.:
            raise ValueError("radius","larger than zero")
        if radius >= self._volume.grid_step + self._volume.half_grid_step:
            error = ValueError("radius","not much larger than grid_step")
        if swc_type < 0.:
            raise ValueError("swc_type","larger than zero")
        # enter data into self._fronts
        front = self._fronts[nid][fid]
        if parent:
            front.radius = radius
            if cylinder: # make a cylinder
                if parent.is_cylinder(): # connect to end of cylinder
                    front.orig = parent.end
                    front.end = coordinate
                else: # connecting to sphere: compute point on its surface
                    front.orig = parent._sphere_surf(coordinate)
                    front.end = coordinate
                front._set_cylinder()
            else: # make a sphere: check whether distances are compatible with self.radius
                if parent.is_cylinder(): # connect to end of cylinder
                    vec = coordinate - parent.end
                    target = front.radius # desired distance
                else: # connect to other sphere, include its radius
                    vec = coordinate - parent.orig
                    target = parent.radius + front.radius # desired distance
                dist = vec.length()
                if abs(dist - target) < 0.001: # OK
                    front.orig = front.end = coordinate
                else: # recompute new_pos
                    front.orig = front.end = parent.end + (vec.norm() * target)
            flength = front.length()
            if parent.order > 0:
                front.order = parent.order # maybe by calling method
                front._sid = parent._sid
                if len(name) == 0: # copy parent name
                    front.name = parent.name
            else: # parent is soma
                front.order = 1 # first front after soma
                front._sid = parent._fid
                if (swc_type == 0) and (parent.swc_type == 1): # change swc_type for sure for soma parent
                    swc_type = 3 # dendrite
            if len(name) > 0:
                front.name = name[:BNAMELENGTH].encode('utf-8')
            front.path_length = parent.path_length + flength
            front._nid = nid
            front._fid = fid
            front._pid = parent._fid
            if swc_type > 0:
                front.swc_type = swc_type
            else:
                front.swc_type = parent.swc_type
            if not trailing:
                front.set_growing() # default
            if parent._does_storing():
                front._set_storing()
        else: # defaults to only somata
            front.radius = radius
            front.end = front.orig = coordinate
            if len(name) > 0:
                front.name = name[:BNAMELENGTH].encode('utf-8')
            front.path_length = radius
            front._nid = nid
            front._fid = fid
            front._pid = -1
            front._sid = -1 # will be replaced by index into self._neurons
            front.swc_type = 1
            front.order = 0
        front.num_children = 0
        front._cid = 0
        front.birth = self.cycle
        front.death = -1 # alive and well
        front._set_active() # enabled
        if self.verbose >= 7:
            print (self.my_id,"generated",front)
        if self.my_id > 1: # send info to admin
            if self._n_newAID >= self._new_range:
                raise OverflowError("_new_AIDs","max_active")
            if trailing:
                aid = ActiveFrontID(ID(nid,fid),b't')
            else:
                aid = ActiveFrontID(ID(nid,fid),b'e')
            self._new_AIDs[self._n_start + self._n_newAID] = aid
            self._n_newAID += 1
        # increase counters for next front
        self._f_next_indices[nid][self.my_id] += 1
        return front

    # adds a child to its parent: how this is done depends on number of existing children
    # parent should be locked
    # _trailing_axon is only method which does this directly, bypassing _add_child
    def _add_child(self,parent,child_fid):
        if self._automatic and \
            (self._front_lock[parent._nid][parent._fid] != self.my_id):
            raise BugError("_add_child","parent not locked")
        if self.verbose >= 7:
            print (self.my_id,"adding child",child_fid,"to",parent)
        if self._c_next >= self._c_max:
            raise OverflowError("_children","max_fronts")
        if parent.num_children == 0: # first child: store locally
            parent._cid = child_fid
        else:
            c_next = self._c_next
            if parent.num_children == 1: # make a linked list in children
                # store previous child first
                self._children[c_next] = Child(c_next + 1, parent._cid)
                # store new child
                self._children[c_next + 1] = Child(0, child_fid)
                parent._cid = c_next # store first link in children
                self._c_next += 2 # increment counter
            else: # extend the list
                # traverse the existing list to find end of list
                child_link = parent._cid
                while self._children[child_link].next > 0:
                    child_link = self._children[child_link].next
                # store new child
                self._children[c_next] = Child(0, child_fid) # store new child
                # link it to previous list entry
                self._children[child_link].next = c_next
                self._c_next += 1 # increment counter
        parent.num_children += 1

    # replaces old_child to new_child for parent
    def _replace_child(self,parent,old_child,new_child):
        ocid = old_child._fid
        if self.verbose >= 7:
            print (self.my_id,"replacing child",ocid,"to",new_child._fid,"for",parent._fid)
        if parent.num_children == 1: # only child: stored locally
            if parent._cid == ocid:
                # replace current record
                parent._cid = new_child._fid
                return
        elif parent.num_children > 1: # traverse children list
            child_link = parent._cid
            while True:
                if self._children[child_link]._cid == ocid:
                    self._children[child_link]._cid = new_child._fid
                    return
                if self._children[child_link].next > 0:
                     child_link = self._children[child_link].next
                else:
                    break # while True
        raise BugError("_replace_child","child to replace not found in parent list" + str(parent))

    # deletes child from parent and from children
    # Note: empty space in children is not reused
    def _delete_child(self,parent,child):
        cid = child._fid
        if self.verbose >= 7:
            print (self.my_id,"deleting child",child,"from",parent)
        if parent.num_children == 1: # only child: stored locally
            if parent._cid == cid:
                # empty current record
                parent._cid = 0
                parent.num_children = 0
                parent._set_child_retracted()
                return
        elif parent.num_children > 1: # traverse children list
            child_link = parent._cid
            prev_link = -1 # index on this entry
            while True:
                if self._children[child_link]._cid == cid:
                    next_link = self._children[child_link].next
                    # reset previous child_link to next one to skip this record
                    if prev_link < 0:
                        if parent.num_children == 2:
                            # will become parent.num_children == 1 -> store in front
                            parent._cid = self._children[next_link]._cid
                        else:
                            parent._cid = next_link
                    else:
                        if parent.num_children == 2:
                            # will become parent.num_children == 1 -> store in front
                            parent._cid = self._children[prev_link]._cid
                        else:
                            self._children[prev_link].next = next_link
                    # empty current record
                    self._children[child_link].next = 0
                    self._children[child_link]._cid = 0
                    parent.num_children -= 1
                    parent._set_child_retracted()
                    return
                if self._children[child_link].next > 0:
                    prev_link = child_link
                    child_link = self._children[child_link].next
                else:
                    break # while True
        parent.get_children(self,printing=True)
        raise BugError("_delete_child","child to delete not found in parent list " + str(child))

    # marks front as deleted by setting death to current cycle
    # removes front as child and from grid data structure, but remains in fronts
    def _delete_front(self,front,filipod=False):
        if self.verbose >= 7:
            print (self.my_id,"deleting",front,"filipod ==",filipod)
        if front.death > 0:
            return  # already deleted
        if (front.num_children > 0) and (not filipod) and (not front.is_arc()):
            raise BugError("_delete_front","retracting front with children " + str(front))
        front.death = self.cycle
        front._set_retracted()
        front._clear_active() # just to be safe
        front.clear_growing() # just to be safe
        front.clear_migrating() # just to be safe
        parent = front.get_parent(self)
        if front.has_synapse(): # remove also the synapse
            front.remove_synapse(self)
        """ causes no arc found problem
        if front.is_arc() and parent: # remove arc info if present
            # assume front retraction signals that arc info has become outdated
            index = self._get_arc(front)
            if index >= 0: # signal to remove arc info
                self._arcs[index].next_point = 0 # will be cleared by _clean_arcs
            apar = parent
            while apar: # remove arc settings iteratively
                if apar.is_arc():
                    apar._clear_arc()
                    apar = apar.get_parent(self)
                else:
                    break
        """
        # soma migration: reconnect parent of this front to its children
        # parent and grid already locked        
        if filipod: 
            children = front.get_children(self)
            if len(children) >= 1:
                pid = parent._fid
                pl_change = -front.length() # compute change in path_length
                # first child: replace deleted front in children
                child = children[0]
                self._replace_child(parent,front,child)
                child._pid = pid
                self._update_path_lengths(child,pl_change)
                # other children: regular add to parent
                for child in children[1:]:
                    child._pid = pid
                    self._add_child(parent,child._fid)
                    self._update_path_lengths(child,pl_change)
            else:
                self._delete_child(parent,front)
        elif parent: # do regular delete
            self.lock(parent)
            self._delete_child(parent,front)
            self.unlock(parent)
        # send info to admin
        if self.my_id > 1: # send info to admin
            if self._n_newAID >= self._new_range:
                raise OverflowError("_new_AIDs","max_active")
            aid = ActiveFrontID(ID(front._nid,front._fid),b'r')
            self._new_AIDs[self._n_start + self._n_newAID] = aid
            self._n_newAID += 1 # update index into new_AIDs
        # remove from grid
        id = front.get_id()
        gids = front._get_gids(self)
        count = 0
        max_count = 10 * len(gids)
        failed = False
        # This loop if most often cause of unsolvable grid lock. Therefore 
        #   an alternative is built: delete the gids later in _delayed_grid_remove
        while len(gids) > 0:
            count += 1
            if count > max_count:
                # let _proc_loop continue the grid removal
                self._deleted_gids.append([id,gids]) # for _delayed_grid_remove
                break
            for gid in gids:
                if self._grid_wlock[gid] == self.my_id: # locked by process
                    # this happens for filipod
                    self._grid_remove(gid,id)
                    self._grid_wlock[gid] = 0
                    gids.remove(gid)
                    break # changed gids
                elif not self._automatic: # interactive mode
                    self._grid_wlock[gid] = 1
                    self._grid_remove(gid,id)
                    self._grid_wlock[gid] = 0
                    gids.remove(gid)
                    break # changed gids
                # else try to get a lock from admin._lock_broker
                if (self._grid_wlock[gid] == 0) or (self._grid_wlock[gid] == 1): 
                    # currently not locked by Process
                    self._gwlock_request[self.my_id] = gid
                    wait = 0.
                    # loop will only be executed on processes, admin.my_id==0
                    while self._grid_wlock[gid] != self.my_id:
                        time.sleep(LOCKPAUSE)
                        wait += LOCKPAUSE
                        if wait > 0.1:
                            # need to stop request for lock
                            self._gwlock_request[self.my_id] = 0
                            # wait a bit
                            time.sleep(3*LOCKPAUSE)
                            wait += 3*LOCKPAUSE
                            if self._grid_wlock[gid] == self.my_id:
                                # request was honored -> go to standard behavior
                                break # failed == False
                            else: # have _delayed_grid_remove deal with this gids
                                self._deleted_gids.append([id,gids])
                                failed = True
                                break
                    if failed: # have _delayed_grid_remove deal with this gids
                        break
                    self._grid_remove(gid,id)
                    # unlock it after use
                    self._grid_wlock[gid] = 0
                    self._grid_lock_wait += wait
                    gids.remove(gid)
                    if self.verbose >= 7:
                        print (self.my_id,"_delete_front removed gid",gid,"with",len(gids),"remaining")
                        if len(gids) == 1:
                            g = gids[0]
                            print ("    _delete_front remaining gid",g,self._grid_wlock[g],self._grid_rlock[g])
                    break # changed gids
            if failed: # have _delayed_grid_remove deal with this gids
                break
        # clear front._gid
        if front._gid != 0:
            self._gids[abs(front._gid)] = 0 # no gids left
            front._gid = 0

    # Central method for NeuroDevSim
    # Tests whether a sphere or cylinder front collides using the grid and
    #   performs write locking. Returns CollisionError in case of collision.
    # Behavior is affected by self.only_first_collision.
    # Parameters:
    #   orig : Point or None : orig point for a cylinder, None: sphere tested.
    #   point : Point : point to be tested, end orig point for a cylinder or
    #                   orig for a sphere.
    #   radius : float : radius to be used for collision testing.
    #   parent : ``Front`` or None : prospective parent if defined or
    #                                migrating soma (is_migrating flag set).
    #   gids : [integer,] : starting list of gids to check, these will remain
    #                       locked for writing
    # Optional Parameters:
    #   lock: boolean : lock all indices in gids for writing, default: True
    #   squeeze : float 0.0 - 0.5: allow fronts to be closer together than rigid 
    #                             objects can be, default: 0.0 (no squeezing)
    #   trailing_axon : boolean : called by _move_to_fili with trailing axon,
    #                             default: False
    #   soma_rad : boolean : use migrating soma radius for filipods,
    #                        default: True
    # Returns nothing or CollisionError, GridCompetitionError, BugError
    def _test_collision(self,orig,point,radius,parent,gids,lock=True,\
                        trailing_axon=False,soma_rad=True,squeeze=0.):
        if self.verbose >= 7:
            print (self.my_id,"testing collision",parent,orig,point,radius,gids,trailing_axon,soma_rad)
        # go through nearby gridpoints to check for collisions
        volume = self._volume
        fronts = self._fronts
        grid = self._grid
        old_colls = False
        ids = nds_list()
        squeezing = 1.0 - squeeze # multiply min_dist by this factor
        if parent: # parent defined
            checked = [parent.get_id()] # list of front indices that were checked
            pkey = parent._key()
            # see whether previous collisions available
            if pkey in self._collisions:
                if self._collisions[pkey][0] == self.cycle: # from this cycle
                    old_colls = True
                else: # from previous cycle, will be deleted next
                    old_colls = False
                # check previous collisions first
                for item in self._collisions[pkey][1]:
                    ids.append(item[0])
        else:   # no parent
            checked = [] # list of front IDs that were already checked
        # test gids locations plus 26 adjoining grid locations
        #   create list of all indices to test
        all_gids = []
        for g0 in gids:
            for diff in volume._sdirections: # 27 offsets to grid index
                gid = g0 + diff
                if (gid not in all_gids) and (gid > 0) and (gid < volume.grid_max):
                    all_gids.append(gid)
        collision_free = True
        if not self.only_first_collision:
            collisions = nds_list() # store all colliding fronts
            distances = []  # store all distances
        # loop through gids till one can be locked and repeat.
        count = 0 # count failed cycles
        max_count = np.random.randint(40,80) # random absolute max count
        while len(all_gids) > 0: # loop through all_gids till all are processed
            if not ids: # find gid entry that can be locked
                if self.my_id <= 1: # call by admin -> no grid locking needed
                    to_remove = []
                    failed = False # cannot fail
                    for gid in all_gids:
                        ids = self._grid_get(gid)
                        to_remove.append(gid)
                        if ids: # filled grid entry found
                            for gid in to_remove:
                                all_gids.remove(gid)
                            break # test ids
                    all_gids = [] # done all without finding filled grid entry
                else: # need to obtain a grid lock
                    failed = True
                    for gid in all_gids:
                        # try to get a lock from admin._lock_broker
                        if self._grid_wlock[gid] == self.my_id: # already locked
                            ids = self._grid_get(gid)
                            all_gids.remove(gid)
                            failed = False
                            break # changed gids
                        wlock = self._automatic and lock and (gid in gids) # do we lock it for writing?
                        wait = 0.
                        if wlock: # locks gids for new front. This is done here so that we can iterate if initial lock attempt fails.
                            gwlock = self._grid_wlock[gid]
                            if gwlock == 0: 
                                # get long-lasting write+read-only lock
                                self._gwlock_request[self.my_id] = gid
                                while self._grid_wlock[gid] != self.my_id:
                                    time.sleep(LOCKPAUSE)
                                    wait += LOCKPAUSE
                                    if wait > 1.0:
                                        if self.verbose >= 7:
                                            print (self.my_id,"_test_collision GridCompetitionError 1",gid)
                                        raise GridCompetitionError(gid)
                            elif (gwlock > 0):
                                # locked for writing by other process, because
                                #  this can be reciprocal we try to have only
                                #  one of the competing processes fail. The even/odd
                                #  test based on cycle ensures that each process has
                                #  the same probability of GridCompetitionError.
                                if ((count > 11) and ((self.cycle % 2) == (self.my_id % 2))) \
                                                                                or (count > max_count):
                                    # unlock all current writing locks
                                    for gid0 in gids:
                                        if self._grid_wlock[gid0] == self.my_id:
                                            self._grid_wlock[gid0] = 0
                                    # exit
                                    if self.verbose >= 7:
                                        print (self.my_id,"_test_collision GridCompetitionError 2",gid)
                                    raise GridCompetitionError(gid)
                                else:
                                    continue # for loop
                            else:
                                continue # for loop
                        elif self._automatic:
                            if self._grid_rlock[gid] == 0: # get brief read-only lock
                                self._grlock_request[self.my_id] = gid
                                while self._grid_rlock[gid] != self.my_id:
                                    time.sleep(LOCKPAUSE)
                                    wait += LOCKPAUSE
                                    if wait > 1.0:
                                        if self.verbose >= 7:
                                            print (self.my_id,"_test_collision GridCompetitionError 3",gid)
                                        raise GridCompetitionError(gid)
                            else:
                                continue # for loop
                            # loop will only be executed on processes, admin.my_id==0
                        else:
                            constellation._grid_rlock[gid] = 1
                            wait = 0.
                        ids = self._grid_get(gid)
                        if not wlock: # unlock if read-only
                            self._grid_rlock[gid] = 0 # unlock it after use
                        self._grid_lock_wait += wait
                        all_gids.remove(gid)
                        failed = False
                        break # for loop, changed gids
                if failed: # cycle failed
                    time.sleep(LOCKPAUSE)
                    self._grid_lock_wait += 2 * LOCKPAUSE
                    count += 1
                    continue # while loop
            for id in ids:
                if id in checked: # already processed
                    continue
                if id._nid < 0: # id._nid==0 caused by recent _grid_remove
                    raise BugError("_test_collision","bad nid " + str(id._nid))
                front = fronts[id._nid][id._fid]
                if front.is_retracted():
                    # this can happen for deleted gids removal this cycle
                    continue
                # check this now to make sure we do not get fronsts[0][0]
                if id._nid == 0: # id._nid==0 caused by recent _grid_remove
                    continue
                if parent and parent._is_related(self,front,trailing_axon):
                    # only check for unrelated fronts
                    continue
                if orig: # testing point for a cylinder
                    #fdist = _cylinder_front_dist(orig,point,front,valid=True)
                    fdist = _cylinder_front_dist(orig,point,front)
                else:
                    fdist = _point_front_dist(point,front)
                #if fdist > 0.: # reliable distance
                # compute minimal distance
                if soma_rad and (front.swc_type == 12): # filipodium, check if migrating soma
                    soma = fronts[front._nid][front._sid]
                    if soma.has_migrated(): # yes -> use its radius
                        min_dist = (radius + soma.radius) * squeezing
                    else:
                        min_dist = (radius + front.radius) * squeezing
                else:
                    min_dist = (radius + front.radius) * squeezing
                if fdist <= min_dist: # collision detected
                    if parent and (front._nid == parent._nid) and \
                        (front._fid == parent._pid) and \
                        (parent.length() <= min_dist):
                        # comparing to grandparent with a parent that is shorter than min_dist
                        pass
                    else: # real collision, store it and make error
                        if parent:
                            if old_colls:
                                self._collisions[pkey][1].append((id,fdist))
                            else:
                                self._collisions[pkey] = \
                                    [self.cycle,[(id,fdist)]]
                                old_colls = True
                        if self.only_first_collision:
                            for gid0 in gids: # unlock the gids
                                if self._grid_wlock[gid0] == self.my_id:
                                    self._grid_wlock[gid0] = 0
                            raise CollisionError\
                                    (self.only_first_collision,front,fdist)
                        else:
                            collision_free = False
                            lock = False # stop locking gids
                            collisions.append(front)
                            distances.append(fdist)
                checked.append(id)
            ids = []
        # check whether some all_gids self._grid_wlock stayed locked
        for gid0 in all_gids:
            if gid0 in gids: # primary gid stays locked
                cycle
            if self._grid_wlock[gid0] == self.my_id:
                print (Fore.RED + "ERROR: _test_collision did not unlock all_gids gid" + str(gid0) + " on " + str(self.my_id) + Fore.RESET)
        
        if collision_free:
            return
        else:
            for gid0 in gids: # unlock the gids first
                if self._grid_wlock[gid0] == self.my_id:
                    self._grid_wlock[gid0] = 0
            raise CollisionError(self.only_first_collision,collisions,distances)

    # move a migrating somata front to the filipod position
    # returns coordinate and two lists of write locked gids if successful or triggers CollisionError or GridCompetitionError if no solution found
    def _move_to_fili(self,front,filipod,axon,old_gids):
        if self.verbose >= 7:
            print ("Process",self.my_id,"_move_to_fili",front,filipod)
        # in case of a collision we generate different angles relative to filipod.end
        #   using alternate_locations
        count = 0
        subtr = 1 # index of first alternate_locations is count - subtr
        if front.radius < 1.0: # search range 0.5 and 1.0 * radius
            scale = 0.5
            trials = 21
        else: # search range 0.25, 0.5 and 1.0 * radius
            scale = 0.25
            trials = 31
        if axon: # get boolean version
            trail_axon = True
        else:
            trail_axon = False
        while count < trials:
            if count == 0:
                direction = filipod.end - front.orig
                count = 1
            else:
                if (count % 10) == 1: # get fresh alternate_locations
                    points = filipod.alternate_locations(front.orig,
                                            scale*front.radius,10,random=True)
                    if count > 1:
                        subtr += 10
                        scale *= 2.0
                index = count - subtr
                count += 1
                if index < len(points): # points list may be smaller than 10
                    direction = filipod.end - points[index]
                else:
                    continue
            # compute new soma position: so that spherical boundary is at filipod.end
            length = direction.length() - front.radius
            coordinate = front.orig + (direction.norm() * length)
            result = coordinate.out_volume(self)
            if result != 0.:
                continue            
            new_gids = self._get_gids(coordinate,None,front.radius)
            gids = set(old_gids + new_gids)
            tgids = []
            # test for collisions, excluding the filipodium itself and its children
            fcount = 0
            while True: # loop for GridCompetitionError
                fcount += 1
                if fcount > 200:
                    raise GridCompetitionError(err_gid)
                try: # first test for collision of soma
                    # rarely two filipods of different somata created at same time
                    #   may be closer than they should be -> make soma_rad==False
                    self._test_collision(None,coordinate,front.radius,front,gids,\
                                    trailing_axon=trail_axon,soma_rad=(count < 10))
                    failed = False
                    break # out of while True
                except GridCompetitionError as error:
                    err_gid = error.gid
                    time.sleep(2*LOCKPAUSE) # competition with other processor
                    continue # while True
                except CollisionError as error:
                    col_front = error.collider
                    failed = True
                    break # out of while True
            if failed:
                continue # while count < trials
            if trail_axon: # make sure trailing_axon will not collide either
                # compute future origin
                a_orig = front._sphere_surf(axon.orig)
                tgids = self._get_gids(axon.orig,coordinate,0.)
                acount = 0
                while True: # loop for GridCompetitionError
                    acount += 1
                    if acount > 200:
                        raise GridCompetitionError(err_gid)
                        # raise BugError("_move_to_fili","axon endless GridCompetitionError" + str(self.my_id))
                    try:# coordinates need to be in this order to trigger invalid distance condition
                        self._test_collision(axon.orig,a_orig,axon.radius,\
                                    front,tgids,trailing_axon=True,squeeze=0.1)
                        failed = False
                        break # out of while True
                    except GridCompetitionError as error:
                        err_gid = error.gid
                        time.sleep(2*LOCKPAUSE) # competition with other processor
                        continue # while True
                    except CollisionError as error:
                        col_front = error.collider
                        #   clear gids locked by filipod call to _test_collision
                        for gid in gids:
                            if self._grid_wlock[gid] == self.my_id:
                                self._grid_wlock[gid] = 0
                        raise error # this cannot be fixed
                if failed:
                    continue # while count < trials
            # no CollisionErrors...
            if self.verbose >= 7:
                    print ("Process",self.my_id,"_move_to_fili",front._fid,filipod._fid,coordinate)
            return coordinate, new_gids, tgids
        # no acceptable location found: raise CollisionError with last colliding front
        itd = front.front_distance(col_front,point=True)
        raise CollisionError(self.only_first_collision,col_front,itd[0])

    # make a trailing axon
    # returns list of write locked gids
    def _trailing_axon(self,front,axon):
        my_id = self.my_id
        if self.verbose >= 7:
            print ("Process",my_id,"_trailing_axon",front,axon)
        # check whether we moved enough: new axon should be longer than its radius
        vec = axon.orig - front.orig
        if vec.length() < axon.radius:
            if self.verbose >= 7:
                print ("skipping _trailing_axon",vec.length(),axon.radius)
            return [] # do not make an axon
        # make the new axon front ending at origin of previous axon
        new_axon = self._enter_front(front,True,axon.orig,axon.radius,"axon",2,\
                                                trailing=True)
        # add axon as child to new_axon
        new_axon._cid = axon._fid
        new_axon.num_children = 1
        # replace child in soma
        self._replace_child(front,axon,new_axon)
        # make axon a child of new_axon
        axon._pid = new_axon._fid
        # update path_lengths of all axons
        self._update_path_lengths(axon,new_axon.length())
        # give same active status as axon, but is not growing
        if not axon.is_active():
            new_axon._clear_active() # active by default
        self._newfs.append(new_axon) # store for this manage_front call
        # get grid ids, presume they have already been locked by migrate_soma
        gids = new_axon._get_gids(self)
        id = new_axon.get_id()
        tgids = [] # newly locked gids
        # update the grid: all these gids should be locked
        for gid in gids:
            if self._grid_wlock[gid] != self.my_id: # not yet locked
                if self._automatic:
                    self._gwlock_request[self.my_id] = gid
                    wait = 0.
                    while self._grid_wlock[gid] != self.my_id:
                        time.sleep(LOCKPAUSE)
                        wait += LOCKPAUSE
                        if wait > 2.0:
                            raise BugError("_test_collision","write _trailing_axon for a second on " + str(self.my_id) + " for grid " + str(gid))
                    self._grid_lock_wait += wait
                else:
                    self._grid_wlock[gid] = 1
                tgids.append(gid)
            self._grid_set(gid,id)
        new_axon._store_gids(self,gids)
        if self.verbose >= 7:
            print ("Process",self.my_id," makes trailing_axon",new_axon._fid)
        return tgids

    # recursive routine over all children:
    #    due to soma migration path_lengths of all existing children changed
    def _update_path_lengths(self,front,change):
        front.path_length += change
        children = front.get_children(self)
        for child in children:
            self._update_path_lengths(child,change)

    ### Grid routines

    # Returns all ID values
    # Returns:
    #  [id,]: nds_list of all front indices close to grid point, can be empty
    def _grid_get(self,gid,time_out=0.1):
        if (self.my_id > 1) and (self._grid_rlock[gid] != self.my_id) and \
                                (self._grid_wlock[gid] != self.my_id):
            raise BugError("_grid_get","grid entry not locked "  + str(self.my_id) + " " + str(gid) + " " +  str(self._grid_rlock[gid])+ " " +  str(self._grid_wlock[gid]))
        if self.verbose >= 7:
            print (self.my_id,"_grid_get",gid)
        exid = self._grid[gid]
        if exid == 0: # empty
            return []
        ids = nds_list()
        while True:
            for i in range(self._extra_size):
                gfid = self._grid_extra[exid + i]
                if gfid._nid > 0: # valid entry
                    ids.append(copy.copy(gfid))
                    # weird error where despite proper locking last entry is
                    #   overwritten by new exid, cache synchronization problem?
                    if ids[-1]._nid <= 0:
                        # solve bug
                        exid = - ids[-1]._nid
                        ids.pop() # remove offending gfid
                        continue
                    if i == self._extra_size_1: # last entry of block filled
                        return ids
                    else:
                        continue
                if gfid._nid == 0: # end of used block
                    return ids
                elif gfid._nid < 0: # link to next block
                    exid = -gfid._nid
    
    def _grid_set(self,gid,id):
        if (self.my_id > 1) and (abs(self._grid_wlock[gid]) != self.my_id):
            raise BugError("_grid_set","grid entry not locked "  + str(self.my_id) + " " + str(gid) + " " +  str(self._grid_wlock[gid]))
        # now request immediate write lock
        if self.my_id > 1:
            self._gwlock_request[self.my_id] = -gid
            wait = 0
            while self._grid_wlock[gid] != -self.my_id:
                time.sleep(LOCKPAUSE)
                wait += LOCKPAUSE
                if wait > 1.0:
                    raise BugError("_grid_set","write waited for a second on " + str(self.my_id) + " for grid " + str(gid))
            self._grid_lock_wait += wait
        exid = self._grid[gid]
        g_next = self._g_next
        if self.verbose >= 7:
            print ("_grid_set",id,exid,g_next)
        if exid == 0: # empty -> create new grid_extra entry
            if g_next == self._g_max:
                raise OverflowError("_grid_extra","max_fronts")
            self._grid[gid] = g_next # replace with link to new block
            exid = g_next # index into grid_extra
            self._g_next += self._extra_size # update g_next
        # enter into grid_extra
        id2 = None
        while True:
            if exid >= self._max_extra:
                raise OverflowError("grid_extra","grid_extra")
            for i in range(self._extra_size_1): # first entries are simple
                n = exid + i
                gfid = self._grid_extra[n]
                if gfid == id: # already entered
                    if self.my_id != 1:
                        self._grid_wlock[gid] = self.my_id # reset standard write-lock
                    return
                elif gfid._nid == 0: # empty -> enter new id
                    if id2: # only possible for new block -> i == 0
                        if i > 0:
                            raise BugError("_grid_set","id2 in old block")
                        self._grid_extra[n] = id2
                        self._grid_extra[n+1] = id
                        id2 = None
                    else:
                        self._grid_extra[n] = id
                    if self.my_id != 1:
                        self._grid_wlock[gid] = self.my_id # reset standard write-lock
                    return
                elif gfid._nid < 0: # only last entry can be link to next block
                    raise BugError("_grid_set","early negative")
            # now process last entry in block
            n = exid + self._extra_size_1
            gfid = self._grid_extra[n]
            if gfid._nid < 0: # link to next block
                exid = -gfid._nid # jump to next block
            else: # may need to create new block
                if gfid._nid > 0: # previous value
                    id2 = copy.copy(gfid)
                else: # gfid is empty
                    self._grid_extra[n] = id
                    if self.my_id != 1:
                        self._grid_wlock[gid] = self.my_id # reset standard write-lock
                    return
                # create new block
                self._grid_extra[n] = ID(-g_next,0) # enter link to new block
                exid = g_next # jump to next block
                self._g_next += self._extra_size # update g_next

    # tries to remove front index from grid_extra
    # recuperates space by putting final entry in the new empty location
    def _grid_remove(self,gid,id):
        if self._grid_wlock[gid] != self.my_id:
            raise BugError("_grid_remove","grid entry not locked " + str(self.my_id) + " " + str(gid) + " " +  str(self._grid_wlock[gid]))
        # now request immediate write lock
        if self.my_id > 1:
            self._gwlock_request[self.my_id] = -gid
            wait = 0
            while self._grid_wlock[gid] != -self.my_id:
                time.sleep(LOCKPAUSE)
                wait += LOCKPAUSE
                if wait > 1.0:
                    raise BugError("_grid_set","write waited for a second on " + str(self.my_id) + " for grid " + str(gid))
            self._grid_lock_wait += wait
        if self.verbose >= 7:
            print (self.my_id,"_grid_remove",gid,id)
        exid = self._grid[gid]
        if exid == 0: # empty -> error
            raise BugError("_grid_remove","gid empty " + str(gid))
        ind = -1  # index of id if found
        last = False # found last entry
        while True:
            for i in range(self._extra_size):
                n = exid + i
                gfid = self._grid_extra[n]
                if self.verbose >= 7:
                    print (self.my_id,"_grid_remove loop",i,n,gfid,ind,last)
                if gfid == id: # found, now find last entry
                    prev_ind = ind = n
                    prev_gfid = ID(0,0) # in case this is the last entry
                    if (i == self._extra_size_1): # is last entry
                        last = True
                elif gfid._nid < 0: # link to next block, should also be i == EXTRASIZE_1
                    exid = -gfid._nid
                    break # for loop
                elif gfid._fid == 0: # empty: previous one was last entry
                    if ind >= 0: # found fid
                        last = True
                        break
                    else:
                        if self.verbose >= 2:
                            print(Fore.MAGENTA + "_grid_remove id not found",gid,id,Fore.RESET)
                        if self.my_id != 1:
                            self._grid_wlock[gid] = self.my_id # reset standard write-lock
                        return  # not found
                else: # regular entry
                    prev_gfid = gfid # potential last entry
                    prev_ind = n
                    if (i == self._extra_size_1):
                        if (ind >= 0): # last entry & found fid
                            last = True
                            break
                        else:
                            if self.verbose >= 2:
                                print(Fore.MAGENTA + "_grid_remove id not found",gid,id,Fore.RESET)
                            if self.my_id != 1:
                                self._grid_wlock[gid] = self.my_id # reset standard write-lock
                            return  # not found
            # if existing, link to next block is not removed: will be used
            #   again if this gid list fills up again. Otherwise next block
            #   would never be used again as there is no recycling.
            if last: # found fid and last entry
                self._grid_extra[ind] = prev_gfid
                if prev_ind != ind:
                    self._grid_extra[prev_ind] = ID(0,0) # clear original location of last entry
                if self.verbose >= 7:
                    print (self.my_id,"_grid_remove completed")
                if self.my_id != 1:
                    self._grid_wlock[gid] = self.my_id # reset standard write-lock
                return

    ### gid routines
    
    # get grid ids that will need to be updated for a front from orig to end or
    #   orig only (sphere, radius should be provided also)
    # returns a list of gid positions
    def _get_gids(self,orig,end,radius):
        # find grid positions for origin and end
        gid0 = orig._grid(self._volume)
        gids = [gid0] # list of grid positions
        hg_step = self._volume.half_grid_step
        if end: # cylinder
            gid1 = end._grid(self._volume)
            if gid1 != gid0:
                gids.append(gid1)
                # interpolate intervening point(s) if needed
                vec = end - orig
                lvec = vec.length()
                if lvec >= self._volume.grid_step:
                    ratio = int(lvec / hg_step)
                    nvec = vec.norm()
                    for n in range(1,ratio):
                        ipol = orig + nvec * n * hg_step
                        gid = ipol._grid(self._volume)
                        if gid not in gids:
                            gids.append(gid)
        elif radius >= hg_step: # add points for large sphere radius
            gids = self._get_sphere_gids(orig,radius)
            if gid0 not in gids:
                gids = [gid0] + gids
        return gids

    # get grid ids that will need to be updated for a spherical front
    # can be called with different radius value
    # returns a list of gid positions
    def _get_sphere_gids(self,orig,radius):
        gids = []
        if radius >= self._volume.grid_step: # very large soma
            large = True
            ext_rad = radius + self._volume.half_grid_step # include grid points close to outside
            grid_step = self._volume.grid_step
        else:
            large = False
        # need to generate 6 grid entries relative to real center position
        sign = -1
        for i in range(2): # lfd versus rbu directions
            for j in range(3): # each coordinate
                if j == 0:
                    p = orig + Point(sign * radius, 0., 0.)
                elif j == 1:
                    p = orig + Point(0., sign * radius, 0.)
                else:
                    p = orig + Point(0., 0., sign * radius)
                gid0 = p._grid(self._volume)
                if (gid0 not in gids) and (gid0 > 0) and \
                                        (gid0 < self._volume.grid_max):
                    gids.append(gid0)
                if large: # large soma: include additional grid points
                    # use 26 adjoining grid locations
                    for diff in self._volume._sdirections[1:]:
                        gid = gid0 + diff
                        if (gid not in gids) and (gid >= 0) and (gid < self._volume.grid_max):
                            # test whether inside or close to outside of sphere
                            line = self._grid_point(gid) - orig
                            if line.length() <= ext_rad:
                                gids.append(gid)
            sign = 1
        return gids
    
    # return [x,y,z] value of grid index
    def _grid_xyz(self,gid):
        x = (gid - 1) // self._volume.grid_size2
        rest = gid - 1 - (x * self._volume.grid_size2)
        sim_volume = self._volume.sim_volume
        grid_step = self._volume.grid_step
        return [sim_volume[0][0] + x * grid_step,
                sim_volume[0][1] + rest // self._volume.grid_size1 * grid_step,
                sim_volume[0][2] + rest % self._volume.grid_size1 * grid_step]

    # return Point value of grid index
    def _grid_point(self,gid):
        x = (gid - 1) // self._volume.grid_size2
        rest = gid - 1 - (x * self._volume.grid_size2)
        sim_volume = self._volume.sim_volume
        grid_step = self._volume.grid_step
        return Point (sim_volume[0][0] + x * grid_step,
                      sim_volume[0][1] + rest // self._volume.grid_size1 * grid_step,
                      sim_volume[0][2] + rest % self._volume.grid_size1 * grid_step)

    # perform grid remove that _delete_front could not do in time
    # data stored in self._deleted_gids as: [id,[gid0,gid1,]]
    # if single = True it will return after performing a single update
    def _delayed_grid_remove(self,single):
        if self.verbose >= 7:
            print ("Process",self.my_id,"_delayed_grid_remove",single,len(self._deleted_gids))
        if single: # keep the search short
            max_n = min(5,len(self._deleted_gids)) # limit number of searches
        else:
            max_n = len(self._deleted_gids)
            to_delete = [] # unsorted list of completed entries
        for n in range(max_n):
            id = self._deleted_gids[n][0]
            gids = self._deleted_gids[n][1]
            for k in range(2 * len(gids)):
                for gid in gids:
                    if self._grid_wlock[gid] == self.my_id: # locked by process
                        # this happens for filipod
                        self._grid_remove(gid,id)
                        gids.remove(gid)
                        break # changed gids
                    # else try to get a lock from admin._lock_broker
                    #  this call may overlap with admin._retract_branches and 
                    #  therefore lock by process 1 should be respected
                    if self._grid_wlock[gid] == 0: 
                        # currently not locked by Process
                        self._gwlock_request[self.my_id] = gid
                        wait = 0.
                        # loop will only be executed on processes, admin.my_id==0
                        while self._grid_wlock[gid] != self.my_id:
                            time.sleep(LOCKPAUSE)
                            wait += LOCKPAUSE
                            if wait > 1.0:
                                raise BugError("_delayed_grid_remove","waited for a second " + str(self.my_id) + " " + str(gid))
                        self._grid_remove(gid,id)
                        # unlock it after use
                        self._grid_wlock[gid] = 0
                        self._grid_lock_wait += wait
                        gids.remove(gid)
                        if single:
                            if len(gids) == 0: # emptied gids
                                del self._deleted_gids[n] # remove entry
                            return
                        else:
                            if len(gids) == 0: # emptied gids
                                to_delete.append(n)
                            break # gids changed
                if len(gids) == 0: # emptied gids
                    break
        # remove completed items
        if not single:
            to_delete.sort(reverse=True)  # order of appends can be random
            for index in to_delete: # remove higher indices first
                del self._deleted_gids[index] # remove entry            
    
    # print complete gid entry
    def _print_gid(self,gid,procID=False):
        exid = self._grid[gid]
        new_exid = True
        while new_exid:
            if procID:
                print (self.my_id,"calls",gid,"->",exid)
            else:
                print (gid,"->",exid)
            new_exid = False
            for i in range(self._extra_size):
                n = exid + i
                gfid = self._grid_extra[n]
                print ("  ",i,n,gfid)
                if gfid._nid < 0:
                    exid = -gfid._nid
                    new_exid = True
                    break # for loop
                    
    ### Arc routines
    
    # store arc in shared array
    # arguments:
    #   caller : ``Front`` : Front that made the arc
    #   sphere : ``Front`` : sphere Front around which the arc goes
    #   angle : float : angle covered in degrees
    #   count : integer : number of ``Point`` returned each cycle
    #   points : [``Point``,] : list of ``Point``
    # returns:
    #   index : integer : index into self._arcs
    def _store_arc(self,caller,sphere,angle,count,points):
        if self.verbose >= 7:
            print ("Process",self.my_id,"_store_arc",caller._fid,len(points))
        num_points = len(points) # number of points to be stored
        if num_points > ARCLENGTH:
            raise ValueError("length of arc","less or equal to " + str(ARCLENGTH))
        # find empty slot
        arcn = -1
        for n in range(self._max_arcs):
            if self._empty_arcs[n]:
                arcn = self._a_start + n
                self._empty_arcs[n] = False
                break
        if arcn < 0:
            raise OverflowError("_arcs","max_arcs")
        # Store in self._arcs long-term
        if caller.swc_type == 12: # filipod may be deleted 
            soma = caller.get_soma(self)
            caller_fid = soma._fid # use soma instead
        else:
            caller_fid = caller._fid
        neuron = caller.get_neuron(self)
        # initialize Arc data structure
        arc = self._arcs[arcn] # get next free entry in shared array 
        arc.cycle = self.cycle
        arc.complete = 0 # cycle when all entries has been used
        arc.neuron_id = neuron._neid
        arc.caller_fid = caller_fid # front that made the arc or its soma
        arc.last_fid = caller_fid
        arc.sphere = sphere.get_id() # sphere front around which the arc goes
        arc.arc_angle = angle # angle covered in degrees
        arc.num_points = num_points # number of points stored
        arc.count = count # number of points to return each cycle
        arc.next_point = 0 # next unused point, updated by _find_point and _next_point
        # enter points in arc_points data structure
        #   arc.index is fixed, set in Constellation.__init__
        for n in range(num_points):
            self._arc_points[arc.index + n] = points[n]
        if self.verbose >= 7:
            print (self.my_id,"_store_arc",caller._fid,caller.end,":",nds_list(points))
        return arcn
            
    # find arc for continuation of existing arc
    # arguments:
    #   front : ``Front`` : front with is_arc()==True
    # returns:
    #   index : integer : index into self._arcs or -1 if failed
    def _get_arc(self,front):
        if self.verbose >= 7:
            print ("Process",self.my_id,"_get_arc",front)
        if not front.is_arc():
            raise BugError("_get_arc","not an arc " + str(front))
        fid = front._fid
        key = front._key()
        neuron = front.get_neuron(self)
        neid = neuron._neid
        # initial find: create list of arcs belonging to this neuron and
        #   test for last_fid
        neuron_arcs = []
        for n in range(self._tot_arcs):
            arc = self._arcs[n]
            if self.verbose >= 7:
                print (self.my_id,"_get_arc",fid,"evaluating last_fid for",n,arc)
            if arc.neuron_id != neid:  # empty arcs have neid == 0
                continue
            if arc.complete > 0: # finished
                continue
            if arc.last_fid == fid:
                if key in self._used_arcs: # local and temporary
                    self._used_arcs[key].append(n)
                else:    
                    self._used_arcs[key] = [n]
                if self.verbose >= 7:
                    print (self.my_id,"arc found on last_fid",fid,n)
                return n
            neuron_arcs.append(n)
        # not found on last_fid -> need to use caller_fid, find caller:
        #   find front that created the arc: ancestor that is not an arc
        parent = front.get_parent(self)
        if (front.swc_type == 12) and \
                ((parent.swc_type == 12) or (parent.swc_type == 1)):
            # caller_id is that of the soma
            soma = front.get_soma(self)
            caller_fid = soma._fid # use soma instead
            caller_cycle = 0 # not known
        else: # find parent without arc, may be the soma
            caller = front
            while caller.is_arc():
                caller = caller.get_parent(self)
                if not caller: 
                    raise BugError("_get_arc","calling front not found for " + str(front))
            caller_fid = caller._fid
            caller_cycle = caller.birth
        if self.verbose >= 7:
            print (self.my_id,"_get_arc",front.get_id(),"looking for",caller_fid,caller_cycle)
        # now go through all neuron related arcs: 
        for n in neuron_arcs:
            arc = self._arcs[n]
            if self.verbose >= 7:
                print (self.my_id,"_get_arc",fid,caller_fid,"evaluating caller_fid for",arc)
            if arc.cycle <= caller_cycle: # too old
                continue
            if arc.caller_fid != caller_fid:
                continue
            # a caller can have muliple arcs (in case of branching)
            # -> check whether front coordinate is in points list: should
            #    be last returned
            for k in range(arc.index,arc.index + arc.next_point + 1):
                if self.verbose >= 7:
                    print (self.my_id,"testing",k,fid,caller_fid,front.end, self._arc_points[k])
                if front.end == self._arc_points[k]:
                    # found it: store in _used_arcs
                    if key in self._used_arcs: # local and temporary
                        self._used_arcs[key].append(n)
                    else:    
                        self._used_arcs[key] = [n]
                    if self.verbose >= 7:
                        print (self.my_id,"arc found on caller_fid",fid,n)
                    return n
        return -1 # failed
        
    # returns nds_list of all arc points
    def _get_arc_points(self,arc):
        points = nds_list()
        for n in range(arc.num_points):
            points.append(self._arc_points[arc.index + n])
        return points

    # update self._empty_arcs: called after admin.import_simulation
    def _update_empty_arcs(self):
        if self.verbose >= 7:
            print ("Process",self.my_id,"_update_empty_arcs")
        for n in range(self._max_arcs):
            arc = self._arcs[self._a_start + n]
            if arc.cycle or arc.neuron_id: # entry present
                self._empty_arcs[n] = False
    
    # remove unused or completed arcs from storage, called at end of cycle
    def _clean_arcs(self):
        if self.verbose >= 7:
            print ("Process",self.my_id,"_clean_arcs")
        for n in range(self._max_arcs):
            if not self._empty_arcs[n]:
                arc = self._arcs[self._a_start + n]
                if arc.next_point == 0: # unused arc
                    arc.cycle = 0 # clears entry
                    self._empty_arcs[n] = True
                    continue
                if (self.cycle - arc.cycle) > 10: # really old
                    arc.cycle = 0 # clears entry
                    self._empty_arcs[n] = True
                    continue
                if (arc.complete > 0) and ((self.cycle - arc.complete) > 1): 
                    # arc finished while ago
                    arc.cycle = 0 # clears entry
                    self._empty_arcs[n] = True
                    continue
    

    # print runtime statistics for processor
    def _print_stats(self,all=True):
        # stagger printing by my_id
        time.sleep(LOCKPAUSE * self.my_id * 50)
        print ("Process",self.my_id,"cumulative run time {:8.2f}".format(self._proc.run_time))
        if all:
            print ("Process",self.my_id,"cumulative idle time {:8.2f}".format(self._proc.wait_time))
            print ("Process",self.my_id,"cumulative front lock wait time {:6.4f}".format(self._front_lock_wait))
            print ("Process",self.my_id,"cumulative grid lock wait time {:6.4f}".format(self._grid_lock_wait))

# Instruction structure used to communicate between admin and processes
# Main instruction is provided by clock set by admin:
#   0: simulation in waiting state
#   >0: == current cycle of process: wait till other processes have finised
#       > current cycle of process: new cycle started, update local cycle
#   -1: finished simulation loop: winter sleep
#   -10: print statistics with constellation._print_stats

# ActiveFrontID: Data structure that communicates from processes to admin:
#   ID combined with a status flag and optionally an index
# status codes:
#   'a': make front active
#   'd': retract branch starting at front
#   'e': new front to be stored
#   'g': make front inactive, till_cycle in index makes growing
#   'i': make front inactive, till_cycle in index makes active
#   'm': make front inactive, till_cycle in index makes migrating
#   'p': the id of the front that was called (always first one in new_AIDs, special use of index)
#   'r': retracted front to be updated in db
#   's': new substrate to be stored, index in ID._fid
#   't': a trailing axon was created
#   'y': new synapse to be stored, index in ID._fid
#   'z': synapse to be removed, index in ID._fid
#   '0': empty_AID
#   '1': master admin <-> slave1 range communication at begin simulation_loop
#   '2': master admin <-> slave1 communication at begin/end of simulation_loop
# index for 'p':
#     0: manage_front not started or still running
#    >0: manage_front ended, index of last new entry made in self._new_AIDs

class ActiveFrontID(Structure):
    _fields_ = [('ID', ID), ('status', c_char), ('index', c_int)]

    def __str__(self):
        return "type " + str(self.ID._nid) + ", index " + str(self.ID._fid) + ", status " + str(self.status)

### Process methods

# Proc_agent: parallel processing
class Proc_agent(object):

    def __init__(self,my_id,verbose,master_constell):
        self.my_id = my_id
        self.verbose = verbose
        self.master_clock = master_constell[5]
        # most of master_constell is accessed from self.constellation
        self._num_procsP2 = master_constell[9]
        self._proc_status = master_constell[26]
        self._proc_status[my_id] = 1 # signal that this proc is online
        self._volume = master_constell[34]
        self._fronts = master_constell[0]
        self._actives = master_constell[1]
        act_range = master_constell[13]
        self._new_range = int(act_range * 1.5)
        self._debug = master_constell[24]
        # array to communicate results of current cycle to Admin, ordered as:
        #   0: ID of first parent front with index==n0: number of entries 
        #                                              for this parent front
        #   1 - (n0 - 1): data about new fronts, etc defined above ActiveFrontID
        #   n0: ID of second parent front with index==n1: number of entries 
        #                                              for this parent front
        #   (n0 + 1) - (n1 - 1): data about new fronts, etc 
        #   ...
        # index of parent front is used to signal completion when > 0
        self._new_AIDs = master_constell[2] # all needed
        # processor specific range in new_AIDs shared array
        self._n_start = (my_id - 2) * int(act_range * 1.5) # starting index into new_AIDs
        self._grid_wlock = master_constell[8]
        self._gids = master_constell[19]
        ## local lists ##
        #  managed fronts
        # constellation for public use
        self.constellation = Constellation(self,master_constell)
        # timing counters
        self.cycle_start = 0.
        self.cycle_run_end = 0.
        self.run_time = 0. # cumulative run times
        self.wait_time = 0. # cumulative waiting times

# Processes return status in proc_status RawArray
# Possible values:
# 0: after initialization
#  0: before any cycling
#  1: finished _update_empty_arcs during import_simulation
# 2-20: processing:
#   2: started cycle
#   3: started a manage_front call
#   10: completed a manage_front call
#   20: finished cycle
# Admin slaves have their own codes, listed there

    # To prevent overwriting self._new_AIDs before admin has had time to read it,
    #  the index is reset to the beginning of the processor range (self._n_start)
    #  every odd cycle, even cycles continue to fill up self._new_AIDs
    def _proc_loop(self):
        if self.verbose >= 5:
            print ("Process",self.my_id,"starting _proc_loop")
        cycle = self.constellation.cycle
        my_id =  self.my_id
        reserve = my_id + self._num_procsP2 - 2 # index possible prefetched ID
        first_stats = True
        odd_cycle = True # number 1 is odd
        while True: # endless loop
            new_cycle = self.master_clock.value
            if new_cycle > cycle: # start new cycle
                if self._debug and ((new_cycle - cycle) > 1):
                    raise BugError("_proc_loop","skipped cycle " + str(cycle))
                # new cycle started
                self.cycle_start = time.time()
                self._proc_status[my_id] = 2
                if self.cycle_run_end > 0.:
                    self.wait_time += self.cycle_start - self.cycle_run_end
                cycle = self.constellation.cycle = new_cycle
                if self.verbose >= 6:
                    print (self.my_id,"starting cycle",cycle,self.master_clock.value)
                # start cycle at beginning of range in self._new_AIDs 
                #   Admin will have reset self._new_AIDs[self._n_starts[pid]]
                self.constellation._n_newAID = 0 # counter is in constellation
                self.constellation._used_arcs = {} # clear dict
                if self._debug:
                    cleared_gids = [] # all gids unlocked here
                # now do the work: get a front and manage it
                while True:
                    id = self._actives[my_id]
                    if id._nid == 0: # try reserve
                        id = self._actives[reserve]
                        if id._nid == 0: # also empty
                            if self.constellation._deleted_gids:
                                # Try to further remove one remaining grid entry
                                self.constellation._delayed_grid_remove(True)
                            else:
                                time.sleep(PAUSE)
                            continue
                        else:
                            res_id = True
                    else:
                        res_id = False
                    if id._nid > 0: # a new front
                        self._proc_status[my_id] = 3 # started
                        try:
                            front = self._fronts[id._nid][id._fid]
                        except Exception as error:
                            print (error)
                            raise BugError("_proc_loop","bad id error" + str(id) + " at " + str(cycle))
                        # check whether front has been initialized
                        count = 0
                        # fix weird (cache? predictive branching?) bug where an initialized front is initially returned as empty
                        while front._sid == 0:
                            if count > 10:
                                raise BugError("_proc_loop","bad id " + str(id) + " at " + str(cycle) + ", " + str(self.constellation[id._nid][id._fid]))
                            time.sleep(LOCKPAUSE)
                            front = self._fronts[id._nid][id._fid]
                            count += 1
                        if res_id:
                            self._actives[reserve] = empty_ID # reset to zero
                        else:
                            self._actives[my_id] = empty_ID # reset to zero
                        if front.is_active():
                            if self.verbose >= 6:
                                print (my_id,self.constellation._n_newAID,"managing",front)
                            self.constellation.lock(front) # will trigger LockError if not possible
                            gid_locked = front._gid < 0 # will be reset by Admin
                            ### compute synaptic input if present
                            try:
                                if front._yid > 0: # postsynapse present:
                                    front._pre_manage_front\
                                                (self.constellation)
                            except: # no _yid attribute
                                pass
                            self.constellation._manage = front # keep track for error checking
                            self.constellation._newfs = [] # clear
                            # send back front processed
                            if self.constellation._n_newAID >= self._new_range:
                                raise OverflowError("_new_AIDs","max_active")
                            start_n = self.constellation._n_newAID # need this later
                            self._new_AIDs[self._n_start + start_n] = ActiveFrontID(front.get_id(),b'p',0)
                            # increase local counter
                            self.constellation._n_newAID += 1
                            front.manage_front(self.constellation)
                            if self.verbose >= 6:
                                print (my_id,self.constellation._n_newAID,"end manage_front",front._fid,gid_locked)
                            # update flags depending on methods called in manage_front
                            if front.is_migrating():
                                if front._has_moved_now():
                                    front._set_moved()
                                else:
                                    front._clear_moved()
                            self.constellation.unlock(front)
                            # first set next one to zero so that Admin stops here
                            self._new_AIDs[self._n_start + \
                                        self.constellation._n_newAID].index = 0
                            # now store new self.constellation._n_newAID: signal Admin
                            #   that this can be processed
                            self._new_AIDs[self._n_start + start_n].index = self.constellation._n_newAID
                            self._proc_status[my_id] = 10 # finished front
                            # unlock gids that where locked by Admin
                            if gid_locked: # Admin locked gids
                                ngid = abs(front._gid)
                                for i in range(ngid+1, ngid+1+self._gids[ngid]):
                                    gid = self._gids[i]
                                    if self._grid_wlock[gid] == my_id: # unlock
                                        self._grid_wlock[gid] = 0
                                        if self._debug:
                                            cleared_gids.append(gid)
                                # Admin will reset front._gid
                        else:
                            raise BugError("_proc_loop","on " + str(my_id) + " at " + str(cycle) + " not active " + str(front))
                    elif id._nid < 0: # done for this cycle                        
                        # perform clean ups
                        if self.verbose >= 6:
                            print (my_id,"starting end cycle")
                        while self.constellation._deleted_gids:
                            # Remove all remaining grid entries
                            self.constellation._delayed_grid_remove(False)
                            if self.verbose >= 7:
                                print (self.my_id,"_deleted_gids",self.constellation._deleted_gids)
                        self._proc_status[my_id] = 20 # finished cycle
                        self._actives[my_id] = empty_ID # reset to zero
                        if self.constellation._used_arcs: # remove unused arcs
                            self.constellation._clean_arcs()
                        if self.verbose >= 6:
                            print (self.my_id,"finishing cycle",cycle)
                        self.cycle_run_end = time.time()
                        self.run_time += self.cycle_run_end - self.cycle_start
                        if self._debug:
                            print (my_id,"cleared gids:",cleared_gids)
                        break # while True 
            elif new_cycle >= 0: # waiting for next instruction
                time.sleep(PAUSE)
                continue
            elif new_cycle == -5: # after import_simulation
                self.constellation._update_empty_arcs()
                self._proc_status[my_id] = 1 # done with _update_empty_arcs
            elif new_cycle == -10:
                # capture exceptional use
                if first_stats:
                    self.constellation._print_stats()
                    first_stats =  False # one time only
                else:
                    time.sleep(0.1)
            elif new_cycle == -1: # simulation loop finished
                time.sleep(0.1)
                continue
            else:
                raise BugError("_proc_loop","unknown cycle " + str(cycle))

def _proc_init(my_id,verbose,master_constell,seed):
    if not seed: # generate random seed
        np.random.seed()
        seed = np.random.randint(1000000)
    lseed = seed * my_id + 1313 * my_id # make it predictable but different on each process
    np.random.seed(lseed)
    if verbose >= 6:
        print ("Process",my_id,"online, seed",lseed)
    proc = Proc_agent(my_id,verbose,master_constell)
    proc._proc_loop()
    if verbose >= 6:
        print ("Process",my_id,"finishes")
    return

### Admin methods
# Admin_agent: initializes and runs simulation
class Admin_agent(object):
    """ Main class implementing the conductor that controls NeuroDevSim simulations.
    
    Only the **attributes** listed below can be read. Some can be changed as indicated, otherwise DO NOT change these attributes.
    
    Attributes:
        constellation : ``Constellation`` : the simulation constellation, needed in interactive mode.
        cycle : integer : simulation cycle.
        importable_db : boolean : make the database suitable for ``import_simulation`` method. **Can be changed**, default False.
        seed : integer : seed value used, either set by user or by ``Admin_agent``.
        total_fronts : integer : total number of fronts made during the simulation, updated at end of each cycle.
        verbose : integer : verbose level as defined above. **Can be changed** but this affects ``Admin_agent`` only, processes verbose level is not changed.
        
    Parameters
    ----------
        num_procs : integer >= 0: number of extra processes to start for parallel simulation. A 0 value activates an interactive session with no parallel computing.
        db_name : string : name of the sqlite database that stores simulation results. Can be empty for an interactive session.
        sim_volume : list of 2 lists or list of 3 floats: coordinates specifying simulation volume in µm as: [[left, front, bottom], [right, back, top]]; if only all positive [right, back, top] is provided it is assumed to be [[0.,0.,0.], [right, back, top]]
        neuron_types : list of strings: list with names of all ``Front`` subclasses that can be used.
        general optional1:
        debug : boolean : perform extra error checking that will slow down simulation, default False.
        num_admins : integer 1-3 : run extra admin core if plot==False, default 1.
        seed : integer > 0: seed for random number generator, default: None
        verbose : integer 0-7 : increasing amounts of information printed to console, see documentation above for details, 0: no output, default 2.
        preset array sizes optional2:
        max_fronts : integer > 0: maximum number of fronts for each ``Front`` subclass, default 20000.
        max_active : integer > 0 : maximum number of growing fronts each cycle, default 2000.
        max_arcs : integer > 0 : maximum number of arcs stored, default 40 per process.
        max_substrate : integer > 0 : maximum number of ``Substrate``, default 1000.
        max_synapse : integer > 0 : maximum number of ``Synapse``, default 1000.
        grid_step : float > 1.0 : distance between grid points, default 20 µm.
        grid_extra : integer > 0 : size of blocks for grid entries, default 10.
        dict_size : integer > 0 : size of blocks for local grid storage, default 20
        plotting in notebooks optional3:
        plot : boolean : if True an interactive 3D plot is generated, default False.
        time_lapse : boolean : plots a time lapse sequence after simulation, overrides *plot* and ``no_axis=True``, default False.
        axis_ticks : boolean : show axis ticks, default True.
        azim : float : azimuth in degrees of camera, default -60.
        box : box format [[left, front, bottom], [right, back, top]] to plot, allows to zoom in, default full *sim_volume.*
        color_scheme : integer 0-3 : controls how colors change: 0: every neuron has a different color; 1: neurons of same type have same color, different types have different colors; 2: different branches in a neuron have different colors, based on branch_name; 3: color set by front attribute, default 0.
        color_data : list : [front attribute name, min value, max value], data necessary for color_scheme 3, default None.
        elev : float : elevation in degrees of camera, default 30.
        no_axis : boolean : suppress drawing of axes, default False.
        radius_scale : float : change thickness of cylindrical fronts, default 1. normal size.
        scale_axis : boolean or list of 3 floats : list as [1.0,1.0,1.0] decrease one or more values to change relative scaling of axes, value for largest axis should be close to 1.0; default False.
        soma_black : boolean : all somata are black for increased contrast, default True.
        sphere_scale : float : change size of spherical fronts, default 1. normal size.
    """

    def __init__(self,num_procs,db_name,sim_volume,neuron_types,seed=None,\
                 verbose=2,num_admins=1,debug=False,max_neurons=1000,\
                 max_fronts=20000,max_active=2000,max_arcs=40,\
                 max_substrate=1000,max_synapse=1000,grid_step=20.,\
                 grid_extra=10,dict_size=20,plot=False,time_lapse=False,\
                 azim=-60.0,elev=30.0,box=False,no_axis=False,axis_ticks=True,\
                 scale_axis=False,soma_black=True,color_scheme=0,\
                 radius_scale=1.0,sphere_scale=1.0,color_data=None):
        self.start = time.time()
        if num_procs < 0:
            raise ValueError("num_procs","larger or equal to zero")
        if len(neuron_types) == 0:
            raise ValueError("neuron_types","at least one neuron_types declaration required")
        if max_neurons <= 1:
            raise ValueError("max_neurons","larger than one")
        if num_procs > 0:
            if max_fronts <= 100 * num_procs:
                raise ValueError("max_fronts","larger than 100 * num_procs")
            if max_active <= 10 * num_procs:
                raise ValueError("max_active","larger than 10 * num_procs")
            if max_substrate <= 10 * num_procs:
                raise ValueError("max_substrate","larger than 10 * num_procs")
            if max_synapse <= 10 * num_procs:
                raise ValueError("max_synapse","larger than 10 * num_procs")
        else:
            if max_fronts < 100:
                raise ValueError("max_fronts","larger than 99")
            if max_active < 10:
                raise ValueError("max_active","larger than 9")
            if max_substrate < 10:
                raise ValueError("max_substrate","larger than 9")
            if max_synapse < 10:
                raise ValueError("max_synapse","larger than 9")
        if max_arcs <= 10:
            raise ValueError("max_arcs","larger than 10")
        if (grid_step < 1.) or (grid_step > 100.):
            raise ValueError("grid_step","in range 1-100 µm")        
        if grid_extra < 10:
            raise ValueError("grid_extra","larger than nine")
        if dict_size < 10:
            raise ValueError("dict_size","larger than nine")
        self.master = True
        if plot:
            self._num_admins = 2 # always use to admins for notebook plotting
        else:
            if num_procs > 1:
                if (num_admins < 1) or (num_admins > 3):
                    raise ValueError("num_admins","in range 1-3")
                self._num_admins = num_admins # how many admin instances to use
            else:
                self._num_admins = 1
        if num_procs == 0:
            if verbose > 0:
                print (Fore.BLUE + nds_version() + " starting interactive mode" + Fore.RESET)
            _automatic = False
            num_procs = 1 # rest of code requires value larger than zero
        else:
            if verbose > 0:
                print (Fore.BLUE + nds_version() + " starting on",num_procs + min(2,self._num_admins),"cores" + Fore.RESET)
            _automatic = True
        self._num_procs = num_procs
        num_procsP1 = num_procs + 1
        # Process 1 is admin, Process 2 - num_procsP2 are other cores
        self._num_procsP2 = num_procs + 2
        self.my_id = 1
        self.verbose = verbose
        # extract grid parameters from sim_volume data
        self._volume = Volume(sim_volume,grid_step)
        # number of declared neuron types
        self._num_types = len(neuron_types)
        self._num_types_1 = self._num_types + 1
        self._neuron_types = []
        for t in neuron_types:
            self._neuron_types.append(_strip_neuron_type(t))
        # dictionary with key neuron names containing database neuron_ids
        self._names_in_DB = {} # dictionary by key neuron_name of neuron_id in DB
        self._max_neurons = max_neurons
        # sizes of shared arrays and striping ranges
        if not isinstance(max_fronts,list): # turn into list of proper length
            self._max_fronts = []
            for n in range(self._num_types):
                self._max_fronts.append(max_fronts + 1)
            sum_fronts = (max_fronts + 1)* self._num_types
        elif len(max_fronts) != self._num_types:
            raise ValueError("neuron_types or max_fronts","length mismatch between neuron_types and max_fronts lists")
        else:
            self._max_fronts = max_fronts # sizes of fronts arrays
            sum_fronts = 0 # sum all max values
            for n in range(self._num_types):
                max_fronts[n] += 1
                sum_fronts += max_fronts[n]
        self._front_range = []
        for n in range(self._num_types):
             # subdivision of specific fronts array for admin + each process
            self._front_range.append(self._max_fronts[n] // num_procsP1)
        self._max_active = max_active # indirectly determines size of new_AIDs array
        self._act_range = max_active // num_procs  # subdivision of active array for each process
        self._max_new = int(1 + max_active * 1.5)  # size of new_AIDs array
        self._new_range = int(self._act_range * 1.5)  # subdivision of new_AIDs array for each process
        # size of children array: enough for 2 children for each front, bit extra to compensate for ranges
        self._max_children = int(1.2 * sum_fronts)
        child_range = self._max_children // num_procsP1
        # arcs
        self._max_arcs = max_arcs # for each process
        self._tot_arcs = max_arcs * num_procs # full array size
        self._max_arc_points = self._max_arcs * ARCLENGTH
        # size of substrates array
        self._max_sub_names = max_substrate // 10 # first 10% for substrate with unique names
        self._max_sub_admin = self._max_sub_names + max_substrate // 30 # reserve 30% for admin
        if (max_substrate - self._max_sub_admin) / num_procs < 5: # reserve minimum for each processor
            self._max_substrate = max_substrate + 5 * num_procs
        else:
            self._max_substrate = max_substrate
        sub_range = (self._max_substrate - self._max_sub_admin) // num_procs
        # size of synapses array
        self.max_synapse = max_synapse
        syn_range = self.max_synapse // num_procsP1
        # size of grid_extra, starts at 1
        self._max_extra = int(grid_extra * sum_fronts) + 1
        # admin also gets a block in self.extra
        self._extra_range = self._max_extra // num_procsP1
        self._extra_size = grid_extra
        self._extra_size_1 = grid_extra - 1
        # size of self._gids, starts at 1
        self._max_gids = int(grid_extra * sum_fronts / 2) + 1
        # admin also gets a block in self._gids
        self._gids_range = self._max_gids // num_procsP1

        self.cycle = 0 # start
        self._first_cycle = True # after import_simulation may be True for cycle > 0
        self._simulation_allowed = True # default, may change after import_simulation

        ### make shared data structures
        # Instructions to processes
        self._clock = RawValue('i',0) # main control for cycle progression
        self._clock.value = 0 # waiting mode
        # Processor status: written by each processor, read by others
        self._proc_status = RawArray(c_int, self._num_procsP2)
        # neurons
        self._neurons = RawArray(Neuron, self._max_neurons)
        self._n_next = 1 # 0 not used
        # substrate
        self._substrate = RawArray(Substrate, self._max_substrate)
        # Fronts
        # substrate + all the fronts, striped by processor
        self._fronts = [self._substrate] # one RawArray for each type
        # locks by front index, striped by processor:
        # 0: no lock
        # > 0: my_id locked
        self._front_lock = [RawArray(c_short, self._max_substrate)]
        # for each process next empty space in fronts and its maximum value
        #   can be updated both by admin (add_neurons) or
        #   process (add_front)
        self._f_next_indices = [0] # first index not used
        self._f_max_indices = [0] # first index not used
        for n in range(self._num_types):
            self._fronts.append(RawArray(neuron_types[n], self._max_fronts[n]))
            self._front_lock.append(RawArray(c_short, self._max_fronts[n]))
            indices = RawArray(c_int, self._num_procsP2)
            maxind = RawArray(c_int, self._num_procsP2)
            #    initialize, first index used for admin
            for i in range(1,self._num_procsP2):
                indices[i] = (i-1) * self._front_range[n] + 1 # zero not used
                maxind[i] = indices[i] + self._front_range[n]
            self._f_next_indices.append(indices)
            self._f_max_indices.append(maxind)
        # indices of next active front for each core plus reserve
        self._actives = RawArray(ID, self._num_procsP2 + num_procs)
        # indices and status of newly made fronts, striped by processor
        self._new_AIDs = RawArray(ActiveFrontID, self._max_new)
        # array of processor specific starting indices
        self._n_starts = [0,0] # no pid zero, not used by admin
        for i in range(num_procs): # pids 2 - numprocP1
            self._n_starts.append(i * self._new_range)
        # children if more than 1 child, striped by processor
        self._children = RawArray(Child, self._max_children)
        # Arcs: self._arcs and self._arc_points will not be filled continuously due 
        #       to unpredictable order of arc deletions, process will keep track
        #       of empty slots in self._arcs, corresponding slot in self._arc_points.
        #   Arcs are completely managed by computing cores, Admin is not aware 
        #   of changes.
        self._arcs = RawArray(Arc, self._tot_arcs)
        self._arc_points = RawArray(Point, self._max_arc_points * num_procs)
        # Substrate: RawArray defined above
        # first 10% is reserved for substrate with unique names
        self._name_sub = 1
        #   rest is used for linked list to additional substrate with same name
        self._next_sub = self._max_sub_names # index into self._substrate
        # Synapses:
        self._synapses = RawArray(Synapse, self.max_synapse)
        # list of grid points used to list nearby fronts, continuous
        # > 0: fronts index
        # 0 : empty
        # < 0: link to grid_extra
        self._grid = RawArray(c_int, self._volume.grid_max)
        # writing lock status of each grid entry: 0 not locked, > 0 processor id of lock
        self._grid_wlock = RawArray(c_short, self._volume.grid_max)
        # each processor can request a writing grid lock or not (-1)
        self._gwlock_request = RawArray(c_int, self._num_procsP2)
        for i in range(self._num_procsP2):
            self._gwlock_request[i] = 0 # initialize to no request
        # reading lock status of each grid entry: 0 not locked, > 0 processor id of lock
        self._grid_rlock = RawArray(c_short, self._volume.grid_max)
        # each processor can request a reading grid lock or not (-1)
        self._grlock_request = RawArray(c_int, self._num_procsP2)
        for i in range(self._num_procsP2):
            self._grlock_request[i] = 0 # initialize to no request
        # in case more than one front is close to gridpoint: blocks of indices, continuous
        self._g_next = 1 # cannot link to zero from grid
        self._grid_extra = RawArray(ID, self._max_extra)
        self._gids = RawArray(c_int, self._max_gids)
        self._debug = debug

        # combine shared arrays into list
        master_constell = [self._fronts,self._actives,self._new_AIDs,\
                           self._f_next_indices,self._children,self._clock,\
                           self._front_lock,self._grid,self._grid_wlock,\
                           self._num_procsP2,self._grid_extra,\
                           self._extra_range,self._f_max_indices,\
                           self._act_range,child_range,self._extra_size,\
                           self._max_extra,self._neuron_types,self._gids_range,\
                           self._gids,self._substrate,self._max_sub_names,\
                           self._max_sub_admin,sub_range,self._debug,\
                           self._gwlock_request,self._proc_status,\
                           self._grlock_request,self._max_active,grid_step,\
                           self._synapses,syn_range,self._grid_rlock,\
                           self._num_types,self._volume,self._max_arcs,\
                           self._max_arc_points,self._arcs,self._arc_points,\
                           self._tot_arcs,self._n_starts,self._neurons]

        self.constellation = Constellation(self,master_constell)
        self.constellation._automatic = _automatic # store interactive setting
        
        # Start processes as soon as possible
        if _automatic: # initialize parallel processes
            ### start parallel processes
            set_start_method("fork",force=True) # required since Python 3.9, force required for repeated runs
            self.procs = []
            for pid in range(2,self._num_procsP2):
                p = Process(target=_proc_init, \
                            args=(pid,verbose,master_constell,seed))
                self.procs.append(p)
            if self._num_admins > 1: # launch an Admin_slave
                self.admin2 = Process(target=_admin_init, \
                                args=(-(self._num_admins - 1),\
                                        num_procs,verbose,master_constell))
                if verbose >= 6:
                    print ("Master starting slave",self._num_admins - 1)
                self.procs.append(self.admin2)
            else: # regular case: only a single Admin_agent
                self.admin2 = None

            ### now start parallel loop
            for p in self.procs:
                p.start()### make private data structures
                
        ### Local data structures
        # load balancing
        self._growing_fronts = [] # all fronts with is_growing() True
        self._migrating_fronts = [] # all fronts with is_migrating() True
        self._active_fronts = [] # all other fronts with is_active() True
        self._future_active = {} # dict by cycle fronts that need to be activated at that cycle
        # dictionary by key root neuron_name of number of somata made
        self._num_n_name = {}
        # dictionary of all unique substrate names: indices to first and last entry in self._substrate
        self._sub_names = {}
        # data structures for migration and substrate databases
        self._sub_data = False # no substrate database created
        self._mig_tables = 0 # no migration database created, multiple databases possible
        self._mig_fronts = [] # list of all (past) migrating fronts
        self._mig_data = 0 # counts # of migrating fronts in table
        self._mig_inserts = [] # indexed by table number
        self._syn_data = False # no synapses database created
        
        # private variables
        self.total_fronts = 0 # total fronts made, entire simulation
        
        # whether flags, arcs and mig_fronts get stored
        self.importable_db = False 
        # attribute storage: changed by attrib_to_db method, 0 index ignored
        #   boolean for each neuron_types
        self._num_store_db = 0 # number of new tables in database
        # 0: no storage, 1: continuous storage, 2: last_only storage
        self._store_types = [0] * self._num_types_1
        # database info
        self._db_new_tables = [] # names of attrib_to_db tables
        self._db_new_objects = [] # attribute object: 1 Front, 2 Neuron, 3 Synapse, 4 ID
        # list of lists of attribute strings for continuous storage: a list/neuron_type
        self._storec_attrs = []
        # list of lists of attribute strings for final storage: a list/neuron_type
        self._storef_attrs = []
        self._storec_db_tabs = [] # index into self._db_new_tables for each attr
        self._storef_db_tabs = [] # index into self._db_new_tables for each attr
        for n in range(self._num_types_1):
            self._storef_attrs.append([])
            self._storef_db_tabs.append([])
            self._storec_attrs.append([])
            self._storec_db_tabs.append([])
            self._db_new_objects.append([])
        # list of active fronts that store attributes, updated every cycle
        self._store_fronts = []
        
        # seed randomizer
        if not seed: # generate random seed
            np.random.seed()
            seed = np.random.randint(1000000)
        np.random.seed(seed)
        if self.verbose >= 4:
            print ("Admin seed",seed)
        self.seed = seed
        
        # debugging
        if debug:
            self._lw_gids = set() # gids write locked during this cycle
            self._lr_gids = set() # gids read locked during this cycle

        # initialize notebook plotting
        if plot: # this admin will only plot, do database and interact with user
            if time_lapse:
                self._plot = 2  # negative flags not initialized
                self._no_axis = True
            else:
                self._plot = 1
                self._no_axis = no_axis
            self._azim = azim
            self._elev = elev
            self._box = box
            if color_scheme < 2:
                self._soma_black = soma_black
            else:
                self._soma_black = False
            self._color_scheme = color_scheme
            self._scale_axis = scale_axis
            self._axis_ticks = axis_ticks
            self._radius_scale = 2. * radius_scale # convert to diameter
            self._sphere_scale = sphere_scale
        else: # no plotting: single admin performs all admin functions
            self._plot = 0
        if self._plot:
            self._initialize_plot(color_data)

        if _automatic: # initialize the database
            # dictionary with key neuron names containing database neuron_ids
            self._names_in_DB = {}
            # database initialization
            self._conn = self._setup_DB(db_name)
        elif self._plot:
            if self._box: # initialize storage of plot items
                self._plot_items = []

        if verbose >= 4:
            runtime = time.time() - self.start
            print ("Admin total startup time: {:.3f}s".format(runtime))
            self.sim_memory()

    # Initialize interactive plot
    def _initialize_plot(self,color_data):
        if self.verbose >= 4:
            print ("Admin initializing interactive plot")
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        if (self._color_scheme < 0) or (self._color_scheme > 3):
            raise ValueError("color_scheme","should be in range 0-3")
        if self._color_scheme == 3:
            if not color_data:
                raise ValueError("color_scheme","color_data must be defined for color_scheme=3")
            elif not isinstance(color_data,list):
                raise ValueError("color_data","color_data should be a list")
            elif len(color_data) != 3:
                raise ValueError("color_data","color_data should be a list with 3 entries")
            else:
                self._color_attrib = color_data[0]
                self._color_min = color_data[1]
                self._color_scale = 1. / (color_data[2] - color_data[1])
            import matplotlib.cm as cm
            self._colors = cm.get_cmap('rainbow')
            self._color_fronts = [] # list of [front,Last_value]
        else:
            self._colors = ['r','b','g','tab:orange','c','m','y','tab:red','tab:blue','tab:green',\
                            'tab:brown','tab:purple','tab:pink','tab:gray','tab:olive']
            self._c_mapping = {}
            self._color_index = 0
        if self._box:
            xmin,ymin,zmin = self._box[0]
            xmax,ymax,zmax = self._box[1]
        else:
            xmin,ymin,zmin = self._volume.sim_volume[0]
            xmax,ymax,zmax = self._volume.sim_volume[1]
        if self._plot == 2: # force smaller figs
            self._fig = plt.figure(figsize=(4,4)) # in inches, sigh...
        else: #default size is 8x6 inches
            self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111, projection='3d')
        if self._no_axis:
            self._ax.set_axis_off()
        else:
            self._ax.set_xlabel("X")
            self._ax.set_ylabel("Y")
            self._ax.set_zlabel("Z")
            if not self._axis_ticks:
                self._ax.axes.xaxis.set_ticks([])
                self._ax.axes.yaxis.set_ticks([])
                self._ax.axes.zaxis.set_ticks([])
            if self._scale_axis:
                if len(self._scale_axis) != 3:
                    raise ValueError("scale_axis","scale_axis should be a list with 3 entries or False")
                else:
                    self._scale_axis.append(1.0)
                    self._ax.get_proj = lambda: \
                                np.dot(Axes3D.get_proj(self._ax),np.diag(self._scale_axis))
        self._ax.azim = self._azim
        self._ax.elev = self._elev
        self._ax.set_xlim([xmin,xmax])
        self._ax.set_ylim([ymin,ymax])
        self._ax.set_zlim([zmin,zmax])
        self._lines = {} # store all lines by corresponding front.id so we can retract
        # prestore arrays to draw somata
        res = 20 # number of point precomputed
        u = np.linspace(0, 2 * np.pi, num=res)
        v = np.linspace(0, np.pi, num=res)
        self._x_outer = np.outer(np.cos(u),np.sin(v))
        self._y_outer = np.outer(np.sin(u),np.sin(v))
        self._z_outer = np.outer(np.ones([res]),np.cos(v))
        self._fig.canvas.draw() # do the actual drawing

    # generate soma distributions
    # returns list of points (success) or []
    def _distribute_neurons(self,number,loc,radius,grid):
        # get widths of bounding box
        sizes = []
        fixed_point = True # loc defines a single location
        # compute minimum distance between neurons and from border
        if grid and (len(grid) == 4): # include jitter
            jitter = grid[3]
            diameterP = (2.0 * radius) + jitter # minimal distance between somata centers
            radiusP = radius + jitter # minimal distance from border
        else:
            jitter = 0.0
            diameterP = 2.0 * radius + 0.5  # minimal distance between somata centers
            radiusP = radius # minimal distance from border
        warning = False
        sim_volume = self._volume.sim_volume
        for i in range(3): # 3D
            if len(loc) == 2: # two positions specified
                if loc[0][i] < sim_volume[0][i]:
                    raise VolumeError(loc[0][i])
                if loc[0][i] < sim_volume[0][i] + radiusP:
                    loc[0][i] = sim_volume[0][i] + radiusP
                    warning = True
                if loc[1][i] > sim_volume[1][i]:
                    raise VolumeError(loc[1][i])
                if loc[1][i] > sim_volume[1][i] - radiusP:
                    loc[1][i] = sim_volume[1][i] - radiusP
                    warning = True
                size = loc[1][i]-loc[0][i]
                if size < 0.0:
                    raise ValueError("location along each axis",">= 0")
                elif size > 0.0:
                    fixed_point = False
                sizes.append(size)
            else: # only one position specified
                if loc[i] < sim_volume[0][i]:
                    raise VolumeError(loc[i])
                if loc[i] < sim_volume[0][i] + radiusP:
                    loc[i] = sim_volume[0][i] + radiusP
                    warning = True
                if loc[i] > sim_volume[1][i]:
                    raise VolumeError(loc[i])
                if loc[i] > sim_volume[1][i] - radiusP:
                    loc[i] = sim_volume[1][i] - radiusP
                    warning = True
                sizes.append(0.)
        if (self.verbose >= 2) and warning:
            print (Fore.MAGENTA + "Admin add_neurons: location too close to border to fit soma, value changed to:",loc,Fore.RESET)
        if fixed_point:
            if number > 1:
                raise ValueError("number of neurons","1 for a fixed location")
            if len(loc) == 2:
                loc = loc[0]
        # generate positions for all neurons
        gpoints = []
        if grid:
            # check grid parameters
            ngrid = len(grid)
            if (ngrid < 3) or (ngrid > 4):
                raise TypeError("grid should be specified as (nx,ny,nz) or as (nx,ny,nz,jitter)")
            # create the grid:
            #   compute distance between grid points
            gorigin = []
            goffsets = []
            for i in range(3): # 3D
                if grid[i] < 1:
                    raise ValueError("each grid size",">= 1")
                if int(grid[i]) == 1:
                    gorigin.append(loc[0][i] + sizes[i]/2.0) # use mid point
                    goffsets.append(0.0)
                else:
                    gorigin.append(loc[0][i])
                    offset = sizes[i] / (int(grid[i]) - 1)
                    if offset < diameterP:
                        raise ValueError("grid subdivisions (for given radius)",">= " + "{:4.2f}".format(diameterP))
                    goffsets.append(offset)
            #  create list of all possible grid points
            points = []
            for i in range(int(grid[0])):
                x = gorigin[0] + i * goffsets[0]
                for j in range(int(grid[1])):
                    y = gorigin[1] + j * goffsets[1]
                    for k in range(int(grid[2])):
                        z = gorigin[2] + k * goffsets[2]
                        points.append(Point(x,y,z))
            if number > len(points):
                raise ValueError("number of grid points",">= number of neurons")
            #  select number of points we need
            indices = range(len(points))
            indices = np.random.choice(indices,number,replace=False)
            indices.sort()
            for i in indices:
                gpoints.append(points[i])
            #  apply jitter if required
            if ngrid == 4: # apply jitter
                for i in range(len(gpoints)):
                    xyz0 = copy.copy(gpoints[i]) # store original
                    n = 0
                    while True:
                        if n > 10 * MAXTRIALS:
                            if self.verbose >= 2:
                                print (Fore.MAGENTA + "Warning: jitter too large in add_neurons, continuing without jitter for a neuron" + Fore.RESET)
                            break # while loop
                        xyz = xyz0 + np.random.normal(scale=jitter,size=3)
                        n += 1
                        if xyz.out_volume(self.constellation) == 0.:
                            gpoints[i] = xyz
                            break # while loop

        elif fixed_point:
            gpoints.append(_point_list(loc))
        else: # completely random positions
            # check whether we have enough space for all requested somata
            volume = sizes[0] * sizes[1] * sizes[2]
            allsoma = number * 4.19 * radius**3 # volume of number of spheres
            if (volume > 0.0) and (allsoma / volume) > 0.80:
                raise ValueError("location","provide enough space to allocate all somata safely")
            for i in range(number):
                for j in range(MAXTRIALS):  # repeat till empty space found, do not risk an endless loop
                    offset = [np.random.random() * sizes[0],\
                                np.random.random() * sizes[1],np.random.random() * sizes[2]]
                    xyz = _point_list(loc[0]) + _point_list(offset)
                    safe = True
                    for k in range(len(gpoints)): # test distance with other somata
                        distance = xyz - gpoints[k]
                        if distance.length() < diameterP: # bad distance
                            safe = False
                            break
                    if safe:
                        gpoints.append(xyz)
                        break
                if not safe:
                    raise ValueError("location","not enough space to allocate all somata")
        return gpoints
                    
    # add somata or axons for a neuron type
    # returns success as a boolean
    def add_neurons(self,front_class,neuron_name,number,location,radius,\
                    axon=False,migrating=False,grid=False,origins=[],\
                    print_color=False):
        """ Add new neuron somata or axons to the simulation.
        
        The only way to create new neurons. This method can be called repeatedly to generate different neuron types and/or neurons of the same type at different simulation cycles. By default random placement is performed, either *grid* or *origins* can be used to specify more specific placement of multiple neurons.
        
        By default spherical somata are created, but instead cylindrical axon fronts can be made (*axon* optional parameter. Either will become the root of a neuron tree. By default the first front has is_growing()=True, unless the *migrating* optional parameter is used.
        
        Parameters
        ----------
        front_class : string : name of the ``Front`` subclass to use.
        neuron_name : string of maximum 34 characters : name of the neuron. Names are automatically numbered as *neuron_name_0*, *neuron_name_1*, etc.
        number : integer > 0 : number of neurons to create.
        location : list or list of 2 lists: location of neuron somata specified as bounding coordinates. To add a single neuron [x, y, z] can be specified, to add multiple neurons this should be a box as [[left, front, bottom], [right, back, top]]. Each neuron soma is placed randomly within this box unless *grid* or *origins* is specified. Soma centers are minimally separated by the diameter.
        radius : float > 0.0 : soma radius in µm.
        Optional:
        axon : boolean or list : initialize axon instead of soma; list [dx,dy,dz] specifies an offset to apply to *orig* to generate *end* of cylindrical axon, default False: spherical somata are made.
        grid : boolean or list : place somata on a grid specified as [nx, ny, nz] or as [nx, ny, nz, jitter] where integers nx, ny, nz are the number of positions on the grid along each dimension (there can be more grid positions than number of neurons, but not less) and optional float jitter is the standard deviation in µm of random (normal distribution) jitter applied to each grid position, default False.
        migrating : boolean : set soma.is_migrating()=True and soma.is_growing=False, default False
        origins : [``Point``,] : list of ``Point`` to be used to place the somata.
        print_color : boolean : print neuron colors for interactive plot, default False.

        Returns
        -------
        nds_list of fronts as stored : [Front,]
        """
        front_class = _strip_neuron_type(front_class)
        if front_class not in self._neuron_types:
            raise TypeError("undeclared front_class: " + str(front_class))
        if len(neuron_name) > NNAMELENGTH - 6: # space for '_' + 4 digits + '_'
            raise ValueError("length of neuron_name","maximum: " + str(NNAMELENGTH - 6))
        if number <= 0:
            raise ValueError("number of neurons","larger than zero")
        if radius <= 0.:
            raise ValueError("radius","larger than zero")
        if self.verbose >= 4:
            print ("Admin: add",number,"neurons",neuron_name)
        nid = self._neuron_types.index(front_class) + 1
        num_news = 0 # number of somata created
        self.constellation.cycle = self.cycle # update
        if origins: # list provided
            if len(origins) < number:
                raise ValueError("length of origins list",">= number of neurons")
            for p in origins:
                if not isinstance(p,Point):
                    raise TypeError("origins should be list of Point")
                result = p.out_volume(self.constellation)
                if result != 0.:
                    raise VolumeError(result)
        else: # create new list
            origins = self._distribute_neurons(number,location,radius,grid)
        not_first = False
        if origins: # list of points
            if neuron_name not in self._num_n_name:
                self._num_n_name[neuron_name] = 0 # initialize counter
            # allocate somata in admin block of self._fronts
            new_f = nds_list() # make a list for _write_new_DB
            migrated_f = [] # make a list for _new_mig_DB
            for i in range(number):
                self._num_n_name[neuron_name] += 1
                # give each cell a specific name
                n_name = neuron_name + "_" + str(self._num_n_name[neuron_name])\
                                     + "_"
                p = origins[i]
                count = 0
                coordinate = p
                # make the soma or axon root
                while count < 100: # no endless loop
                    try: # test for collision
                        gids = self.constellation._get_gids(coordinate,None,radius)
                        self.constellation._test_collision(None,coordinate,radius,\
                                                            None,gids,lock=False)
                        break # out of while
                    except CollisionError as error:
                        coordinate = p.wiggle(self.constellation,max(0.51,count / 10.))
                        count += 1
                    if (count == 100) and (self.verbose >= 2):
                        print (Fore.MAGENTA + "Warning: admin not instantiating new soma because of collision at",origins[i],Fore.RESET)
                    continue
                if axon:
                    b_name = 'axon'
                else:
                    b_name = 'soma'
                # enter into self._fronts array
                soma = self.constellation._enter_front(None,False,coordinate,\
                                               radius,b_name,1,nid=nid)
                if axon: # change to axon, can only be done after initialization
                    soma.end = origins[i] + axon
                    soma._set_cylinder()
                    soma.path_length = soma.length()
                    soma.swc_type = 2
                id = ID(nid,soma._fid)
                if self._store_types[nid] > 0: # activate storing
                    soma._set_storing()
                # make the neuron
                if self._n_next >= self._max_neurons:
                    raise OverflowError("_neurons","max_neurons")
                neuron = Neuron(n_name[:NNAMELENGTH].encode('utf-8'),id,0.,0.,\
                                0,0,0,self._n_next)
                soma._sid = self._n_next
                self._neurons[self._n_next] = neuron
                self._n_next += 1
                new_f.append(soma) # list for database
                if migrating:
                    soma.set_migrating()
                    self._migrating_fronts.append(soma)
                    migrated_f.append(soma)
                else:
                    soma.set_growing()
                    self._growing_fronts.append(soma)
                # enter in grid
                gids = soma._get_gids(self.constellation)
                for gid in gids:
                    self.constellation._grid_set(gid,id)
                soma._store_gids(self.constellation,gids)
                num_news += 1 # counts for this call only
                if self.verbose >= 6:
                    print ("Admin made new front",str(soma))
                if self._plot: # initialize color mapping
                    if self._color_scheme == 0: #  colored by neuron_name
                        self._c_mapping[n_name] = self._colors[self._color_index%len(self._colors)]
                        if print_color or (self.verbose >= 6):
                            print ("neuron",n_name,soma._fid,"color",\
                            self._c_mapping[n_name][4:])
                        self._color_index = self._color_index + 1
                    elif self._color_scheme == 1: # color only different types of neurons different
                        self._c_mapping[n_name] = \
                                    self._colors[(nid - 1)%len(self._colors)]
                not_first = True
            self.total_fronts += num_news
        if self.constellation._automatic:
            if new_f:
                self._write_new_DB(new_f) # write to database
            if migrated_f:
                self._new_mig_DB(migrated_f) # make migration columns
        return new_f

    # add substrate
    # the self.substrates RawArray is divided into multiple zones:
    # - first 10%: first substrate with new name -> same index as in self._sub_names
    # - next 30%: reserved for entry by Admin_agent.add_substrate
    # - rest: striped per processor, for entry by Constellation.add_substrate
    # can be called in multiple ways:
    # - as Admin_agent method with only master:
    #   enters into arrays and updates database
    # - as Admin_agent from import_simulation (only master): database==False
    #   enters into arrays
    # - as Admin_agent method with master & slave:
    #   master enters into arrays and communicates to slave to update database
    # - as Admin_agent completing the Constellation method:
    #   checks whether this is 'new' (different name) substrate, in that case it
    #   copies it to the bottom part of the array, otherwise links it to the
    #   existing substrate. Database update depends on whether a slave exists.
    def add_substrate(self,substrate,index=-1,database=True):
        """ Add substrate to the simulation.
        
        Parameters
        ----------
        substrate : a ``Substrate`` or list of ``Substrate``.
        Optional:
        internal_use_only

        Returns
        -------
        nds_list of substrate as stored : [Substrate,]
        """
        if isinstance(substrate,Substrate):
            subs = [substrate]
        elif not isinstance(substrate,list):
            raise TypeError("substrate must be Substrate or [Substrate,]")
        else:
            subs = substrate
        not_first = False
        new_s = nds_list()
        for sub in subs:
            name = sub.get_name()
            if name not in self._sub_names: # create entry for new name
                # initialize with no link in last_sub
                if self._name_sub >= self._max_sub_names:
                    raise OverflowError("substrate","max_substrate")
                self._sub_names[name] = [self._name_sub,0]
                next_sub = self._name_sub
                self._name_sub += 1
                last_sub = 0
                index = -1 # ignore stored value if called by simulation_loop
            else: # extend existing name entry
                next_sub = self._next_sub
                first_sub = self._sub_names[name][0] # first substrate entry with this name
                last_sub = self._sub_names[name][1] # last substrate entry with this name
            if sub.orig.out_volume(self.constellation) != 0.:
                if self.verbose >= 2:
                    print (Fore.MAGENTA + "Warning: substrate",name,sub.orig,"outside simulation volume is ignored", Fore.RESET)
                continue
            if sub.n_mol <= 0:
                if self.verbose >= 2:
                    print (Fore.MAGENTA + "Warning: substrate",name,sub.orig,"outside simulation volume is ignored", Fore.RESET)
                print (Fore.MAGENTA + "Warning: substrate",name,"invalid n_mol",sub.n_mol,"is ignored", Fore.RESET)
                continue
            if sub.rate < 0.:
                if self.verbose >= 2:
                    print (Fore.MAGENTA + "Warning: substrate",name,"invalid rate",sub.rate,"is ignored", Fore.RESET)
                continue
            if sub.diff_c < 0.:
                if self.verbose >= 2:
                    print (Fore.MAGENTA + "Warning: substrate",name,"invalid diff_c",sub.diff_c,"is ignored", Fore.RESET)
                continue
            if index < 0: # enter into substrate array, called from main
                if next_sub >= self._max_sub_admin:
                    raise OverflowError("substrate","max_substrate")
                if self.verbose >= 6:
                    print ("Admin: add_substrate",name,next_sub)
                sub._fid = next_sub
                self._substrate[next_sub] = sub
                new_s.append(self._substrate[next_sub])
                if next_sub >= self._max_sub_names: # we used a self._next_sub value -> increase
                    self._next_sub += 1
            else:
                next_sub = index # use existing index for updates
            # update linked list
            if last_sub > 0: # extend linked list
                self._substrate[last_sub]._next = next_sub # update index
            elif next_sub >= self._max_sub_names: # enter first index of linked list
                self._substrate[first_sub]._next = next_sub # update index
            self._sub_names[name][1] = next_sub # update last_sub index
            if self._plot:
                if self._color_scheme < 3:
                    if not (name in self._c_mapping):
                        self._c_mapping[name] = self._colors[self._color_index%len(self._colors)]
                        self._color_index = self._color_index + 1
                    col = self._c_mapping[name]
                else:
                    col = 'k'
                self._ax.scatter(sub.orig.x,sub.orig.y,sub.orig.z,color=col,marker='+')
            # enter into database
            if self.constellation._automatic:
                if not self._sub_data:
                    self._sub_DB()
                values = (None,sub.get_name(),sub.orig.x,sub.orig.y,sub.orig.z,\
                            sub.n_mol,sub.rate,sub.diff_c,self.cycle,-1)
                self._conn.cursor().execute("INSERT into substrate_data VALUES (?,?,?,?,?,?,?,?,?,?)",values)
            next_sub = self._next_sub # always true
            not_first = True
        return new_s

    # Import fronts etc. from previous simulation as starting condition
    def import_simulation(self,stored_db_name,copy_db=True):
        """ Import a previous neurodevsim simulation. This can be used either for interactive model debugging or to continue the simulation, in the latter case the *num_procs* should be identical. The database should be from a simulation that ran to finish and completed its ``admin.destruction`` method.
        
        This assumes that the same values for the *sim_volume* and *neuron_types* ``Admin_agent`` attributes have been defined as for the stored simulation. Any optional changes to ``Admin_agent`` array size attributes should also be applied.
        
        This method should be the first method called after initalization of ``Admin_agent``.  New, additional neuron types are allowed but they should be at the end of the previous *neuron_types* list.
        
        The method tries to recreate the original data structures as much as possible. Hidden data structures should contain identical information but the order may be different. Additional, user-defined ``Front`` attributes that were stored using ``attrib_to_db`` are restored.
        
        Parameters
        ----------
        stored_db_name : string : name of the sqlite database that stores previous simulation results. This should be a different name than the *db_name* used for ``Admin_agent`` initialization.
        Optional:
        copy_db : boolean : the content of *stored_db_name* is copied to *db_name*, default True.
        """
        start = time.time()
        if self.cycle > 0:
            raise TypeError("import_simulation can only be run at cycle 0")
        # Check validity of stored_db_name
        if not os.path.isfile(stored_db_name):
            raise TypeError("stored_db_name not found")
        try:
            oldconn = sqlite3.connect(stored_db_name)
        except Exception:
            raise TypeError(str(stored_db_name) + ": file cannot be opened")
        oldconn.row_factory = sqlite3.Row
        cursor = oldconn.cursor()
        #   check version
        try:
            cursor.execute("select * from neurodevsim") # Should be only one row
        except Exception:
            raise TypeError(str(stored_db_name) + ": not a NeuroDevSim database")
        nret = cursor.fetchone()
        oldversion = nret['version']
        if (oldversion < 93):
            raise TypeError(str(stored_db_name) + ": incompatible old  NeuroDevSim database")
        if not nret['importable']:
            raise TypeError(str(stored_db_name) + ": NeuroDevSim database is not suitable for import, set admin.importable_db before storing")
        # check num_cycles
        num_cycles = nret['num_cycles']
        if num_cycles == 0:
            raise TypeError(str(stored_db_name) + ": database from unfinished simulation, cannot be used for import_simulation")
        #    check volume
        old_volume = [[nret['xmin'],nret['ymin'],nret['zmin']],\
                      [nret['xmax'],nret['ymax'],nret['zmax']]]
        if old_volume != self._volume.sim_volume:
            raise TypeError(str(stored_db_name) + ": different simulation volume " + str(old_volume))
        #    check neuron types
        cursor.execute("select * from neuron_types")
        trets = cursor.fetchall()
        n = 0
        for row in trets:
            if _strip_neuron_type(str(row['neuron_type'])) != str(self._neuron_types[n]):
                raise TypeError(str(stored_db_name) + ": neuron_type mismatch " + str(row['neuron_type']) + " != " + str(self._neuron_types[n]))
            n += 1
        if self.verbose > 0:
            print (Fore.BLUE + "Admin: import",stored_db_name,Fore.RESET)
        #    check num_procs
        if self.constellation._automatic and \
                        (self._num_procs != nret['num_procs']):
            print (Fore.MAGENTA + "Warning: different number of processors than before (" + str(nret['num_procs']) + ") -> simulation_loop method disabled" + Fore.RESET)
            self._simulation_allowed = False
        # Copy database:
        if self.constellation._automatic and copy_db:
            version = nds_version(raw=True)
            if version > oldversion: # more recent version
                # check whether still compatible -> otherwise error
                pass
            self._conn.close() # close existing database file
            if self.verbose >= 6:
                print ("Admin copying",stored_db_name,"to",self.db_name)
            try:
                os.remove(self.db_name)
            except Exception:
                pass
            shutil.copyfile(stored_db_name,self.db_name)
            # connect to the copy
            try:
                self._conn = sqlite3.connect(self.db_name)
            except Exception:
                raise TypeError(str(db_file_name) + ": file cannot be opened")
            # copy the original row 1 as second one
            values = (None,nret['xmin'],nret['ymin'],nret['zmin'],nret['xmax'],\
                        nret['ymax'],nret['zmax'],nret['num_cycles'],\
                        nret['num_procs'],nret['version'],nret['run_time'],\
                        nret['importable'],nret['substrate'],\
                        nret['migration'],nret['synapses'],\
                        nret['attributes'],nret['arcs'])
            cursor.execute("INSERT into neurodevsim VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",values)
            # update info
            if version > oldversion: # more recent version
                self._conn.cursor().execute("UPDATE neurodevsim SET version = ? WHERE id = 1",(version,))
            if self._num_types > len(trets): # additional neuron types defined
                tvalues = []
                for n in range(len(trets),self._num_types):
                    tvalues.append((None,n + 1,str(self._neuron_types[n])))
                self._conn.cursor().executemany("INSERT into neuron_types VALUES (?,?,?)",tvalues)
        # Start reading data
        #   all neurons: names and nid and create Neuron
        cursor.execute("select * from neuron_data")
        narets = cursor.fetchall()
        neuron_nid = {} # dict by neuron_id of its nid
        for row in narets:
            if (oldversion < 93.2):
                nid = row['array_id']
            else:
                nid = row['type_id']
            neuron_id = row['neuron_id']
            name = row['name']
            # make the neuron, soma not created yet
            if self._n_next >= self._max_neurons:
                raise OverflowError("_neurons","max_neurons")
            neuron = Neuron(name.encode('utf-8'),empty_ID,row['firing_rate'],\
                            row['CV_ISI'],row['num_fronts'],\
                            row['num_retracted'],row['num_synapses'],\
                            self._n_next)
            self._neurons[self._n_next] = neuron
            if self.constellation._automatic:
                self._names_in_DB[name] = self._n_next
            neuron_nid[self._n_next] = nid
            self._n_next += 1
            neuron_name = neuron.get_base_name() # extract original name
            if neuron_name not in self._num_n_name:
                self._num_n_name[neuron_name] = 1 # initialize counter
            else:
                self._num_n_name[neuron_name] += 1
            if self._plot: # initialize color mapping
                if self._color_scheme == 0: #  colored by neuron_name
                    self._c_mapping[name] = \
                            self._colors[self._color_index%len(self._colors)]
                    self._color_index = self._color_index + 1
                elif self._color_scheme == 1: # color only different types of neurons different
                    self._c_mapping[name] = \
                                    self._colors[(nid - 1)%len(self._colors)]
        #   migrating somata: only need their final position
        if nret['migration'] > 0: # load migration_data
            self._mig_tables = nret['migration']
            self.mig_col = -1 # start at 2 but + 3 by _add_mig_DB
            mig_table = nret['migration']
            # new data structures that can be accessed in interactive mode
            mig_index = self.constellation.mig_index = {} # dict by front._key() containing index into mig_history
            mig_history = self.constellation.mig_history = {} # dict by col_key containing [[coordinate,][cycle,]]
            first = True
            for i in range(mig_table):
                try:
                    cursor.execute("select * from migration_data" + str(i + 1))
                except Exception:
                    raise TypeError(str(stored_db_name) + ": expected migration_data" + str(i + 1) + " table")
                mrets = cursor.fetchall()
                num_cols = len(mrets[0])
                self.mig_col += num_cols - 2 # for next neuron
                # create mig_inserts
                if self.constellation._automatic:
                    query = "INSERT into migration_data" + str(i + 1) + " VALUES (?"
                    for k in range(1,num_cols):
                        query += ",?"
                    self._mig_inserts.append(query)
                # turn mrets data into a list of xyz
                for row in mrets:
                    cycle = row['cycle']
                    for ncol in range(2,num_cols,3):
                        if row[ncol]:
                            index = i * 1800 + ncol
                            if index not in mig_history:
                                # create empty spot for original location
                                mig_history[index] = [[None],[0]]
                            mig_history[index][0].append\
                                (Point(row[ncol],row[ncol+1],row[ncol+2]))
                            mig_history[index][1].append(cycle)
        else:
            mrets = None
        #   all fronts:
        cursor.execute("select * from front_data")
        frets = cursor.fetchall()
        # process rows and enter into shared arrays but do not update children yet
        child_fronts = [] # list of all fronts that need to be registered with parent
        num_fronts = 0
        self.front_data_id = len(frets) + 1
        for row in frets:
            birth = row['birth']
            neuron_id = row['neuron_id']
            nid = neuron_nid[neuron_id]
            fid = row['front_id']
            id = ID(nid,fid)
            # update self._f_next_indices to value higher, is done repeatedly
            proc = 1 + ((fid - 1) // self._front_range[nid - 1]) # processor that made this front 
            if self._f_next_indices[nid][proc] <= fid:
                self._f_next_indices[nid][proc] = fid + 1
            front = self._fronts[nid][fid]
            front.name = row['branch'][:BNAMELENGTH].encode('utf-8')
            swc = row['swc_type']
            front.swc_type = swc
            if row['shape'] == 2:
                front._set_cylinder()
            orig = Point(row['orig_x'],row['orig_y'],row['orig_z'])
            end = Point(row['end_x'],row['end_y'],row['end_z'])
            front.radius = row['radius']
            parent_id = row['parent_id']
            front.order = row['b_order']
            front.path_length = row['path_len']
            front.birth = birth
            death = row['death']
            front._nid = nid
            front._fid = fid
            front._pid = parent_id
            front._flags = row['flags']
            front._dbid = row['id']
            if parent_id < 0: # a soma or an axon: update neuron info
                self._neurons[neuron_id].soma_ID = id
                front._sid = neuron_id                  
                if swc == 1: # soma: get migration status
                    mig = row['migration']
                    if mig > 0: # migrating or migrated
                        if mig in mig_history:
                            mig_index[front._key()] = mig
                            # update first position
                            mig_history[mig][0][0] = orig
                            mig_history[mig][1][0] = birth
                            # find last entry or entry for this cycle
                            cycle = 0
                            n = 0
                            while n < len(mig_history[mig][0]):
                                cycle = mig_history[mig][1][n]
                                n += 1
                            orig = end = mig_history[mig][0][n - 1]
                        else:
                            if self.verbose >= 2:
                                print (Fore.MAGENTA + "Warning: import_simulation empty soma migration info " + str(id) + Fore.RESET)        
            else:
                if self._neurons[neuron_id].soma_ID._nid == 0:
                    raise BugError("import_simulation","missing soma_ID " + str(self._neurons[neuron_id]))                    
                front._sid = self._neurons[neuron_id].soma_ID._fid
            front.orig = orig # may have changed for migrating soma
            front.end = end
            if death > 0: # retracted
                front.death = death
                self._neurons[neuron_id].num_retracted + 1
            else: # regular front
                if parent_id > 0:
                    child_fronts.append(front)
                self._neurons[neuron_id].num_fronts + 1
                num_fronts += 1
                # check migration, growth and active status
                if front.is_migrating(): # migrating
                    self._migrating_fronts.append(front)
                    if self.verbose >= 6:
                        print ("Admin: imported",front," as migrating")
                    if front._does_storing() and \
                                        front not in self._store_fronts:
                        self._store_fronts.append(front)
                elif front.is_growing(): # growing
                    self._growing_fronts.append(front)
                    if self.verbose >= 6:
                        print ("Admin: imported",front," as growing")
                    if front._does_storing() and \
                                        front not in self._store_fronts:
                        self._store_fronts.append(front)
                elif front.is_active():
                    self._active_fronts.append(front)
                    if self.verbose >= 6:
                        print ("Admin: imported",front," as active")
                    if front._does_storing() and \
                                        front not in self._store_fronts:
                        self._store_fronts.append(front)
                # enter into other data structures
                gids = front._get_gids(self.constellation)
                for gid in gids:
                    self.constellation._grid_set(gid,id)
                front._store_gids(self.constellation,gids) 
                if self._plot:
                    self._plot_front(front)
        if self._plot:
            self._fig.canvas.draw()
        # now register children with parents, for trailing axons these are born afterwards
        for child in child_fronts:
            parent = self._fronts[child._nid][child._pid]
            if parent: # if not all fronts are loaded trailing axon parent may be missing
                if parent.death > 0:
                    # can be a filipod: database is not updated for parent change after _replace_child
                    if child.swc_type == 12: # filipod
                        soma = child.get_soma(self.constellation)
                        if not soma.has_migrated():
                            raise BugError("import_simulation","filipod child of dead parent with non migrating soma " + str(child))                    
                        # test whether soma is close enough: soma should have migrated orig
                        if child.front_distance(soma) <= child.length():
                            child._pid = soma._fid # update parent to soma
                            parent = soma
                        else:
                            raise BugError("import_simulation","filipod child of dead parent too far from migrating soma " + str(child))                    
                    else:    
                        raise BugError("import_simulation","child of dead parent " + str(child))                    
                self.constellation.lock(parent)
                self.constellation._add_child(parent,child._fid)
                self.constellation.unlock(parent)
            elif self.verbose >= 2:
                print (Fore.MAGENTA + "Warning: missing parent in import_simulation",child._nid,child._pid,Fore.RESET)
        if self.verbose >= 6:
            print ("Admin: imported",self._n_next - 1,"neurons and", num_fronts,"fronts")
            print ("      ",len(self._growing_fronts),"growing",len(self._migrating_fronts),"migrating",len(self._active_fronts),"active")
        if nret['migration'] > 0: # also load mig_fronts_data table
            cursor.execute("select * from mig_fronts_data")
            mfrets = cursor.fetchall()
            for row in mfrets:
                if (oldversion < 93.2):
                    f = self._fronts[row['array_id']][row['front_id']]
                else:
                    f = self._fronts[row['type_id']][row['front_id']]
                self._mig_fronts.append(f)
                self._mig_data += 1
        #   all substrate:
        if nret['substrate']:
            cursor.execute("select * from substrate_data")
            srets = cursor.fetchall()
            for row in srets:
                if row['death'] > 0: # removed -> will not be instantiated
                    continue
                orig = Point(row['x'],row['y'],row['z'])
                sub = Substrate(row['name'],orig,row['birth'],row['n_mol'],\
                                row['rate'],row['diff_c'])
                self.add_substrate(sub,database=False)
        #   all synapses:
        if nret['synapses']:
            self._syn_data = True
            self._syn_data_id = 1
            cursor.execute("select * from synapse_data")
            yrets = cursor.fetchall()
            for row in yrets:
                if row['death'] > 0:  # removed -> will not be instantiated
                    continue
                if self.constellation._syn_next >= self.constellation._syn_max:
                    raise OverflowError("_synapses","max_synapse")
                # presynaptic
                neuron_id0 = row['pre_neuron_id']
                nid0 = neuron_nid[neuron_id0]
                fid0 = row['pre_front_id']
                pre_id = ID(nid0,fid0)
                if self._fronts[nid0][fid0]._nid == 0:
                    raise BugError("import_simulation","missing presynaptic front " + str(pre_id))                    
                # postsynaptic
                neuron_id1 = row['post_neuron_id']
                nid1 = neuron_nid[neuron_id1]
                fid1 = row['post_front_id']
                post_id = ID(nid1,fid1)
                if self._fronts[nid1][fid1]._nid == 0:
                    raise BugError("import_simulation","missing postsynaptic front " + str(post_id))                    
                yid = self.constellation._syn_next
                syn = Synapse(pre_id,post_id,row['weight'])
                self._synapses[yid] = syn
                self._fronts[nid0][fid0]._yid = -yid
                self._fronts[nid1][fid1]._yid = yid
                self._fronts[nid1][fid1].syn_input = \
                                            syn._syn_input(self.constellation)
                self._syn_data_id += 1
                self._neurons[neuron_id0].num_synapses + 1
                self._neurons[neuron_id1].num_synapses + 1
                self.constellation._syn_next += 1
        # get the arcs
        if nret['arcs']:
            cursor.execute("select * from arc_data")
            arets = cursor.fetchall()
            for row in arets:
                # get arc data
                aind = row['arc_index']
                arc = self._arcs[aind]
                arc.cycle = row['cycle']
                arc.complete = row['complete']
                arc.neuron_id = row['neuron_id']
                arc.caller_fid = row['caller_fid']
                arc.last_fid = row['last_fid']
                arc.sphere = ID(row['sphere_0'],row['sphere_1'])
                arc.arc_angle = row['angle']
                arc.num_points = row['num_points']
                arc.count = row['count']
                arc.next_point = row['next_point']
                # get arc points
                cursor.execute("select * from arc_points where arc_index=?",(aind,))
                prets = cursor.fetchall()
                if len(prets) != arc.num_points:
                    raise BugError("import_simulation","wrong arc number of points " + str(len(prets)) + " " + str(arc.num_points))
                n = 0
                for row in prets:
                    p = Point(row['p_x'],row['p_y'],row['p_z'])
                    self._arc_points[arc.index + n] = p
                    n += 1
            self._clock.value = -5 # have processes call _update_empty_arcs
        # user defined tables
        if nret['attributes']:
            if self.verbose >= 6:
                print ("Admin: import attributes database")
            cursor.execute("select * from attributes")
            trets = cursor.fetchall()
            for row in trets:
                tname = row['name']
                self._db_new_tables.append(tname) # store as existing table
                self._num_store_db += 1
                # load attribute values and enter them in storage
                attrib = tname[:-5]  # attribute name
                if self.verbose >= 6:
                    print ("Admin: importing",tname,"for",attrib)
                # read data for correct cycle from table
                cursor.execute("select * from " + str(tname) + " WHERE cycle = " + str(num_cycles))
                #cursor.execute("select * from " + str(tname) + " WHERE cycle = " + str(num_cycles))
                arets = cursor.fetchall()
                # now update front attributes
                if attrib in ['firing_rate','CV_ISI']: # Neuron attribute
                    obj_type = Neuron
                    for arow in arets:
                        neid = arow['neuron_id']
                        nid = neuron_nid[neid]
                        neuron = self._neurons[neid]
                        if attrib == 'firing_rate':
                            neuron.set_firing_rate\
                                        (self.constellation,arow[attrib])
                        else:
                            neuron.set_CV_ISI(self.constellation,arow[attrib])
                elif attrib == 'weight': # Synapse attribute
                    obj_type = Synapse
                    for arow in arets:
                        neid = arow['pre_neuron_id']
                        nid = neuron_nid[neid]
                        fid = arow['pre_front_id']
                        front = self.constellation.front_by_id(ID(nid,fid))
                        if front._yid != 0:
                            synapse = self._synapses[abs(front._yid)]
                            synapse.set_weight(self.constellation,arow[attrib]) 
                        else:
                            raise BugError("import_simulation","missing synapse for additional attribute '" + str(attrib) + "' in front " + str(front))
                elif len(arets[0]) == 7: # a Front Point
                    obj_type = Front
                    for arow in arets:
                        neid = arow['neuron_id']
                        nid = neuron_nid[neid]
                        fid = arow['front_id']
                        front = self.constellation.front_by_id(ID(nid,fid))
                        attr0 = arow[attrib + '_x']
                        attr1 = arow[attrib + '_y']
                        attr2 = arow[attrib + '_z']
                        try:
                            setattr(front,attrib,Point(attr0,attr1,attr2))
                        except:
                            raise BugError("import_simulation","missing additional attribute '" + str(attrib) + "' in front " + str(front))
                elif len(arets[0]) == 6: # a Front ID
                    obj_type = Front
                    for arow in arets:
                        neid = arow['neuron_id']
                        nid = neuron_nid[neid]
                        fid = arow['front_id']
                        front = self.constellation.front_by_id(ID(nid,fid))
                        attr0 = arow[attrib + '_0']
                        attr1 = arow[attrib + '_1']
                        try:
                            setattr(front,attrib,ID(attr0,attr1))
                        except:
                            raise BugError("import_simulation","missing additional attribute '" + str(attrib) + "' in front " + str(front))
                else: # regular Front attribute
                    obj_type = Front
                    for arow in arets:
                        neid = arow['neuron_id']
                        nid = neuron_nid[neid]
                        fid = arow['front_id']
                        front = self.constellation.front_by_id(ID(nid,fid))
                        try:
                            setattr(front,attrib,arow[attrib])
                        except:
                            raise BugError("import_simulation","missing additional attribute '" + str(attrib) + "' in front " + str(front))
                # reinitialize storage if necessary
                if row['last_only']: # last_only storage
                    if self.importable_db: # only do this if importable_db flag set
                        self.attrib_to_db(self._neuron_types[nid-1],attrib,\
                                          row['type'],object=obj_type,\
                                          neuron_name=row['neuron_name'],\
                                          last_only=True)
                else: # always do continuous storage
                    self.attrib_to_db(self._neuron_types[nid-1],attrib,\
                                          row['type'],object=obj_type,\
                                          neuron_name=row['neuron_name'])
        if nret['arcs']: # check whether all processes completed _update_empty_arcs
            for pid in range(2,self._num_procsP2):
                first = True
                while self._proc_status[pid] != 1:
                    if first and self.verbose >= 6:
                        print ("Admin waiting for Process",pid,"to finish _update_empty_arcs")
                        first = False
                    time.sleep(PAUSE)
        self.cycle = num_cycles
        if self.verbose > 1:
            print (Fore.BLUE + "Admin: import database lasted","{:8.2f}".format(time.time() - start),"seconds",Fore.RESET)

    def plot_item(self,item,color='r',symbol=None,line=False):
        """ Plot a structure in an interactive notebook plotting context. Item can be a ``Point``, ``Front`` or ``Substrate`` or a list of these.      
        Parameters
        ----------
        item : ``Point``, ``Front`` or ``Substrate` or list : item(s) to be plotted.
        Optional:
        color : char or string : valid color name to use, default red
        line : boolean : how to plot a list of ``Point``, as individual points (False) or as a line (True), default False
        symbol : char or None : symbol to use for ``Point`` or ``Substrate`` instead of default one, default None 

        Returns
        -------
        matplotlib object: plot object
        """
        if isinstance(item,list): # turn all into list
            if line and isinstance(item[0],Point): # plot a line
                lines = []
                for n in range(len(item) - 1):
                    line = self._ax.plot([item[n].x,item[n+1].x],\
                                     [item[n].y,item[n+1].y],\
                                     [item[n].z,item[n+1].z],\
                                     linewidth=0.5,color=color)
                    lines.append(line)
                return lines
            else: # plot items
                items = item
        else:
            items = [item]
        for struct in items:
            if isinstance(struct,Front):
                if struct.is_cylinder(): # cylinder -> line
                    line = self._lines[struct._key()] = self._ax.plot([struct.orig.x,struct.end.x],\
                                     [struct.orig.y,struct.end.y],\
                                     [struct.orig.z,struct.end.z],\
                                     linewidth=2*struct.radius,color=color)
                else: # sphere -> circle
                    x = struct.orig.x + (struct.radius * self._x_outer)
                    y = struct.orig.y + (struct.radius * self._y_outer)
                    z = struct.orig.z + (struct.radius * self._z_outer)
                    line = self._lines[struct._key()] = self._ax.plot_surface(x,y,z,rstride=2,cstride=2,\
                                color=color,linewidth=0,antialiased=False,\
                                shade=False)
            elif isinstance(struct,Point):
                if symbol:
                    mark = symbol
                else:
                    mark = 'o'
                line = self._ax.scatter(struct.x,struct.y,struct.z,\
                                                color=color,marker=mark)
            elif isinstance(item,Substrate):
                if symbol:
                    mark = symbol
                else:
                    mark = '+'
                line = self._ax.scatter(struct.orig.x,struct.orig.y,\
                            struct.orig.z,color=color,marker=mark)
            else:
                raise TypeError("item cannot be plotted")
        self._fig.canvas.draw()
        return line
                
    def flash_front(self,front):
        """ Flashes a plotted front by changing its color a few times.      
        Parameters
        ----------
        front : ``Front`` : front to flash.
        """
        key = front._key()
        if key in self._lines:
            line = self._lines[key]
            sphere = not front.is_cylinder()
            if sphere:
                if self._soma_black:
                    orig_col = 'k'
                    flash_col = 'r'
                else:
                    orig_col = 'r'
                    flash_col = 'k'
                pause = PAUSE
            else:
                orig_col = line[0].get_color()
                if (orig_col == 'r') or (orig_col == 'tab:red'):
                    flash_col = 'k'
                else:
                    flash_col = 'r'
                pause = 10*PAUSE
            for i in range(5):
                if sphere:
                    line.set_facecolor(flash_col)
                else:
                    line[0].set_color(flash_col)
                self._fig.canvas.draw()
                time.sleep(pause)
                if sphere:
                    line.set_facecolor(orig_col)
                else:
                    line[0].set_color(orig_col)
                self._fig.canvas.draw()
        elif self.verbose >= 2:
            print (Fore.MAGENTA + "Warning: front not in list of plot items",Fore.RESET)

    # run a number of cycles in simulation:
    #  is a stub for launch by Admin_agent (standard) or Admin_slave (notebook plotting)
    def simulation_loop(self,num_cycles):
        """ Runs the simulation for a number of cycles.
        
        Parameters
        ----------
        num_cycles : integer > 0.
        """
        if num_cycles <= 0:
            raise ValueError("num_cycles","larger than zero")
        if not self._simulation_allowed:
            raise TypeError("simulation_loop disabled by import_simulation")
        if not self.constellation._automatic:
            raise TypeError("simulation_loop disabled by interactive mode")
        if self._num_admins == 1: # one loop running that also performs lock brokering
            if self.verbose >= 6:
                print ("Admin starting _simulation_loop1")
            self._simulation_loop1(num_cycles)
        elif self._num_admins == 2: # separate loop file saving and plotting
            if self._plot:
                if self.verbose >= 4:
                    print ("Admin starting _simulation_loop2p")
                self._simulation_loop2p(num_cycles)
            else:
                if self.verbose >= 4:
                    print ("Admin starting _simulation_loop2")
                self._simulation_loop2(num_cycles)
        elif self._num_admins == 3: # separate loop for lock brokering
            if self.verbose >= 4:
                print ("Admin starting _simulation_loop3")
            self._simulation_loop3(num_cycles)
        else:# start plotting loop of Admin_agent
            if self.verbose >= 4:
                print ("Master starting _master_loop")
            self._master_loop(num_cycles)
            
    def attrib_to_db(self,front_class,attribute,sql_type,neuron_name="",\
                     object=Front,last_only=False):
        """ Store (new) attribute(s) in the simulation database. This can be used to either store changes to predefined attributes like synaptic *weight* or neuron *firing rate*, or to store user defined additional attributes. A new database table with the name of attribute will be created.
        
        By default *attrib_to_db* starts storing specified ``Front`` attribute(s) for **all active fronts** belonging to *front_class* after this method was run and continues till end of the simulation. Alternatively, active fronts stored can be limited to those of a specific neuron using optional *neuron_name*. 
        
        Another mode of output occurs when optional ``last_only=True``. Now specified attribute(s) for **all fronts** belonging to *front_class* (including inactive ones) are stored only after last cycle of the simulation during ``admin.destruction()`` call.
        
        To store (only predefined) ``Neuron`` (*firing_rate*,...) or ``Synapse`` (at present only *weight* changes) attributes the *object* optional parameter should be used.
        
        This method can only be **called after at least one neuron of *front_class* is instantiated**: ``add_neurons`` must have been called. Only *ctypes* attributes or ``ID`` or ``Point`` (for ``Fronts`` only) can be stored, not other classes can be stored. 
        
        Parameters
        ----------
        front_class : string : name of the ``Front`` subclass to store.
        attribute : string or [string] : name(s) of the attribute(s) to store. Can be a single attribute or a list of attributes from same *front_class*. Recognized predefined attributes are: *CV_ISI*, *firing_rate* and *weight*.
        sql_type : string or [string] : defines for each attribute the data type to be used in the database, choice of "id", "int", "point", "real", "text".
        Optional:
        last_only : boolean : output for all fronts but only at time of ``admin.destruction()``, default False: store for active fronts every cycle from now on.
        neuron_name : string : output only attributes from fronts belonging to neuron with *neuron_name* (wildcard), default: "" (all neurons).
        object : ``Front``, ``Neuron`` or ``Synapse`` : output attribute from one of these objects, default ``Front``.
        """
        if self.verbose >= 4:
            if last_only:
                print ("Admin setting up storage of",attribute,"of",front_class,"at end of simulation")
            else:
                 print ("Admin setting up continuous storage of",attribute,"of",front_class)
       # get neuron_type_index, this also tests for existence of front_class
        front_class = _strip_neuron_type(front_class)
        type_ind = self.constellation.neuron_type_index(front_class)
        # get all neurons affected
        if len(neuron_name) > 0:
            neurons = \
                self.constellation.neurons_by_name(name,type_index=type_ind)
        else:
            neurons = self.constellation.neurons_by_type(type_ind)
        if not neurons:
            raise TypeError("no neurons found for front_class: " + str(front_class))
        # code will test existence of attributes on first neuron
        neuron = neurons[0]
        soma = neuron.get_neuron_soma(self.constellation)
        # test optional object
        if object == Front: # default
            obj_type = 1
        elif object == Neuron:
            obj_type = 2
        elif object == Synapse:
            obj_type = 3
            if not hasattr(soma,'_yid'):
                raise TypeError("Illegal Synapse object: " + str(front_class) + " is not a SynFront")
        else:
            raise TypeError("Unknown optional object " + str(object))     
        # sql types implemented
        legal_sql = ["id","int","point","real","text"]
        # now test existence of attributes
        if isinstance(attribute,list): # list of attributes provided
            if not isinstance(sql_type,list):
                raise TypeError("attribute and sql_type should both be lists if attribute is a list")
            if len(sql_type) != len(attribute):
                raise TypeError("attribute and sql_type should both be lists of same length")
            for attr in attribute:
                if not hasattr(soma,attr):
                    raise TypeError(str(front_class) + " does not have attribute " + str(attr))
                    return
                if last_only:
                    if attr in self._storef_attrs[type_ind]:
                        raise TypeError(str(attr) + " already listed for last_only storage")
            n = 0
            for sqlt in sql_type:
                if sqlt not in legal_sql:
                    raise ValueError("sql_type","from " + str(legal_sql))
                if (sqlt == "id") or (sqlt == "point"):
                    if obj_type != 1: # IDs only allowed for fronts
                        raise TypeError(str(attribute[n]) + " ID or Point can only be stored for Front")
                n += 1
            attrs = attribute
            sqls = sql_type
        else:
            if isinstance(sql_type,list):
                raise TypeError("sql_type should not be a list if attribute is not")
            # differentiate between Front, Neuron or Synapse attributes
            if obj_type == 1:
                if not hasattr(soma,attribute):
                    raise TypeError(str(front_class) + " does not have attribute " + str(attribute))
                if attribute in ['orig','end','name','radius','path_length',\
                                 'swc_type','order','birth','death']:
                    raise TypeError("Front attribute '" + str(attribute) + "' already stored in database")
            elif obj_type == 2:
                if not hasattr(Neuron,attribute):
                    raise TypeError("Neurons do not have attribute " + str(attribute))
                if attribute in ['neuron_name','soma_ID','num_fronts',\
                                 'num_retracted','num_synapses']:
                    raise TypeError("Neuron attribute '" + str(attribute) + "' already stored in database")
            elif obj_type == 3:
                if not hasattr(Synapse,attribute):
                    raise TypeError("Synapses do not have attribute " + str(attribute))
                if attribute in ['pre_syn','post_syn']:
                    raise TypeError("Synapse attribute '" + str(attribute) + "' already stored in database")
            if last_only:
                if attribute in self._storef_attrs[type_ind]:
                    raise TypeError(str(attribute) + " already listed for last_only storage")
            if sql_type not in legal_sql:
                raise ValueError("sql_type","from " + str(legal_sql))
            if (sql_type == "id") or (sql_type == "point"):
                if obj_type != 1: # IDs only allowed for fronts
                    raise TypeError(str(attribute[n]) + " ID or Point can only be stored for Front")
            attrs = [attribute] # convert to list
            sqls = [sql_type]
        # make the new database tables
        if self._num_store_db == 0: # make attributes table
            self._attr_DB()
        for n in range(len(attrs)):
            attr = attrs[n]
            sqlt = sqls[n]
            tname = str(attr) + "_data"
            if tname in self._db_tables:
                raise TypeError("cannot store " + attr + " because name already used for standard database table " + tname)            
            if tname not in self._db_new_tables: # make a new table
                if sqlt == "id":
                    conn_str = "CREATE TABLE " + tname + \
                           " (id INTEGER PRIMARY KEY AUTOINCREMENT, " + \
                           "neuron_id int, front_id int, cycle int, " + \
                           str(attr) + "_0 int, " + str(attr) + "_1 int)"
                elif sqlt == "point":
                    conn_str = "CREATE TABLE " + tname + \
                           " (id INTEGER PRIMARY KEY AUTOINCREMENT, " + \
                           "neuron_id int, front_id int, cycle int, " + \
                           str(attr) + "_x real, " + str(attr) + "_y real, " + \
                           str(attr) + "_z real)"
                else:
                    conn_str = "CREATE TABLE " + tname + \
                           " (id INTEGER PRIMARY KEY AUTOINCREMENT, " + \
                           "neuron_id int, front_id int, cycle int, " + \
                           str(attr) + " " + str(sqlt) + ")"
                self._conn.execute(conn_str)
                self._num_store_db += 1 # increase number of tables stored
                self._conn.cursor().execute("UPDATE neurodevsim SET attributes = ? WHERE id = 1",(self._num_store_db,))
                self._conn.cursor().execute("INSERT into attributes VALUES (?,?,?,?,?)" ,(None,tname,sqlt,neuron_name,last_only))
                self._db_new_tables.append(tname)
            ind = self._db_new_tables.index(tname)
            if last_only:
                self._storef_db_tabs[type_ind].append(ind)
            else:
                self._storec_db_tabs[type_ind].append(ind)
            if sqlt == "id": # gets its own obj_type
                self._db_new_objects[type_ind].append(4)
            elif sqlt == "point": # gets its own obj_type
                self._db_new_objects[type_ind].append(5)
            else:
                self._db_new_objects[type_ind].append(obj_type)
        # prepare for last_only=True storage
        if last_only:
            self._store_types[type_ind] = 2
            self._storef_attrs[type_ind].extend(attrs)
            return
        # prepare for continuous storage
        self._storec_attrs[type_ind].extend(attrs)
        self._store_fronts = []
        # all active fronts, set only after first cycle
        active_f = self._growing_fronts + self._migrating_fronts + \
                      self._active_fronts
        # tag all somata of storing neurons
        for neuron in neurons:
            soma = self._fronts[neuron.soma_ID._nid][neuron.soma_ID._fid]
            soma._set_storing()
            if not active_f: # before first cycle
                self._store_fronts.append(soma)
        # tag and add to _store_fronts all active storing neurons
        for front in active_f:
            if front.order == 0: # a soma
                neid = front._sid
            else:
                neid = self._fronts[front._nid][front._sid]._sid
            fneuron = self._neurons[neid]
            if fneuron in neurons: # this front stores attribute(s)
                front._set_storing() # mark as storing
                self._store_fronts.append(front)
                    
    def sim_memory(self):
        """ Print memory use in Mb.
        """
        pid = os.getpid()
        py = psutil.Process(pid)
        real_mem = int(py.memory_info()[0]/1e+6)  # real memory use in MB
        virt_mem = int(py.memory_info()[1]/1e+6)  # virtual memory use in MB
        print ("Admin with",self._num_procs+1,"cores, real memory:",real_mem,"MB, virtual memory:",virt_mem,"MB")


    def sim_statistics(self,verbose=3):
        """ Print runtime statistics for a simulation and returns runtime.
        
        Print processor specific waiting times and load balancing statistics.
        
        Parameters
        ----------
        Optional Parameters
        verbose : integer 0-3 : print increasing amount of data, for verbose > 2 processor specific data is included.
        """
        runtime = time.time()-self.start
        if verbose == 1:
            print (self._num_procs+self._num_admins,",{:8.2f} ,".format(runtime))
        else:
            print ("On",self._num_procs+min(2,self._num_admins),"cores: total run time {:8.2f}".format(runtime),"seconds")
        if verbose > 2:
            self._clock.value = -10 # start processes with instruct
        return runtime
        
    def grid_statistics(self,bin_width=1):
        """ Print a histogram about grid use by fronts.
        
        Parameters
        ----------
        bin_width : integer : width of bins to use, default 1.
        
        Returns
        -------
        grid data : list containing number of fronts for each grid point
        """
        data = [] # one entry for each grid point
        n_max = 0 # largest entry -> sets number of bins
        nt = 0 # total number of entries
        for gid in range(self._volume.grid_max):
            exid = self._grid[gid]
            if exid == 0: # empty
                data.append(0)
                continue
            n = 0 # number of entries for this grid point
            finished = False
            while True:
                for i in range(self._extra_size):
                    gfid = self._grid_extra[exid + i]
                    if gfid._nid > 0: # valid entry
                        n += 1
                        if i == self._extra_size_1: # last entry of block filled
                            finished = True
                            break
                    if gfid._nid == 0: # end of used block
                        finished = True
                        break
                    elif gfid._nid < 0: # link to next block
                        exid = -gfid._nid
                        break
                if finished:
                    break
            if n > n_max:
                n_max = n
            nt += n
            data.append(n)
        # make bins
        edges = [0]
        e = int(bin_width)
        while e < n_max:
            edges.append(e)
            e += int(bin_width)
        hist, bin_edges = np.histogram(data,bins=edges)
        print ("Grid use for grid_step",self._volume.grid_step,":")
        for n in range(len(hist)):
            print (bin_edges[n],"<",bin_edges[n+1],":",hist[n])
            #print ("{:4.1f}".format(bin_edges[n]),"- {:4.1f}:".format(bin_edges[n+1]),hist[n])
        print ("Mean number of grid points used for each front: {:5.2f}".format(nt / self.total_fronts))
        return data

    def destruction(self,update_db=True,exit=True):
        """ Shut down NeuroDevSim: end all processes and admin.
        
        By default it also quits the program, any following statements are ignored.
        
        Parameters
        ----------
        Optional:
        exit : boolean : exit program after shutting down, default: True
        update_db : boolean : update database with final values. As this updates all front data this may take a few seconds. Affects many tables, default: True
        """
        if self.constellation._automatic:
            self._close_DB(update_db)
            for p in self.procs: # includes self.admin2 if forked
                if p.is_alive():
                    p.terminate()
        if self.verbose == -2: # benchmarking option
            print ("{:8.2f}".format(time.time()-self.start),"seconds")
        if self.verbose >= 3:
            print (self.total_fronts,"fronts made for seed",self.seed,"on",self._num_procs,"processes")
        if self.verbose > 0:
            print (Fore.BLUE + "NeuroDevSim admin terminated, total run time","{:8.2f}".format(time.time()-self.start),"seconds",Fore.RESET)
        if exit and not self._plot:
            sys.exit()

    # run a number of cycles in simulation dong all Admin duties
    #   this is the complete simulation loop for a single Admin without plotting
    def _simulation_loop1(self,num_cycles):
        # make sure processes are online
        gids = [] # not used in single process run but referenced in print
        if self._first_cycle: 
            for pid in range(2,self._num_procsP2):
                first = True
                while self._proc_status[pid] == 0:
                    if first and self.verbose >= 6:
                        print ("Admin waiting for Process",pid,"to activate")
                        first = False
                    time.sleep(PAUSE)
        self._first_cycle = False
        for nc in range(num_cycles):
            self.cycle += 1
            if self.verbose > 0:
                print (Fore.BLUE + "NeuroDevSim admin starting cycle",self.cycle,Fore.RESET)
            if self.cycle in self._future_active:
                for item in self._future_active[self.cycle]:
                    status = item[0]
                    f = item[1]
                    f._set_active() # always make active
                    if status == b'i': # make active only
                        if f not in self._active_fronts:
                            self._active_fronts.append(f)
                    elif status == b'g': # make growing
                        f.set_growing()
                        if f not in self._growing_fronts:
                            self._growing_fronts.append(f)
                    elif status == b'm': # make migrating
                        f.set_migrating()
                        if f not in self._migrating_fronts:
                            self._migrating_fronts.append(f)
                    else:
                        raise BugError("_simulation_loop1","unknown status " + str(status))
                    if f._does_storing() and f not in self._store_fronts:
                        self._store_fronts.append(f)
                del self._future_active[self.cycle]
            new_f = [] # list of new fronts for database
            newaxon_f = [] # list of new trailing axons for database
            migrated_f = [] # list of new migrated fronts for database
            retracted_f = [] # list of roots of branches to retract
            new_syn_id = [] # list of new synapses for database
            del_syn_id = [] # list of deleted synapses fronts for database
            any_f_moved = False # check whether any migration happened this cycle
            deleted_f = [] # list of all fronts that were deleted
            new_inactive = [] # list of fronts that become inactive next cycle
            next_growing = [] # self._growing_fronts for next cycle
            next_migrating = [] # self._migrating_fronts for next cycle
            next_active = [] # self._active_fronts for next cycle
            num_f_in = num_f_out = len(self._growing_fronts) + \
                                   len(self._migrating_fronts) + \
                                   len(self._active_fronts)
            if num_f_in == 0:
                if self.verbose > 0:
                    print (Fore.BLUE + "NeuroDevSim admin ending: no active fronts on cycle",self.cycle,Fore.RESET)        
                return
            elif num_f_in < self._num_procs:
                max_procs = 2 + num_f_in
                stop_procs = True
            else:
                max_procs = self._num_procsP2
                stop_procs = False
            # clear index of first entry of self._new_AIDs and remove negat_ID in Processes
            for pid in range(2,self._num_procsP2):
                self._actives[pid] = empty_ID
                self._new_AIDs[self._n_starts[pid]].index = 0
            start_AIDs = [0] * self._num_procsP2 # relative start of return data from each processor in self._new_AIDs
            self._clock.value = self.cycle # start processes
            if self.verbose >= 6:
                print ("Admin sending",num_f_in,"fronts",max_procs,stop_procs)
            start = time.time()
            # keep cycling till all fronts are processed
            # current balancing approach: tightly locked -> cores will have to
            #   wait for next front
            if self._num_procs > 1:
                lock_gids_th = self._num_procsP2 # lock gids if num_f_in > lock_gids_th
            else: # don't bother locking on single processor run
                lock_gids_th = num_f_in # never lock
            # control pre-fetching of fronts: not at end because some
            #   processes may take a lot of time and cannot execute the 
            #   prefetched front.
            reserve_th = self._num_procsP2 # get next front ready if num_f_in > _num_procsP2
            grow_index0 = 0 # start from beginning of list
            migr_index0 = 0 # start from beginning of list
            if self._debug: # keep track of ids sent and received
                target = num_f_in
                sent_id = []
                receiv_id = []
            n_fails = 0 # number of failed fetch loops since last success
            if self.verbose >= 6:
                last_fid = 0 # used to prevent endless repetition of same message
            while num_f_out: 
                self._lock_broker() # essential duty, called many times
                # find pid needing a front or prefetch
                for pid in range(2,max_procs):
                    ns0 = self._n_starts[pid] + start_AIDs[pid]
                    aid = self._new_AIDs[ns0]
                    ns1 = aid.index # encodes action to take: 0 for no action
                    while ns1 > 0: # possibly data for multiple parent fronts
                        num_f_out -= 1 # finished one
                        # process front that was called
                        id = aid.ID
                        if self._debug: # keep track of ids sent and received
                            receiv_id.append(id)
                        if self.verbose >= 6:
                            print ("Admin received from",pid,id,ns0,ns1,num_f_in,num_f_out,n_fails)
                        front = self._fronts[id._nid][id._fid]
                        if front._gid < 0: # locked gids
                            # unlocked by process
                            front._gid = -front._gid # set back to unlocked
                            # reset to start of lists as gid has been freed
                            grow_index0 = 0 # start from beginning of list
                            migr_index0 = 0 # start from beginning of list
                        # store for next cycle if active, automatically in order called now
                        if front.is_migrating():
                            next_migrating.append(front)
                            if front not in self._mig_fronts:
                                migrated_f.append(front)
                            if front.has_moved():
                                any_f_moved = True
                            if front._does_storing() and \
                                                front not in self._store_fronts:
                                self._store_fronts.append(front)
                        elif front.is_growing():
                            next_growing.append(front)
                            if front._does_storing() and \
                                                front not in self._store_fronts:
                                self._store_fronts.append(front)
                        elif front.is_active():
                            next_active.append(front)
                            if front._does_storing() and \
                                                front not in self._store_fronts:
                                self._store_fronts.append(front)
                        elif (front in self._store_fronts) and (front.order > 0):
                            self._store_fronts.remove(front) # not active anymore
                        # process other new_AIDs
                        for n in range(ns0 + 1, self._n_starts[pid] + ns1):
                            self._lock_broker() # essential duty, called many times
                            status = self._new_AIDs[n].status
                            id = self._new_AIDs[n].ID
                            if status == b'e':
                                front = self._fronts[id._nid][id._fid]
                                new_f.append(front)
                                if front.is_growing():
                                    next_growing.append(front)
                                    if front._does_storing() and \
                                                front not in self._store_fronts:
                                        self._store_fronts.append(front)
                                elif front.is_active():
                                    next_active.append(front)
                                    if front._does_storing() and \
                                                front not in self._store_fronts:
                                        self._store_fronts.append(front)
                            elif status == b't': # a trailing axon was created
                                front = self._fronts[id._nid][id._fid]
                                newaxon_f.append(front)
                                if front.is_active():
                                    next_active.append(front)
                                    if front._does_storing() and \
                                                front not in self._store_fronts:
                                        self._store_fronts.append(front)
                            elif status == b'r': # retracted front
                                front = self._fronts[id._nid][id._fid]
                                deleted_f.append(front)
                                new_inactive.append(front) # sometimes kept active
                            elif status == b'a': # enable front
                                front = self._fronts[id._nid][id._fid]
                                if front.is_migrating():
                                    next_migrating.append(front)
                                    if front not in self._mig_fronts:
                                        migrated_f.append(front)
                                    if front._does_storing() and \
                                                    front not in self._store_fronts:
                                        self._store_fronts.append(front)
                                elif front.is_growing():
                                    next_growing.append(front)
                                    if front._does_storing() and \
                                                    front not in self._store_fronts:
                                        self._store_fronts.append(front)
                                elif front.is_active():
                                    next_active.append(front)
                                    if front._does_storing() and \
                                                    front not in self._store_fronts:
                                        self._store_fronts.append(front)
                            elif (status == b'i') or (status == b'g') or\
                                        (status == b'm'): # disable front
                                front = self._fronts[id._nid][id._fid]
                                # can be added to next_active later
                                new_inactive.append(front)
                                till_cycle = self._new_AIDs[n].index
                                if till_cycle > self.cycle:
                                    # future activation at till_cycle requested
                                    if till_cycle in self._future_active:
                                        self._future_active[till_cycle].\
                                                    append([status,front])
                                    else:
                                        self._future_active[till_cycle] = \
                                                    [[status,front]]
                            elif status == b'd': # retract branch or single front
                                front = self._fronts[id._nid][id._fid]
                                retracted_f.append(front)
                            elif status == b's': # store substrate
                                ind = id._fid
                                self.add_substrate(self._substrate[ind],index=ind)
                            elif status == b'y': # store synapse
                                new_syn_id.append(id._fid)
                            elif status == b'z': # remove synapse
                                del_syn_id.append(id._fid)
                            else:
                                print (pid,self._n_starts[pid],start_AIDs[pid],ns0,ns1,n)
                                raise BugError("_simulation_loop","on " + str(pid) + " unknown self._new_AIDs status " + str(self._new_AIDs[n]))
                        start_AIDs[pid] = ns1 # update for next round
                        ns0 = self._n_starts[pid] + ns1
                        # check whether more data ready
                        aid = self._new_AIDs[ns0]
                        ns1 = aid.index
                    # Deal with end of cycle
                    if num_f_in == 0: # no more fronts to send
                        # check that both fetch and pre-fetch are done or in process
                        if (self._actives[pid]._nid == 0) and \
                            (self._actives[pid + self._num_procs]._nid == 0):
                                self._actives[pid] = negat_ID # signal end of cycle
                        continue
                    # Check whether we should fetch or pre-fetch
                    if (self._actives[pid]._nid > 0) and (num_f_in <= reserve_th):
                        # previous fetch unused and no more pre-fetching
                        #   (works also for negat_ID in self._actives[pid])
                        continue
                    # check whether pre-fetch necessary
                    if self._actives[pid + self._num_procs]._nid > 0: 
                        continue # already done prefetch
                    # pid has no instructions
                    found = False # found a front for pid
                    # decide whether we should check for gid competition
                    if n_fails <= self._num_procs: # not too many fails
                        lock_gids = num_f_in > lock_gids_th
                    else:
                        lock_gids = False 
                        grow_index0 = 0 # start from beginning of list
                        migr_index0 = 0 # start from beginning of list
                    # do growing fronts first
                    for n in range(grow_index0,len(self._growing_fronts)):
                        front = self._growing_fronts[n]
                        if self.verbose >= 6:
                            print ("Admin trying grow",pid,front)
                        self._lock_broker() # essential duty, called many times
                        if lock_gids: # check whether free
                            gids = []
                            # check gid availibility: bypass _get_gids for speed
                            ngid = front._gid
                            failed = False
                            for i in range(ngid + 1, ngid + 1 + self._gids[ngid]):
                                gid = self._gids[i]
                                if self._grid_wlock[gid] != 0:
                                    if (self.verbose >= 6) and (front._fid != last_fid):
                                        last_fid = front._fid # do not repeat this one
                                        print ("Admin grow failed",pid,num_f_in,gid,self._grid_wlock[gid],front)
                                    grow_index0 = n # skip this one next pid
                                    failed = True # already locked
                                    break
                                gids.append(gid)
                            if failed:
                                continue # for front in self._growing_fronts
                        # use this front
                        if self._actives[pid]._nid > 0: # save in reserve
                            if self.verbose >= 6:
                                print ("Admin sent growing reserve to",pid,front.get_id(),gids)
                            self._actives[pid + self._num_procs] = front.get_id()
                        else: # save in regular place
                            if self.verbose >= 6:
                                print ("Admin sent growing to",pid,front.get_id(),gids)
                            self._actives[pid] = front.get_id()
                        if self._debug: # keep track of ids sent and received
                            sent_id.append(front.get_id())
                        num_f_in -= 1 # started one
                        found = True
                        if lock_gids: # now really lock them
                            for gid in gids: # lock gids for admin
                                self._grid_wlock[gid] = pid # lock to process
                            front._gid = -ngid # mark as locked
                        self._lock_broker() # essential duty, called many times
                        del self._growing_fronts[n] # done this one
                        break
                    if found:
                        n_fails = 0 # reset
                        continue # for pid in range
                    # do migrating fronts next
                    for n in range(migr_index0,len(self._migrating_fronts)):
                        front = self._migrating_fronts[n]
                        if self.verbose >= 6:
                            print ("Admin trying mig",pid,front)
                        self._lock_broker() # essential duty, called many times
                        if lock_gids: # check whether free
                            gids = []
                            # check gid availibility: bypass _get_gids for speed
                            ngid = front._gid
                            failed = False
                            for i in range(ngid + 1, ngid + 1 + self._gids[ngid]):
                                gid = self._gids[i]
                                if self._grid_wlock[gid] != 0:
                                    if (self.verbose >= 6) and (front._fid != last_fid):
                                        last_fid = front._fid # do not repeat this one
                                        print ("Admin mig failed",pid,num_f_in,gid,self._grid_wlock[gid],front)
                                    migr_index0 = n # skip this one next pid
                                    failed = True # already locked
                                    break
                                gids.append(gid)
                            if failed:
                                continue # for front in self._migrating_fronts
                        # use this front
                        if self._actives[pid]._nid > 0: # save in reserve
                            if self.verbose >= 6:
                                print ("Admin sent migrating reserve to",pid,front.get_id(),gids)
                            self._actives[pid + self._num_procs] = front.get_id()
                        else: # save in regular place
                            if self.verbose >= 6:
                                print ("Admin sent migrating to",pid,front.get_id(),gids)
                            self._actives[pid] = front.get_id()
                        if self._debug: # keep track of ids sent and received
                            sent_id.append(front.get_id())
                        num_f_in -= 1 # started one
                        found = True
                        if lock_gids: # now really lock them
                            for gid in gids: # lock gids for admin
                                self._grid_wlock[gid] = pid # lock to process
                            front._gid = -ngid # mark as locked
                        self._lock_broker() # essential duty, called many times
                        del self._migrating_fronts[n] # done this one
                        break
                    if found:
                        n_fails = 0 # reset
                        continue # for pid in range                            
                    # do active fronts last
                    if self._active_fronts:
                        if self.verbose >= 6:
                            print ("Admin trying act",pid,front)
                        self._lock_broker() # essential duty, called many times
                        # use this front
                        fid = self._active_fronts.pop(0).get_id()
                        if self._actives[pid]._nid > 0: # save in reserve
                            self._actives[pid + self._num_procs] = fid
                            if self.verbose >= 6:
                                print ("Admin sent active reserve to",pid,self._actives[pid + self._num_procs])
                        else: # save in regular place
                            self._actives[pid] = fid
                            if self.verbose >= 6:
                                print ("Admin sent active to",pid,self._actives[pid])
                        if self._debug: # keep track of ids sent and received
                            sent_id.append(fid)
                        num_f_in -= 1 # started one
                        n_fails = 0 # reset
                        continue # for pid in range
                    # failed to find one
                    if self._actives[pid]._nid == 0: # failure of a fetch cycle
                        n_fails += 1
                if stop_procs: # less active fronts than processes
                    self._lock_broker() # essential duty, called many times
                    for pid in range(max_procs,self._num_procsP2):
                       self._actives[pid] = negat_ID # signal end of cycle
                    stop_procs = False # do this only once 
            # do branch retractions
            if retracted_f:
                r_del = self._retract_branches(retracted_f,new_inactive)
                deleted_f += r_del
            self._lock_broker()
            # save new/changed fronts from in database
            if new_f:
                self.total_fronts += len(new_f)
                self._write_new_DB(new_f)
                self._lock_broker()
            if migrated_f:
                self._new_mig_DB(migrated_f)
                self._lock_broker()
            if any_f_moved:
                self._write_mig_DB()
                self._lock_broker()
            if newaxon_f:
                self.total_fronts += len(newaxon_f)
                self._write_new_DB(newaxon_f,trail_axon=True)
                self._lock_broker()
            if new_syn_id:
                self._write_syn_DB(new_syn_id)
                self._lock_broker()
            if deleted_f:
                self._DB_del_update(deleted_f)
                self._lock_broker()
            if del_syn_id:
                self._del_syn_DB(del_syn_id)
                self._lock_broker()
            if self._store_fronts:
                self._store_attribs()
                self._lock_broker()
            self._conn.commit()
            self._lock_broker()
            # reset self._growing_fronts, self._migrating_fronts, self._active_fronts
            for front in new_inactive: # make inactive and remove from lists
                self._lock_broker()
                if front.is_growing():
                    front.clear_growing()
                    if front in next_growing:
                        next_growing.remove(front)
                if front.is_migrating():
                    front.clear_migrating()
                    if front in next_migrating:
                        next_migrating.remove(front)
                front._clear_active()
                if front in next_active:
                    next_active.remove(front)
                if front in self._store_fronts:
                    self._store_fronts.remove(front)
            self._growing_fronts = next_growing
            self._migrating_fronts = next_migrating
            self._active_fronts = next_active
            # check that all processes have reached end of cycle status
            for n in range(10000):
                not_finish = 0
                for pid in range(2,max_procs):
                    self._lock_broker()
                    if self._proc_status[pid] != 20:
                        not_finish = pid
                if not_finish == 0:
                    break # out of for n
            if not_finish > 0:
                raise BugError("_simulation_loop","process " + str(not_finish) + " does not reach end of cycle")
            if self._debug:
                self._lock_check() # check that all locks have been cleared
        self._clock.value = -1 # end of simulation loop: go into winter sleep
            
    # run a number of cycles in simulation doing Admin file saving, interacts with _slave_loop1
    def _simulation_loop2(self,num_cycles):
        # make sure processes are online
        gids = [] # not used in single process run but referenced in print
        if self._first_cycle: 
            for pid in range(1,self._num_procsP2): # include slave1
                first = True
                while self._proc_status[pid] == 0:
                    if first and self.verbose >= 6:
                        print ("Admin waiting for Process",pid,"to activate")
                        first = False
                    time.sleep(PAUSE)
        # do we need to transmit new fronts to slave1?
        if self._growing_fronts or self._migrating_fronts or self._active_fronts:
            n = 0 # index into self._new_AIDs which gets overwritten, not used now
            num_soma = 3 + len(self._growing_fronts) # first range
            self._new_AIDs[n] = ActiveFrontID(empty_ID,b'1',num_soma)
            n += 1
            num_soma2 = num_soma + len(self._migrating_fronts) # next range
            self._new_AIDs[n] = ActiveFrontID(empty_ID,b'1',num_soma2)
            n += 1
            num_soma3 = num_soma2 + len(self._active_fronts) # next range
            self._new_AIDs[n] = ActiveFrontID(empty_ID,b'1',num_soma3)
            n += 1
            for soma in self._growing_fronts:
                self._new_AIDs[n] = ActiveFrontID(soma.get_id(),b'2',num_soma)
                n += 1
            for soma in self._migrating_fronts:
                self._new_AIDs[n] = ActiveFrontID(soma.get_id(),b'2',num_soma2)
                n += 1
            for soma in self._active_fronts:
                self._new_AIDs[n] = ActiveFrontID(soma.get_id(),b'2',num_soma3)
                n += 1
            if self.verbose >= 6:
                print ("Admin sending fronts to slave1",num_soma-3,num_soma2-num_soma,num_soma3-num_soma2)            
            self._new_AIDs[n].index = 0 # stop sign
            self._proc_status[1] = 5 # signal slave1 to start processing data
            # update number of outgoing fronts
            num_f_out = self._proc_status[0] + num_soma3 - 3
            # clear self._growing_fronts and self._migrating_fronts
            self._growing_fronts = [] # only used by add_neurons in _sim..._loop2
            self._migrating_fronts = [] # only used by add_neurons in _sim..._loop2
        else: # use old value for number of outgoing fronts
            num_f_out = self._proc_status[0]
        self._first_cycle = False
        # execute all cycles
        for nc in range(num_cycles):
            self.cycle += 1
            if self.verbose > 0:
                print (Fore.BLUE + "NeuroDevSim admin starting cycle",self.cycle,Fore.RESET)
            new_f = [] # list of new fronts for database
            newaxon_f = [] # list of new trailing axons for database
            migrated_f = [] # list of new migrated fronts for database
            retracted_f = [] # list of roots of branches to retract
            new_syn_id = [] # list of new synapses for database
            del_syn_id = [] # list of deleted synapses fronts for database
            any_f_moved = False # check whether any migration happened this cycle
            deleted_f = [] # list of all fronts that were deleted
            if num_f_out == 0:
                if self.verbose > 0:
                    print (Fore.BLUE + "NeuroDevSim admin ending: no active fronts on cycle",self.cycle,Fore.RESET)        
                return
            elif num_f_out < self._num_procs:
                max_procs = 2 + num_f_out
                stop_procs = True
            else:
                max_procs = self._num_procsP2
                stop_procs = False
            # check whether slave1 is ready
            if self.verbose >= 6:
                    print ("Admin status",self._proc_status[1])
            while self._proc_status[1] == 5: # still processing initial data
                time.sleep(PAUSE)
            if self.verbose >= 6:
                    print ("Admin status",self._proc_status[1])
            # clear index of first entry of self._new_AIDs and remove negat_ID in Processes
            for pid in range(2,self._num_procsP2):
                self._actives[pid] = empty_ID
                self._new_AIDs[self._n_starts[pid]].index = 0
            start_AIDs = [0] * self._num_procsP2 # relative start of process block
            self._clock.value = self.cycle # start processes
            if self.verbose >= 6:
                print ("Admin sending",num_f_out,"fronts",max_procs,stop_procs)
            start = time.time()
            # signal slave1 to start
            self._proc_status[1] = 10
            while num_f_out: 
                # process return from process: almost no overlap with slave1
                for pid in range(2,max_procs):
                    ns0 = self._n_starts[pid] + start_AIDs[pid]
                    aid = self._new_AIDs[ns0]
                    ns1 = aid.index
                    while ns1 > 0: # possibly data for multiple parent fronts
                        num_f_out -= 1 # finished one
                        # process front that was called
                        id = aid.ID
                        if self.verbose >= 6:
                            print ("Admin received from",pid,id,ns0,ns1,num_f_out)
                        front = self._fronts[id._nid][id._fid]
                        # store for next cycle if active, automatically in order called now
                        if front.is_migrating():
                            if front not in self._mig_fronts:
                                migrated_f.append(front)
                            if front.has_moved():
                                any_f_moved = True
                            if front._does_storing() and \
                                                front not in self._store_fronts:
                                self._store_fronts.append(front)
                        elif front.is_growing():
                            if front._does_storing() and \
                                                front not in self._store_fronts:
                                self._store_fronts.append(front)
                        elif front.is_active():
                            if front._does_storing() and \
                                                front not in self._store_fronts:
                                self._store_fronts.append(front)                 
                        elif (front in self._store_fronts) and (front.order > 0):
                            self._store_fronts.remove(front) # not active anymore
                        # process other new_AIDs
                        for n in range(ns0 + 1, self._n_starts[pid] + ns1):
                            status = self._new_AIDs[n].status
                            id = self._new_AIDs[n].ID
                            front = self._fronts[id._nid][id._fid]
                            if status == b'e':
                                new_f.append(front)
                                if front.is_growing() or front.is_active():
                                    if front._does_storing() and \
                                                front not in self._store_fronts:
                                        self._store_fronts.append(front)
                            elif status == b't': # a trailing axon was created
                                newaxon_f.append(front)
                                if front.is_active():
                                    if front._does_storing() and \
                                                front not in self._store_fronts:
                                        self._store_fronts.append(front)
                            elif status == b'a': # enable front
                                if front.is_migrating():
                                    if front not in self._mig_fronts:
                                        migrated_f.append(front)
                                    if front._does_storing() and \
                                                front not in self._store_fronts:
                                        self._store_fronts.append(front)
                                if front.is_growing() or front.is_active():
                                    if front._does_storing() and \
                                                front not in self._store_fronts:
                                        self._store_fronts.append(front)
                            elif (status == b'i') or (status == b'g') or\
                                        (status == b'm'): # disable front
                                pass
                            elif status == b'r': # retracted front
                                deleted_f.append(front)
                            elif status == b'd': # retract branch
                                retracted_f.append(front)
                            elif status == b's': # store substrate
                                ind = id._fid
                                # also writes to database
                                self.add_substrate(self._substrate[ind],index=ind)
                            elif status == b'y': # store synapse
                                new_syn_id.append(id._fid)
                            elif status == b'z': # remove synapse
                                del_syn_id.append(id._fid)
                            else:
                                raise BugError("_simulation_loop2p","on " + str(pid) + " unknown self._new_AIDs status " + str(self._new_AIDs[n]))
                        start_AIDs[pid] = ns1 # update for next round
                        ns0 = self._n_starts[pid] + ns1
                        # check whether more data ready
                        aid = self._new_AIDs[ns0]
                        ns1 = aid.index
                # save new/changed fronts in database during update cycle
                if new_f: # regular fronts
                    self.total_fronts += len(new_f)
                    self._write_new_DB(new_f)
                    new_f = []
                if newaxon_f: # trailing axons
                    self.total_fronts += len(newaxon_f)
                    self._write_new_DB(newaxon_f,trail_axon=True)
                    newaxon_f = []
                if new_syn_id:
                    self._write_syn_DB(new_syn_id)
                    new_syn_id = []
                if deleted_f: # declare death
                    self._DB_del_update(deleted_f)
                    deleted_f = []
                if del_syn_id:
                    self._del_syn_DB(del_syn_id)
                    del_syn_id = []
                if migrated_f: # update migration status
                    self._new_mig_DB(migrated_f)
                    migrated_f = []
            if any_f_moved: # add row to migration table
                self._write_mig_DB()
            # signal to slave1 that branch retractions can be done
            #   check whether slave1 has started
            if retracted_f:
                while self._proc_status[1] != 10: # slave1 not started yet
                    time.sleep(PAUSE)
                self._proc_status[1] = 15
            # check that all processes have reached end of cycle status
            if self.verbose >= 6:
                print ("Admin checking end of cycle")
            for n in range(10000):
                not_finish = 0
                for pid in range(1,max_procs):
                    if self._proc_status[pid] != 20:
                        not_finish = pid
                if not_finish == 0:
                    break # out of for n
                elif (n > 5000) and (not_finish == 1): # waiting for slave to complete retractions
                    time.sleep(PAUSE)                
            if not_finish > 0:
                raise BugError("_simulation_loop2","process " + str(not_finish) + " does not reach end of cycle")
            self._proc_status[1] = 25 # no need for lock brokering anymore
            num_f_out = self._proc_status[0] # get number fronts next cycle
            if retracted_f: # get deleted fronts
                n = 0 # index into self._new_AIDs
                deleted_f = []
                del_syn_id = []
                num_dels = self._new_AIDs[0].index # number of deleted fronts
                for n in range(num_dels):
                    id = self._new_AIDs[n].ID
                    self._new_AIDs[n].status = b'0'
                    front = self._fronts[id._nid][id._fid]
                    if front.has_synapse(): # remove the synapse
                        del_syn_id.append(abs(front._yid))
                        front.remove_synapse(self.constellation)
                    deleted_f.append(front)
                self._DB_del_update(deleted_f)
                if del_syn_id:
                    self._del_syn_DB(del_syn_id)
            if self._store_fronts:
                self._store_attribs(check=True)
            self._conn.commit() # save changes from this cycle
        self._clock.value = -1 # end of simulation loop: go into winter sleep
        self._proc_status[1] == 30

    # run a number of cycles in simulation doing Admin file saving and 
    #   plotting, interacts with _slave_loop1
    # identical to _simulation_loop2 but contains plotting instructions
    def _simulation_loop2p(self,num_cycles):
        # make sure processes are online
        gids = [] # not used in single process run but referenced in print
        if self._first_cycle: 
            for pid in range(1,self._num_procsP2): # include slave1
                first = True
                while self._proc_status[pid] == 0:
                    if first and self.verbose > 1:
                        print ("Admin waiting for Process",pid,"to activate")
                        first = False
                    time.sleep(PAUSE)
        # do we need to transmit new fronts to slave1?
        if self._growing_fronts or self._migrating_fronts or self._active_fronts:
            n = 0 # index into self._new_AIDs which gets overwritten, not used now
            num_soma = 3 + len(self._growing_fronts) # first range
            self._new_AIDs[n] = ActiveFrontID(empty_ID,b'1',num_soma)
            n += 1
            num_soma2 = num_soma + len(self._migrating_fronts) # next range
            self._new_AIDs[n] = ActiveFrontID(empty_ID,b'1',num_soma2)
            n += 1
            num_soma3 = num_soma2 + len(self._active_fronts) # next range
            self._new_AIDs[n] = ActiveFrontID(empty_ID,b'1',num_soma3)
            n += 1
            for soma in self._growing_fronts:
                if soma._key() not in self._lines: # prevent repeat plotting after import_simulation
                    self._plot_front(soma)
                self._new_AIDs[n] = ActiveFrontID(soma.get_id(),b'2',num_soma)
                n += 1
            for soma in self._migrating_fronts:
                if soma._key() not in self._lines: # prevent repeat plotting after import_simulation
                    self._plot_front(soma)                
                self._new_AIDs[n] = ActiveFrontID(soma.get_id(),b'2',num_soma2)
                n += 1
            for soma in self._active_fronts:
                if soma._key() not in self._lines: # prevent repeat plotting after import_simulation
                    self._plot_front(soma)
                self._new_AIDs[n] = ActiveFrontID(soma.get_id(),b'2',num_soma3)
                n += 1
            if self.verbose >= 6:
                print ("Admin sending fronts to slave1",num_soma-3,num_soma2-num_soma,num_soma3-num_soma2)
            self._new_AIDs[n].index = 0 # stop sign
            self._proc_status[1] = 5 # signal slave1 to start processing data
            # update number of outgoing fronts
            num_f_out = self._proc_status[0] + num_soma3 - 3
            # clear self._growing_fronts and self._migrating_fronts
            self._growing_fronts = [] # only used by add_neurons in _sim..._loop2
            self._migrating_fronts = [] # only used by add_neurons in _sim..._loop2
        else: # use old value for number of outgoing fronts
            num_f_out = self._proc_status[0]
        # execute all cycles
        self._first_cycle = False
        for nc in range(num_cycles):
            self.cycle += 1
            if self.verbose > 0:
                print (Fore.BLUE + "NeuroDevSim admin starting cycle",self.cycle,Fore.RESET)
            new_f = [] # list of new fronts for database
            newaxon_f = [] # list of new trailing axons for database
            migrated_f = [] # list of new migrated fronts for database
            retracted_f = [] # list of roots of branches to retract
            new_syn_id = [] # list of new synapses for database
            del_syn_id = [] # list of deleted synapses fronts for database
            any_f_moved = False # check whether any migration happened this cycle
            deleted_f = [] # list of all fronts that were deleted
            if num_f_out == 0:
                if self.verbose > 0:
                    print (Fore.BLUE + "NeuroDevSim admin ending: no active fronts on cycle",self.cycle,Fore.RESET)        
                return
            elif num_f_out < self._num_procs:
                max_procs = 2 + num_f_out
                stop_procs = True
            else:
                max_procs = self._num_procsP2
                stop_procs = False
            # check whether slave1 is ready
            while self._proc_status[1] == 5: # still processing initial data
                time.sleep(PAUSE)
            # clear index of first entry of self._new_AIDs and remove negat_ID in Processes
            for pid in range(2,self._num_procsP2):
                self._actives[pid] = empty_ID
                self._new_AIDs[self._n_starts[pid]].index = 0
            start_AIDs = [0] * self._num_procsP2 # relative start of process block
            self._clock.value = self.cycle # start processes
            if self.verbose >= 6:
                print ("Admin sending",num_f_out,"fronts",max_procs,stop_procs)
            start = time.time()
            # signal slave1 to start
            self._proc_status[1] = 10
            while num_f_out: 
                # process return from process: almost no overlap with slave1
                for pid in range(2,max_procs):
                    ns0 = self._n_starts[pid] + start_AIDs[pid]
                    aid = self._new_AIDs[ns0]
                    ns1 = aid.index
                    while ns1 > 0: # possibly data for multiple parent fronts
                        num_f_out -= 1 # finished one
                        # process front that was called
                        id = aid.ID
                        if self.verbose >= 6:
                            print ("Admin received from",pid,id,ns0,ns1,num_f_out)
                        front = self._fronts[id._nid][id._fid]
                        # store for next cycle if active, automatically in order called now
                        if front.is_migrating():
                            if front not in self._mig_fronts:
                                migrated_f.append(front)
                            if front.has_moved():
                                self._plot_migration(front)
                                any_f_moved = True
                            if front._does_storing() and \
                                                front not in self._store_fronts:
                                self._store_fronts.append(front)
                        elif front.is_growing():
                            if front._does_storing() and \
                                                front not in self._store_fronts:
                                self._store_fronts.append(front)
                        elif front.is_active():
                            if front._does_storing() and \
                                            front not in self._store_fronts:
                                self._store_fronts.append(front)                 
                        elif (front in self._store_fronts) and (front.order > 0):
                            self._store_fronts.remove(front) # not active anymore
                        # process other new_AIDs
                        for n in range(ns0 + 1, self._n_starts[pid] + ns1):
                            status = self._new_AIDs[n].status
                            id = self._new_AIDs[n].ID
                            front = self._fronts[id._nid][id._fid]
                            if status == b'e':
                                new_f.append(front)
                                self._plot_front(front)
                            elif status == b't': # a trailing axon was created
                                newaxon_f.append(front)
                                self._plot_front(front)
                            elif status == b'a': # enable front
                                if front.is_migrating():
                                    if front not in self._mig_fronts:
                                        migrated_f.append(front)
                                    if front._does_storing() and \
                                                front not in self._store_fronts:
                                        self._store_fronts.append(front)    
                                elif front.is_growing() or front.is_active():
                                    if front._does_storing() and \
                                                front not in self._store_fronts:
                                        self._store_fronts.append(front)    
                            elif status == b'i': # disable front
                                pass
                            elif status == b'r': # retracted front
                                deleted_f.append(front)
                                self._remove_plot(front)
                            elif status == b'd': # retract branch
                                #print ("_simulation_loop2p: status d",front._fid)
                                retracted_f.append(front)
                            elif status == b's': # store substrate
                                ind = id._fid
                                # also writes to database
                                self.add_substrate(self._substrate[ind],index=ind)
                            elif status == b'y': # store synapse
                                new_syn_id.append(id._fid)
                            elif status == b'z': # remove synapse
                                del_syn_id.append(id._fid)
                            else:
                                raise BugError("_simulation_loop2p","on " + str(pid) + " unknown self._new_AIDs status " + str(self._new_AIDs[n]))
                        start_AIDs[pid] = ns1 # update for next round
                        ns0 = self._n_starts[pid] + ns1
                        # check whether more data ready
                        aid = self._new_AIDs[ns0]
                        ns1 = aid.index
                # update plots for color_scheme==3
                if self._color_scheme == 3:
                    for n in range(len(self._color_fronts)):
                        item = self._color_fronts[n]
                        value = getattr(item[0],self._color_attrib)
                        if value != item[1]: # compare with last value
                            col2 = (value - self._color_min) * self._color_scale
                            self._color_fronts[n][1] = value # store new value
                            if item[0].is_cylinder():
                                self._lines[item[0]._key()][0].set_color(self._colors(col2))
                            else:
                                self._lines[item[0]._key()].set_color(self._colors(col2))
                # save new/changed fronts in database during update cycle
                if new_f: # regular fronts
                    self.total_fronts += len(new_f)
                    self._write_new_DB(new_f)
                    new_f = []
                if newaxon_f: # trailing axons
                    self.total_fronts += len(newaxon_f)
                    self._write_new_DB(newaxon_f,trail_axon=True)
                    newaxon_f = []
                if new_syn_id:
                    self._write_syn_DB(new_syn_id)
                    new_syn_id = []
                if deleted_f: # declare death
                    self._DB_del_update(deleted_f)
                    deleted_f = []
                if del_syn_id:
                    self._del_syn_DB(del_syn_id)
                    del_syn_id = []
                if migrated_f: # update migration status
                    self._new_mig_DB(migrated_f)
                    migrated_f = []
            if any_f_moved: # add row to migration table
                self._write_mig_DB()
            # signal to slave1 that branch retractions can be done
            #   check whether slave1 has started
            if self.verbose >= 6:
                print ("Admin sending start retractions to Slave1",len(retracted_f))
            if retracted_f:
                while self._proc_status[1] == 10: # slave1 not started yet
                    time.sleep(PAUSE)
                self._proc_status[1] = 15
            # check that all processes have reached end of cycle status
            for n in range(20000):
                not_finish = 0
                for pid in range(1,max_procs):
                    if self._proc_status[pid] != 20:
                        not_finish = pid
                if not_finish == 0:
                    break # out of for n
                elif (n > 5000) and (not_finish == 1): # waiting for slave to complete retractions
                    time.sleep(PAUSE)                
            if not_finish > 0:
                raise BugError("_simulation_loop2p","process " + str(not_finish) + " does not reach end of cycle " + str(self._proc_status[not_finish]) + " " + str(max_procs))
            self._proc_status[1] = 25 # no need for lock brokering anymore
            num_f_out = self._proc_status[0] # get number fronts next cycle
            if retracted_f: # get deleted fronts
                n = 0 # index into self._new_AIDs
                deleted_f = []
                del_syn_id = []
                num_dels = self._new_AIDs[0].index # number of deleted fronts
                for n in range(num_dels):
                    id = self._new_AIDs[n].ID
                    self._new_AIDs[n].status = b'0'
                    front = self._fronts[id._nid][id._fid]
                    if front.has_synapse(): # remove the synapse
                        del_syn_id.append(abs(front._yid))
                        front.remove_synapse(self.constellation)
                    deleted_f.append(front)
                    self._remove_plot(front)
                self._DB_del_update(deleted_f)
                if del_syn_id:
                    self._del_syn_DB(del_syn_id)
            if self._store_fronts:
                self._store_attribs(check=True)
            self._conn.commit() # save changes from this cycle
            self._fig.canvas.draw()
        self._clock.value = -1 # end of simulation loop: go into winter sleep
        self._proc_status[1] == 30

    # run a number of cycles in simulation doing most Admin duties except lock brokering
    def _simulation_loop3(self,num_cycles):
        # make sure processes are online
        gids = [] # not used in single process run but referenced in print
        if self._first_cycle: 
            for pid in range(2,self._num_procsP2):
                first = True
                while self._proc_status[pid] == 0:
                    if first and self.verbose >= 6:
                        print ("Admin waiting for Process",pid,"to activate")
                        first = False
                    time.sleep(PAUSE)
        self._first_cycle = False
        for nc in range(num_cycles):
            self.cycle += 1
            if self.verbose > 0:
                print (Fore.BLUE + "NeuroDevSim admin starting cycle",self.cycle,Fore.RESET)
            if self.cycle in self._future_active:
                for item in self._future_active[self.cycle]:
                    status = item[0]
                    f = item[1]
                    f._set_active() # always make active
                    if status == b'i': # make active only
                        if f not in self._active_fronts:
                            self._active_fronts.append(f)
                    elif status == b'g': # make growing
                        f.set_growing()
                        if f not in self._growing_fronts:
                            self._growing_fronts.append(f)
                    elif status == b'm': # make migrating
                        f.set_migrating()
                        if f not in self._migrating_fronts:
                            self._migrating_fronts.append(f)
                    else:
                        raise BugError("_simulation_loop3","unknown status " + str(status))
                    if f._does_storing() and f not in self._store_fronts:
                        self._store_fronts.append(f)
                del self._future_active[self.cycle]
            new_f = [] # list of new fronts for database
            newaxon_f = [] # list of new trailing axons for database
            migrated_f = [] # list of new migrated fronts for database
            retracted_f = [] # list of roots of branches to retract
            new_syn_id = [] # list of new synapses for database
            del_syn_id = [] # list of deleted synapses fronts for database
            any_f_moved = False # check whether any migration happened this cycle
            deleted_f = [] # list of all fronts that were deleted
            new_inactive = [] # list of fronts that become inactive next cycle
            next_growing = [] # self._growing_fronts for next cycle
            next_migrating = [] # self._migrating_fronts for next cycle
            next_active = [] # self._active_fronts for next cycle
            num_f_in = num_f_out = len(self._growing_fronts) + \
                                   len(self._migrating_fronts) + \
                                   len(self._active_fronts)
            if num_f_in == 0:
                if self.verbose > 0:
                    print (Fore.BLUE + "NeuroDevSim admin ending: no active fronts on cycle",self.cycle,Fore.RESET)        
                return
            elif num_f_in < self._num_procs:
                max_procs = 2 + num_f_in
                stop_procs = True
            else:
                max_procs = self._num_procsP2
                stop_procs = False
            # clear index of first entry of self._new_AIDs and remove negat_ID in Processes
            for pid in range(2,self._num_procsP2):
                self._actives[pid] = empty_ID
                self._new_AIDs[self._n_starts[pid]].index = 0
            start_AIDs = [0] * self._num_procsP2 # relative start of process block
            self._clock.value = self.cycle # start processes
            if self.verbose >= 6:
                print ("Admin sending",num_f_in,"fronts",max_procs,stop_procs)
            start = time.time()
            # keep cycling till all fronts are processed
            # current balancing approach: tightly locked -> cores will have to
            #   wait for next front
            if self._num_procs > 1:
                lock_gids_th = self._num_procsP2 # lock gids if num_f_in > lock_gids_th
            else: # don't bother locking on single processor run
                lock_gids_th = num_f_in # never lock
            # control pre-fetching of fronts: not at end because some
            #   processes may take a lot of time and cannot execute the 
            #   prefetched front.
            reserve_th = self._num_procsP2 # get next front ready if num_f_in > lock_gids_th
            grow_index0 = 0 # start from beginning of list
            migr_index0 = 0 # start from beginning of list
            if self._debug: # keep track of ids sent and received
                target = num_f_in
                sent_id = []
                receiv_id = []
            n_fails = 0 # number of failed fetch loops since last success
            if self.verbose >= 6:
                last_fid = 0 # used to prevent endless repetition of same message
            while num_f_out: 
                # find pid needing a front: at present on demand only
                for pid in range(2,max_procs):
                    ns0 = self._n_starts[pid] + start_AIDs[pid]
                    aid = self._new_AIDs[ns0]
                    ns1 = aid.index
                    while ns1 > 0: # possibly data for multiple parent fronts
                        num_f_out -= 1 # finished one
                        # process front that was called
                        id = aid.ID
                        if self._debug: # keep track of ids sent and received
                            receiv_id.append(id)
                        if self.verbose > 1:
                            print ("Admin received from",pid,id,ns0,ns1,num_f_in,num_f_out,n_fails)
                            #print ("   ",pid,ns,start_AIDs[pid],ns0,ns1)
                        front = self._fronts[id._nid][id._fid]
                        if front._gid < 0: # locked gids
                            # unlocked by process
                            front._gid = -front._gid # set back to unlocked
                            # reset to start of lists as gid has been freed
                            grow_index0 = 0 # start from beginning of list
                            migr_index0 = 0 # start from beginning of list
                        # store for next cycle if active, automatically in order called now
                        if front.is_migrating():
                            next_migrating.append(front)
                            if front not in self._mig_fronts:
                                migrated_f.append(front)
                            if front.has_moved():
                                any_f_moved = True
                            if front._does_storing() and \
                                                front not in self._store_fronts:
                                self._store_fronts.append(front)
                        elif front.is_growing():
                            next_growing.append(front)
                            if front._does_storing() and \
                                                front not in self._store_fronts:
                                self._store_fronts.append(front)
                        elif front.is_active():
                            next_active.append(front)
                            if front._does_storing() and \
                                                front not in self._store_fronts:
                                self._store_fronts.append(front)
                        elif (front in self._store_fronts) and (front.order > 0):
                            self._store_fronts.remove(front) # not active anymore
                        # process other new_AIDs
                        for n in range(ns0 + 1, self._n_starts[pid] + ns1):
                            status = self._new_AIDs[n].status
                            id = self._new_AIDs[n].ID
                            front = self._fronts[id._nid][id._fid]
                            if status == b'e':
                                new_f.append(front)
                                if front.is_growing():
                                    next_growing.append(front)
                                    if front._does_storing() and \
                                                front not in self._store_fronts:
                                        self._store_fronts.append(front)
                                elif front.is_active():
                                    next_active.append(front)
                                    if front._does_storing() and \
                                                front not in self._store_fronts:
                                        self._store_fronts.append(front)
                            elif status == b't': # a trailing axon was created
                                newaxon_f.append(front)
                                if front.is_active():
                                    next_active.append(front)
                                    if front._does_storing() and \
                                                front not in self._store_fronts:
                                        self._store_fronts.append(front)
                            elif status == b'r': # retracted front
                                deleted_f.append(front)
                                new_inactive.append(front) # sometimes kept active
                            elif status == b'a': # enable front
                                if front.is_migrating():
                                    next_migrating.append(front)
                                    if front not in self._mig_fronts:
                                        migrated_f.append(front)
                                    if front._does_storing() and \
                                                front not in self._store_fronts:
                                        self._store_fronts.append(front)
                                elif front.is_growing():
                                    next_growing.append(front)
                                    if front._does_storing() and \
                                                front not in self._store_fronts:
                                        self._store_fronts.append(front)
                                elif front.is_active():
                                    next_active.append(front)
                                    if front._does_storing() and \
                                                front not in self._store_fronts:
                                        self._store_fronts.append(front)
                            elif (status == b'i') or (status == b'g') or\
                                        (status == b'm'): # disable front
                                front = self._fronts[id._nid][id._fid]
                                # can be added to next_active later
                                new_inactive.append(front)
                                till_cycle = self._new_AIDs[n].index
                                if till_cycle > self.cycle:
                                    # future activation at till_cycle requested
                                    if till_cycle in self._future_active:
                                        self._future_active[till_cycle].\
                                                    append([status,front])
                                    else:
                                        self._future_active[till_cycle] = \
                                                    [[status,front]]
                            elif status == b'd': # retract branch
                                retracted_f.append(front)
                            elif status == b's': # store substrate
                                ind = id._fid
                                self.add_substrate(self._substrate[ind],index=ind)
                            elif status == b'y': # store synapse
                                new_syn_id.append(id._fid)
                            elif status == b'z': # remove synapse
                                del_syn_id.append(id._fid)
                            else:
                                raise BugError("_simulation_loop","on " + str(pid) + " unknown self._new_AIDs status " + str(self._new_AIDs[n]))
                        start_AIDs[pid] = ns1 # update for next round
                        ns0 = self._n_starts[pid] + ns1
                        # check whether more data ready
                        aid = self._new_AIDs[ns0]
                        ns1 = aid.index
                    # Deal with end of cycle
                    if num_f_in == 0: # no more fronts to send
                        # check that both fetch and pre-fetch are done or in process
                        if (self._actives[pid]._nid == 0) and \
                            (self._actives[pid + self._num_procs]._nid == 0):
                                self._actives[pid] = negat_ID # signal end of cycle
                        continue
                    # Check whether we should fetch or pre-fetch
                    if (self._actives[pid]._nid > 0) and (num_f_in <= reserve_th):
                        # previous fetch unused and no more pre-fetching
                        #   (works also for negat_ID in self._actives[pid])
                        continue
                    # check whether pre-fetch necessary
                    if self._actives[pid + self._num_procs]._nid > 0: 
                        continue # already done prefetch
                    # pid has no instructions
                    found = False # found a front for pid
                    # decide whether we should check for gid competition
                    if n_fails <= self._num_procs: # not too many fails
                        lock_gids = num_f_in > lock_gids_th
                    else:
                        lock_gids = False 
                        grow_index0 = 0 # start from beginning of list
                        migr_index0 = 0 # start from beginning of list
                    # do growing fronts first
                    for n in range(grow_index0,len(self._growing_fronts)):
                        front = self._growing_fronts[n]
                        if self.verbose > 3:
                            print ("Admin trying grow",pid,front)
                        if lock_gids: # check whether free
                            gids = []
                            # check gid availibility: bypass _get_gids for speed
                            ngid = front._gid
                            failed = False
                            for i in range(ngid + 1, ngid + 1 + self._gids[ngid]):
                                gid = self._gids[i]
                                if self._grid_wlock[gid] != 0:
                                    if (self.verbose >= 6) and (front._fid != last_fid):
                                        last_fid = front._fid # do not repeat this one
                                        print ("Admin grow failed",pid,num_f_in,gid,self._grid_wlock[gid],front)
                                    grow_index0 = n # skip this one next pid
                                    failed = True # already locked
                                    break
                                else: # request lock: this admin is not brokering
                                    gids.append(gid)
                                    wait = 0.
                                    self._gwlock_request[1] = gid
                                    while self._grid_wlock[gid] != 1:
                                        time.sleep(LOCKPAUSE)
                                        wait += LOCKPAUSE
                                        if wait > 0.001:
                                            if self.verbose >= 6:
                                                print ("Admin grow failed to acquire lock",pid,gid)
                                            failed = True
                                            break
                                    if failed:
                                        break
                            if failed:
                                # unlock gids
                                for gid in gids:
                                    if self._grid_wlock[gid] == 1:
                                        self._grid_wlock[gid] = 0
                                continue # for front in self._growing_fronts
                        # use this front
                        if self._actives[pid]._nid > 0: # save in reserve
                            if self.verbose >= 6:
                                print ("Admin sent growing reserve to",pid,front.get_id(),gids)
                            self._actives[pid + self._num_procs] = front.get_id()
                        else: # save in regular place
                            if self.verbose >= 6:
                                print ("Admin sent growing to",pid,front.get_id(),gids)
                            self._actives[pid] = front.get_id()
                        if self._debug: # keep track of ids sent and received
                            sent_id.append(front.get_id())
                        num_f_in -= 1 # started one
                        found = True
                        if lock_gids: # now really lock them
                            for gid in gids: # lock gids for admin
                                self._grid_wlock[gid] = pid # lock to process instead of admin
                            front._gid = -ngid # mark as locked
                        del self._growing_fronts[n] # done this one
                        break
                    if found:
                        n_fails = 0 # reset
                        continue # for pid in range
                    # do migrating fronts next
                    for n in range(migr_index0,len(self._migrating_fronts)):
                        front = self._migrating_fronts[n]
                        if self.verbose >= 6:
                            print ("Admin trying mig",pid,front)
                        if lock_gids: # check whether free
                            gids = []
                            # check gid availibility: bypass _get_gids for speed
                            ngid = front._gid
                            failed = False
                            for i in range(ngid + 1, ngid + 1 + self._gids[ngid]):
                                gid = self._gids[i]
                                if self._grid_wlock[gid] != 0:
                                    if (self.verbose >= 6) and (front._fid != last_fid):
                                        last_fid = front._fid # do not repeat this one
                                        print ("Admin mig failed",pid,num_f_in,gid,self._grid_wlock[gid],front)
                                    migr_index0 = n # skip this one next pid
                                    failed = True # already locked
                                    break
                                else: # request lock: this admin is not brokering
                                    gids.append(gid)
                                    wait = 0.
                                    self._gwlock_request[1] = gid
                                    while self._grid_wlock[gid] != 1:
                                        time.sleep(LOCKPAUSE)
                                        wait += LOCKPAUSE
                                        if wait > 0.001:
                                            if self.verbose >= 6:
                                                print ("Admin mig failed to acquire lock",pid,gid)
                                            failed = True
                                            break
                                    if failed:
                                        break
                            if failed:
                                # unlock gids
                                for gid in gids:
                                    if self._grid_wlock[gid] == 1:
                                        self._grid_wlock[gid] = 0
                                continue # for front in self._growing_fronts
                        # use this front
                        if self._actives[pid]._nid > 0: # save in reserve
                            if self.verbose >= 6:
                                print ("Admin sent migrating reserve to",pid,front.get_id(),gids)
                            self._actives[pid + self._num_procs] = front.get_id()
                        else: # save in regular place
                            if self.verbose >= 6:
                                print ("Admin sent migrating to",pid,front.get_id(),gids)
                            self._actives[pid] = front.get_id()
                        if self._debug: # keep track of ids sent and received
                            sent_id.append(front.get_id())
                        num_f_in -= 1 # started one
                        found = True
                        if lock_gids: # now really lock them
                            for gid in gids: # lock gids for admin
                                self._grid_wlock[gid] = pid # lock to process instead of admin
                            front._gid = -ngid # mark as locked
                        del self._migrating_fronts[n] # done this one
                        break
                    if found:
                        n_fails = 0 # reset
                        continue # for pid in range                            
                    # do active fronts last
                    if self._active_fronts:
                        if self.verbose >= 6:
                            print ("Admin trying act",pid,front)
                        # use this front
                        fid = self._active_fronts.pop(0).get_id()
                        if self._actives[pid]._nid > 0: # save in reserve
                            self._actives[pid + self._num_procs] = fid
                            if self.verbose >= 6:
                                print ("Admin sent active reserve to",pid,self._actives[pid + self._num_procs])
                        else: # save in regular place
                            self._actives[pid] = fid
                            if self.verbose >= 6:
                                print ("Admin sent active to",pid,self._actives[pid])
                        if self._debug: # keep track of ids sent and received
                            sent_id.append(fid)
                        num_f_in -= 1 # started one
                        n_fails = 0 # reset
                        continue # for pid in range
                    # failed to find one
                    if self._actives[pid]._nid == 0: # failure of a fetch cycle
                        n_fails += 1
                if stop_procs: # less active fronts than processes
                    for pid in range(max_procs,self._num_procsP2):
                       self._actives[pid] = negat_ID # signal end of cycle
                    stop_procs = False # do this only once 
            # do branch retractions
            if retracted_f:
                r_del = self._retract_branches(retracted_f,new_inactive)
                deleted_f += r_del
            # save new/changed fronts from in database
            if new_f:
                self.total_fronts += len(new_f)
                self._write_new_DB(new_f)
            if migrated_f:
                self._new_mig_DB(migrated_f)
            if any_f_moved:
                self._write_mig_DB()
            if newaxon_f:
                self.total_fronts += len(newaxon_f)
                self._write_new_DB(newaxon_f,trail_axon=True)
            if new_syn_id:
                self._write_syn_DB(new_syn_id)
            if deleted_f:
                self._DB_del_update(deleted_f)
            if del_syn_id:
                self._del_syn_DB(del_syn_id)
            if self._store_fronts:
                self._store_attribs()
            self._conn.commit()
            # reset self._growing_fronts, self._migrating_fronts, self._active_fronts
            for front in new_inactive: # make inactive and remove from lists
                if front.is_growing():
                    front.clear_growing()
                    if front in next_growing:
                        next_growing.remove(front)
                if front.is_migrating():
                    front.clear_migrating()
                    if front in next_migrating:
                        next_migrating.remove(front)
                front._clear_active()
                if front in next_active:
                    next_active.remove(front)
                if front in self._store_fronts:
                    self._store_fronts.remove(front)
            self._growing_fronts = next_growing
            self._migrating_fronts = next_migrating
            self._active_fronts = next_active
            # check that all processes have reached end of cycle status
            for n in range(10000):
                not_finish = 0
                for pid in range(2,max_procs):
                    if self._proc_status[pid] != 20:
                        not_finish = pid
                if not_finish == 0:
                    break # out of for n
            if not_finish > 0:
                raise BugError("_simulation_loop","process " + str(not_finish) + " does not reach end of cycle")
        self._clock.value = -1 # end of simulation loop: go into winter sleep
            
    # retract branches and store in database
    # admin can do this very fast and effectively as it does not have to 
    #   care about removing children first
    # can only be called by process that runs _lock_broker or after _lock_broker
    #   stops running for the cycle
    # returns list of deleted fronts or front IDs depending on returnID
    def _retract_branches(self,retracted_f,new_inactive,returnID=False):
        deleted_f = []
        if self.verbose >= 7:
            print ("Admin _retract_branches",self.my_id,len(retracted_f))
        del_syn = [] # deleted synapses
        for front in retracted_f:
            if self.verbose >= 7:
                print ("Admin _retract_branches",self.cycle,"front",front)
            # only for root remove as a child
            parent = front.get_parent(self.constellation)
            if parent:
                self.constellation._delete_child(parent,front)
            # make list of to be retracted fronts
            retr_fronts = [front]
            front._add_children(self.constellation,retr_fronts,False)
            # now retract them all
            for f in retr_fronts:
                new_inactive.append(f)
                f._set_retracted()
                if f.death > 0:
                    continue  # already deleted
                if self.verbose >= 7:
                    print ("Admin deleting",f)
                # do basic retraction settings
                f.death = self.cycle
                if self.my_id == 1: # only done by real admin
                    if f.has_synapse(): # remove also the synapse
                        del_syn.append(abs(f._yid))
                        f.remove_synapse(self.constellation)
                id = f.get_id()
                if returnID:
                    deleted_f.append(id)
                else:
                    deleted_f.append(f)
                # remove from grid
                gids = f._get_gids(self.constellation)
                while len(gids) > 0: # can still overlap with process requests
                    for gid in gids: # bypass lock_broker
                        self._lock_broker()
                        if (self._grid_wlock[gid] == 0) and \
                                                (self._grid_rlock[gid] == 0):
                            self._grid_wlock[gid] = self.my_id
                            self.constellation._grid_remove(gid,id)
                            self._grid_wlock[gid] = 0
                            gids.remove(gid)
                            break
        if del_syn:
            self._del_syn_DB(del_syn)
        return deleted_f
        
    # plot a migrating soma
    def _plot_migration(self,front):
        key = front._key()
        if key in self._lines:
            prev = self._lines[key]
        else:
            prev = None
        if (self._color_scheme == 3) or self._soma_black:
            col = 'k'
        else:
            col = self._c_mapping[front.get_neuron_name(self.constellation)]
        x = front.orig.x + (front.radius * self._x_outer)
        y = front.orig.y + (front.radius * self._y_outer)
        z = front.orig.z + (front.radius * self._z_outer)
        self._lines[front._key()] = self._ax.plot_surface(x,y,z,rstride=2,cstride=2,\
                                color=col,linewidth=0,antialiased=False,\
                                shade=False)
        if prev:
            try:
                prev.remove()
            except:
                if self.verbose >= 7:
                    print ("Error: _plot_migration did not find surface for",fid)
    
    # plot a new front
    def _plot_front(self,front):
        # do front storage here
        if front.is_growing() or front.is_active():
            if front._does_storing() and front not in self._store_fronts:
                self._store_fronts.append(front)        
        if self._box: # do not plot if completely outside box
            if (front.orig.x<self._box[0][0]) and (front.end.x<self._box[0][0]):
                return # do not plot
            if (front.orig.y<self._box[0][1]) and (front.end.y<self._box[0][1]):
                return # do not plot
            if (front.orig.z<self._box[0][2]) and (front.end.z<self._box[0][2]):
                return # do not plot
            if (front.orig.x>self._box[1][0]) and (front.end.x>self._box[1][0]):
                return # do not plot
            if (front.orig.y>self._box[1][1]) and (front.end.y>self._box[1][1]):
                return # do not plot
            if (front.orig.z>self._box[1][2]) and (front.end.z>self._box[1][2]):
                return # do not plot
        scheme3 = False
        if (self._color_scheme == 2):
            name = front.get_branch_name()
            if not (name in self._c_mapping):
                self._c_mapping[name] = self._colors[self._color_index%len(self._colors)]
                self._color_index = self._color_index + 1
            col = self._c_mapping[name]
        elif self._color_scheme == 3:
            col = 'k' # will be changed
            try:
                value = getattr(front,self._color_attrib)
                scheme3 = True
                col2 = (value - self._color_min) * self._color_scale
                self._color_fronts.append([front,value])
            except:
                pass
                #raise ValueError("color_data",color_attrib + "is not an attribute of" + front)
        else:
            col = self._c_mapping[front.get_neuron_name(self.constellation)]
        key = front._key()
        if front.is_cylinder(): # cylinder -> line
            self._lines[key] = self._ax.plot([front.orig.x,front.end.x],\
                                        [front.orig.y,front.end.y],\
                                        [front.orig.z,front.end.z],\
                                        linewidth=self._radius_scale*front.radius,\
                                        color=col)
            if scheme3:
                self._lines[key][0].set_color(self._colors(col2))
        else: # sphere -> circle
            scale = self._sphere_scale * front.radius
            x = front.orig.x + (scale * self._x_outer)
            y = front.orig.y + (scale * self._y_outer)
            z = front.orig.z + (scale * self._z_outer)
            # optionally color soma black
            if (self._color_scheme < 3) and (front._pid == -1) \
                    and self._soma_black:
                col = 'k'
            self._lines[key] = self._ax.plot_surface(x,y,z,rstride=2,cstride=2,\
                                    color=col,linewidth=0,antialiased=False,\
                                    shade=False)
            if scheme3:
                self._lines[key].set_color(self._colors(col2))
        if not self.constellation._automatic and self._box: # store plot items
            self._plot_items.append(front)

    # remove lines from plot for a deleted front
    def _remove_plot(self,front):
        try:
            if front.is_cylinder():
                self._ax.lines.remove(self._lines[front._key()][0])
            else:
                self._lines[front._key()].remove()
        except Exception as error:
            if self.verbose >= 7:
                print (error)

    ### database ###
    # returns connection or None
    def _setup_DB(self,db_file_name):
        if not db_file_name.endswith(".db"):
            db_file_name = db_file_name + '.db'
        if self.verbose >= 5:
            print ("Admin creating database",db_file_name)
        try:
            os.remove(db_file_name)
        except Exception:
            pass
        try:
            conn = sqlite3.connect(db_file_name)
        except Exception:
            raise TypeError(str(db_file_name) + ": file cannot be opened")
        self.db_name = db_file_name # store actual name
        # Create table with info about the simulation volume, this is needed for the
        #  plotting and movie scripts
        conn.execute("CREATE TABLE neurodevsim (id INTEGER PRIMARY KEY AUTOINCREMENT,\
                                            xmin real,\
                                            ymin real,\
                                            zmin real,\
                                            xmax real,\
                                            ymax real,\
                                            zmax real,\
                                            num_cycles int,\
                                            num_procs int,\
                                            version real,\
                                            run_time real,\
                                            importable int,\
                                            substrate int,\
                                            migration int,\
                                            synapses int,\
                                            attributes int,\
                                            arcs int)")
                
        # Create table with neuron types
        conn.execute("CREATE TABLE neuron_types (id INTEGER PRIMARY KEY AUTOINCREMENT,\
                                            type_id int,\
                                            neuron_type text)")

        # Create table with neuron names
        conn.execute("CREATE TABLE neuron_data (id INTEGER PRIMARY KEY AUTOINCREMENT,\
                                            neuron_id int,\
                                            type_id int,\
                                            name text,\
                                            firing_rate real,\
                                            CV_ISI real,\
                                            num_fronts int,\
                                            num_retracted int,\
                                            num_synapses int)")

        # Create table with data about all the fronts created
        # Though we want to use front.id as the primary key sqlite is not happy if it cannot
        #  make rowid
        #   birth is self.cycle value when front created
        #   death is set to -1 and may become counter value when retracted
        #   migration is set to 0 and may become a migration_data column number
        conn.execute("CREATE TABLE front_data (id INTEGER PRIMARY KEY AUTOINCREMENT,\
                                            neuron_id int,\
                                            front_id int,\
                                            branch text,\
                                            swc_type int,\
                                            shape int, \
                                            orig_x real,\
                                            orig_y real,\
                                            orig_z real,\
                                            end_x real,\
                                            end_y real,\
                                            end_z real,\
                                            radius real,\
                                            parent_id int,\
                                            b_order int,\
                                            path_len real,\
                                            birth int,\
                                            death int,\
                                            migration int,\
                                            flags int)")
        self.front_data_id = 1 # counter corresponding to front_data table id
        cursor = conn.cursor()
        # Save the volume boundaries and boundaries in database
        # initialize for no substrate or migration tables
        sim_volume = self.constellation.sim_volume
        values = (None,sim_volume[0][0],sim_volume[0][1],sim_volume[0][2],\
                  sim_volume[1][0],sim_volume[1][1],sim_volume[1][2],0,self._num_procs,nds_version(raw=True),0.,0,0,0,0,0,0)
        cursor.execute("INSERT into neurodevsim VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",values)
        # Fill neuron_types table
        tvalues = []
        nid = 1
        for t in self._neuron_types:
            tvalues.append((None,nid,t))
            nid += 1
        cursor.executemany("INSERT into neuron_types VALUES (?,?,?)",tvalues)
        # all standard table names: cannot be overwritten
        self._db_tables = ["neurodevsim","neuron_types","neuron_data",\
                           "front_data","attributes","substrate_data",\
                           "migration_data1","synapse_data","arc_data",\
                           "arc_points"]
        return conn

    ### write all new fronts to database, can be a trailing axon
    def _write_new_DB(self,new_f,trail_axon=False):
        if self.verbose >= 7:
            print ("Admin writing to front_data")
        nvalues = [] # for insert into neuron_data
        fvalues = [] # for insert into front_data
        uvalues = [] # for updates of trailing axon parent
        for front in new_f:
            if front.order == 0: # a new soma
                neuron_id = front._sid
                neuron = self._neurons[neuron_id]
                name = front.get_neuron_name(self.constellation)
                nvalues.append((None,neuron_id,front._nid,name,\
                                neuron.firing_rate,neuron.CV_ISI,\
                                1,0,0))
                self._names_in_DB[name] = neuron_id
            else:
                neuron_id = self._fronts[front._nid][front._sid]._sid
            # update neuron counters
            self._neurons[neuron_id].num_fronts += 1
            orig = front.orig
            end = front.end
            if front.is_cylinder():
                shape = 2
            else:
                shape = 1
            fvalues.append((None,neuron_id,front._fid,\
                          front.get_branch_name(),front.swc_type,shape,\
                          orig.x,orig.y,orig.z,end.x,end.y,end.z,\
                          front.radius,front._pid,front.order,\
                          front.path_length,front.birth,front.death,0,\
                          front._flags))
            front._dbid = self.front_data_id # store database id
            self.front_data_id += 1 # increment for next front
            if trail_axon: # make this parent of previous axon front
                uvalues.append((front._fid,front._dbid))
        if nvalues:
            self._conn.cursor().executemany("INSERT into neuron_data VALUES (?,?,?,?,?,?,?,?,?)",nvalues)
        self._conn.cursor().executemany("INSERT into front_data VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",fvalues)
        if uvalues:
            self._conn.cursor().executemany("UPDATE front_data SET parent_id = ? WHERE id = ?",uvalues)

    ### change an entry in the front_data database for deleted fronts
    def _DB_del_update(self,deleted_f):
        if self.verbose >= 7:
            print ("Admin updating front_data for deletion")
        fvalues = [] # for update of front_data
        for front in deleted_f:
            if self.verbose >= 7:
                print ("Admin _del_update",front.get_id(),front._dbid)
            neuron_id = self._fronts[front._nid][front._sid]._sid
            # update neuron counters
            neuron = self._neurons[neuron_id]
            neuron.num_fronts -= 1
            neuron.num_retracted += 1
            fvalues.append((self.cycle,front._dbid))
        self._conn.cursor().executemany("UPDATE front_data SET death = ? WHERE id = ?",fvalues)

    ### create extra table listing attributes table info
    def _attr_DB(self):
        self._conn.execute("CREATE TABLE attributes (id INTEGER PRIMARY KEY AUTOINCREMENT,\
                                            name text,\
                                            type int,\
                                            neuron_name text,\
                                            last_only int)")

    ### create extra table for substrate data
    def _sub_DB(self):
        self._conn.execute("CREATE TABLE substrate_data (id INTEGER PRIMARY KEY AUTOINCREMENT,\
                                            name text,\
                                            x real,\
                                            y real,\
                                            z real,\
                                            amount real,\
                                            rate real,\
                                            diff_c real,\
                                            birth int,\
                                            death int)")
        # register existence of this table
        self._conn.cursor().execute("UPDATE neurodevsim SET substrate = ? WHERE id = 1",(1,))
        self._sub_data = True
        
    ### create extra tables for arc data
    def _arcs_DB(self):
        self._conn.execute("CREATE TABLE arc_data (id INTEGER PRIMARY KEY AUTOINCREMENT,\
                                            arc_index int,\
                                            cycle int,\
                                            complete int,\
                                            neuron_id int,\
                                            caller_fid int,\
                                            last_fid int,\
                                            sphere_0 int,\
                                            sphere_1 int,\
                                            angle real,\
                                            num_points int,\
                                            count int,\
                                            next_point int)")
        self._conn.execute("CREATE TABLE arc_points (id INTEGER PRIMARY KEY AUTOINCREMENT,\
                                            arc_index int,\
                                            p_x real,\
                                            p_y real,\
                                            p_z real)")
        # register existence of this table
        self._conn.cursor().execute("UPDATE neurodevsim SET arcs = ? WHERE id = 1",(2,))
                                            
    ### create extra table for migration data
    ###   as maximum number of columns is 2000, we may need to make multiple
    ###   such tables
    def _mig_DB(self,number,key):
        if self.verbose >= 7:
            print ("Admin making migration_data table")
        key_x = "x_" + key
        key_y = "y_" + key
        key_z = "z_" + key
        query = "CREATE TABLE migration_data" + str(number) + \
                " (id INTEGER PRIMARY KEY AUTOINCREMENT,cycle int," + \
                    key_x + " real," + key_y + " real," + key_z + " real)"
        self._conn.execute(query)
        # register existence of this table
        self._conn.cursor().execute("UPDATE neurodevsim SET migration = ? WHERE id = 1",(number,))
        self._mig_tables = number
        # make new list entry: ",?,?,?" will be added for each extra column
        query = "INSERT into migration_data" + str(number) + " VALUES (?,?,?,?,?"
        self._mig_inserts.append(query)
        # self.mig_col will keep increasing, it is not reset for later tables
        if number == 1:
            self.mig_col = 2 # python number of first _x column in migration_data
        self._mig_data += 1

    ### add extra columns to table for migration data
    def _add_mig_DB(self,key):
        if self.verbose >= 7:
            print ("Admin updating migration_data table")
        if self._mig_data == 0:
            raise BugError("_add_mig_DB","attempt to change non existing migration_data table")
        ntable = (self._mig_data // 600) + 1 # max number of columns is 2000
        if ntable > self._mig_tables: # need to create an extra table
            self._mig_DB(ntable,key) # this also adds the 3 columns
        else:
            qstr = "ALTER TABLE migration_data" + str(ntable) + " ADD COLUMN "
            query = qstr + "x_" + key + " real"
            self._conn.execute(query)
            query = qstr + "y_" + key + " real"
            self._conn.execute(query)
            query = qstr + "z_" + key + " real"
            self._conn.execute(query)
            self._mig_inserts[ntable - 1] += ",?,?,?"
            self._mig_data += 1
        self.mig_col += 3

    ### update list of migrating fronts used in database updates and
    #   enter migration column in front_data
    #   migrated_f: front.is_migrating()==True that are not yet in self._mig_fronts
    def _new_mig_DB(self,migrated_f):
        if self.verbose >= 7:
            print ("Admin marking migrating neurons",nds_list(migrated_f))
        values = [] # database updates
        for front in migrated_f:
            self._mig_fronts.append(front)
            key = front._key()
            if self._mig_tables == 0: # make migration_data table in the database
                self._mig_DB(1,key)
            else: # add extra columns to migration_data table
                self._add_mig_DB(key)
            if front.order > 0:
                raise BugError("_new_mig_DB","migrating front not a soma " + str(front))
            values.append((self.mig_col,front._dbid))
        if values:
            self._conn.cursor().executemany("UPDATE front_data SET migration = ? WHERE id = ?",values)

    ### write all new positions of all moved fronts to database
    def _write_mig_DB(self):
        if self.verbose >= 7:
            print ("Admin updating migration_data")
        values = [None,self.cycle]
        ctable = 1 # table to use
        one_moved = False
        n = 0
        for front in self._mig_fronts:
            ntable = (n // 600) + 1
            if ntable > ctable:
                if one_moved:
                    self._conn.cursor().execute(self._mig_inserts[ctable - 1] + ")",values)
                values = [None,self.cycle]
                ctable = ntable
                one_moved = False
            if front._has_moved_now(): # still migrating
                values.extend([front.orig.x,front.orig.y,front.orig.z])
                one_moved = True
                front._clear_moved_now()
            else:
                values.extend([None,None,None])
            n += 1
        if one_moved:
            self._conn.cursor().execute(self._mig_inserts[ctable - 1] + ")",values)
            self._conn.commit()

    # create table that contains correct order of self._mig_fronts for import_simulation
    def _mig_fronts_DB(self):
        self._conn.execute("CREATE TABLE mig_fronts_data (id INTEGER PRIMARY KEY AUTOINCREMENT,\
                                            type_id int,\
                                            front_id int)")
    
    ### create extra table for synapse data
    def _syn_DB(self):
        self._conn.execute("CREATE TABLE synapse_data (id INTEGER PRIMARY KEY AUTOINCREMENT,\
                                            pre_neuron_id int,\
                                            pre_front_id int,\
                                            post_neuron_id int,\
                                            post_front_id int,\
                                            weight real,\
                                            birth int,\
                                            death int)")
        # register existence of this table
        self._conn.cursor().execute("UPDATE neurodevsim SET synapses = ? WHERE id = 1",(1,))
        self._syn_data_id = 1 # counter corresponding to synapse_data table id
        self._syn_data = True

    # store synapse data in database and update Neuron counters
    def _write_syn_DB(self,synapse_ids):
        if not self._syn_data:
            self._syn_DB()
        values = [] # database updates
        for id in synapse_ids:
            syn = self._synapses[id]
            if self.verbose >= 7:
                print ("Admin storing synapse",id,syn)
            pre = self.constellation.front_by_id(syn.pre_syn)
            pre_id = self._fronts[pre._nid][pre._sid]._sid
            post = self.constellation.front_by_id(syn.post_syn)
            post_id = self._fronts[post._nid][post._sid]._sid
            values.append((None,pre_id,pre._fid,post_id,post._fid,syn.weight,\
                            self.cycle,-1))
            syn._dbid = self._syn_data_id # store database id
            self._syn_data_id += 1 # increment for next synapse
            # update neuron counters
            self._neurons[pre_id].num_synapses += 1
            self._neurons[post_id].num_synapses += 1
        if values:
            self._conn.cursor().executemany("INSERT into synapse_data VALUES (?,?,?,?,?,?,?,?)",values)

    # store synapse data in database and update Neuron counters
    def _del_syn_DB(self,synapse_ids):
        values = [] # database updates
        for id in synapse_ids:
            syn = self._synapses[id]
            pre = self.constellation.front_by_id(syn.pre_syn)
            pre_id = self._fronts[pre._nid][pre._sid]._sid
            post = self.constellation.front_by_id(syn.post_syn)
            post_id = self._fronts[post._nid][post._sid]._sid
            if self.verbose >= 7:
                print ("Admin storing synapse",id,syn)
            values.append((self.cycle,syn._dbid))
            # update neuron counters
            self._neurons[pre_id].num_synapses -= 1
            self._neurons[post_id].num_synapses -= 1
        if values:
            self._conn.cursor().executemany("UPDATE synapse_data SET death = ? WHERE id = ?",values)
            
    # store attributes for all fronts in self._store_fronts
    #  if check==True it is first confirmed whether the front is still active,
    #       if not it will be deleted from the list 
    def _store_attribs(self,check=False):
        if self.verbose >= 7:
            print ("Admin storing optional attributes",len(self._store_fronts))
        to_delete = [] # fronts to be removed from _store_fronts
        for front in self._store_fronts:
            if check and (not front.is_active()):
                to_delete.append(front)
                continue
            nid = front._nid
            fid = front._fid
            if front.order == 0: # a soma
                if not front.is_active(): # soma 'remember' storage
                    continue
                neid = front._sid
            else:
                neid = self._fronts[nid][front._sid]._sid
            n_attrs = len(self._storec_attrs[nid])
            avalues = [] # all attributes for different tables
            for n in range(n_attrs):
                attr = self._storec_attrs[nid][n]
                if self._db_new_objects[nid][n] == 1: # Front
                    value = getattr(front,attr)
                    avalues.append\
                            ((None,neid,fid,self.cycle,value))
                elif self._db_new_objects[nid][n] == 2: # Neuron
                    attr = self._storec_attrs[nid][n]
                    value = getattr(self._neurons[neid],attr)
                    avalues.append((None,neid,0,self.cycle,value))
                elif self._db_new_objects[nid][n] == 4: # ID stored in Front
                    id = getattr(front,attr)
                    avalues.append\
                            ((None,neid,fid,self.cycle,id._nid,id._fid))
                elif self._db_new_objects[nid][n] == 5: # Point stored in Front
                    p = getattr(front,attr)
                    avalues.append\
                            ((None,neid,fid,self.cycle,p.x,p.y.p.z))
                elif self._db_new_objects[nid][n] == 3: # Synapse
                    if front._yid != 0:
                        synapse = self._synapses[abs(front._yid)]
                        value = getattr(synapse,attr)
                        avalues.append\
                            ((None,neid,fid,self.cycle,value))
                    else:
                        n_attrs -= 1 # one less to store
                else:
                    raise BugError("_close_DB","unknown object type") 
            for n in range(n_attrs):
                if len(avalues[n]) == 5: # standard attribute
                    conn_str = "INSERT into " + \
                        self._db_new_tables[self._storec_db_tabs[nid][n]] +\
                        " VALUES (?,?,?,?,?)"
                elif len(avalues[n]) == 6: # ID attribute
                    conn_str = "INSERT into " + \
                        self._db_new_tables[self._storec_db_tabs[nid][n]] +\
                        " VALUES (?,?,?,?,?,?)"
                else: # Point attribute
                    conn_str = "INSERT into " + \
                        self._db_new_tables[self._storec_db_tabs[nid][n]] +\
                        " VALUES (?,?,?,?,?,?,?)"
                self._conn.cursor().execute(conn_str,avalues[n])
        for front in to_delete:
            self._store_fronts.remove(front)

    ### update database and close it
    def _close_DB(self,update_db):
        if self.verbose >= 4:
            print ("Admin",self.my_id,"closing database")
        if not update_db: # just close it
            self._conn.commit()
            self._conn.close() # close database file
            return
        # update database:
        # neurodevsim table
        rtime = time.time() - self.start
        self._conn.cursor().execute("UPDATE neurodevsim SET num_cycles = ?, run_time = ?, importable = ? WHERE id = 1",(self.cycle,rtime,self.importable_db))
        # neuron_data and front_data tables
        #  this does not check for overflow because overflow is very unlikely
        if self._n_next == 1: # no neurons present
            self._conn.commit()
            self._conn.close() # close database file
            return            
        # neurons
        nvalues = [] # for update of neuron_data
        for neid in range(1,self._n_next):
            neuron = self._neurons[neid]
            nvalues.append((neuron.num_fronts,neuron.num_retracted,\
                            neuron.num_synapses,neid))
        self._conn.cursor().executemany("UPDATE neuron_data SET num_fronts = ?, num_retracted = ?, num_synapses = ?  WHERE id = ?",nvalues)
        # fronts
        fvalues = [] # for update of front_data
        if self.verbose >= 6:
            print (self._store_types,self._storef_attrs)
            print (self._db_new_tables, self._storef_db_tabs,self._db_new_objects)
        # check whether stored attributes are present
        for nid in range(1,self._num_types_1): # loop over all neuron types
            if self._store_types[nid] == 2: # last_only attrib_to_db option
                storing = True
                store_neuron = [] # list of neuron attribute indices
                avalues = [] # regular attributes
                ivalues = [] # ID attributes
                pvalues = [] # point attributes
                n_attrs = len(self._storef_attrs[nid])
                for n in range(n_attrs):
                    avalues.append([])
                    ivalues.append([])
                    pvalues.append([])
                    if self._db_new_objects[nid][n] == 2: # neuron attribute
                        store_neuron.append(n) 
            else:
                storing = False
            if storing: 
                # collect neuron attributes
                if store_neuron: # only single loop over admin space necessary
                    for fid in range(1,self._f_next_indices[nid][1]):
                        front = self._fronts[nid][fid]
                        if front.order == 0: # a new soma
                            neid = front._sid
                        else:
                            raise BugError("_close_DB","not a soma")
                        for n in store_neuron:
                            attr = self._storef_attrs[nid][n]
                            value = getattr(self._neurons[neid],attr)
                            avalues[n].append((None,neid,0,self.cycle,value))
            elif not self.importable_db: # nothing else to store for this nid
                continue
            # only executed if last_only storing and/or importable_db
            #  collect flags (if importable_db) and stored attributes data
            for pid in range(1,self._num_procsP2):
                ni0 = (pid-1) * self._front_range[nid-1] + 1 # start in self._fronts
                for fid in range(ni0,self._f_next_indices[nid][pid]):
                    front = self._fronts[nid][fid]
                    if front.order == 0: # a soma
                        neid = front._sid
                    else:
                        neid = self._fronts[nid][front._sid]._sid
                    if self.importable_db:
                        fvalues.append((front._pid,front._flags,front._dbid))
                    if storing:
                        for n in range(n_attrs):
                            attr = self._storef_attrs[nid][n]
                            if self._db_new_objects[nid][n] == 1:
                                value = getattr(front,attr)
                                avalues[n].append\
                                        ((None,neid,fid,self.cycle,value))
                            elif self._db_new_objects[nid][n] == 2:
                                continue # dealt with before
                            elif self._db_new_objects[nid][n] == 4:
                                id = getattr(front,attr)
                                ivalues[n].append\
                                    ((None,neid,fid,self.cycle,\
                                        id._nid,id._fid))
                            elif self._db_new_objects[nid][n] == 5:
                                p = getattr(front,attr)
                                pvalues[n].append\
                                    ((None,neid,fid,self.cycle,p.x,p.y,p.z))
                            elif self._db_new_objects[nid][n] == 3:
                                if front._yid != 0:
                                    synapse = self._synapses[abs(front._yid)]
                                    value = getattr(synapse,attr)
                                    avalues[n].append\
                                        ((None,neid,fid,self.cycle,value))
                            else:
                                raise BugError("_close_DB","unknown object type")
            # write stored attributes data
            if storing:
                for n in range(n_attrs):
                    if avalues[n]: # store regular attributes
                        conn_str = "INSERT into " + \
                            self._db_new_tables[self._storef_db_tabs[nid][n]]\
                            + " VALUES (?,?,?,?,?)"
                        self._conn.cursor().executemany(conn_str,avalues[n])
                    elif ivalues[n]: # store ID attributes
                        conn_str = "INSERT into " + \
                        self._db_new_tables[self._storef_db_tabs[nid][n]]\
                            + " VALUES (?,?,?,?,?,?)"
                        self._conn.cursor().\
                                    executemany(conn_str,ivalues[n])
                    elif pvalues[n]: # store point attributes
                        conn_str = "INSERT into " + \
                        self._db_new_tables[self._storef_db_tabs[nid][n]]\
                            + " VALUES (?,?,?,?,?,?,?)"
                        self._conn.cursor().\
                                    executemany(conn_str,pvalues[n])
            # write flags data
            if fvalues:
                self._conn.cursor().executemany("UPDATE front_data SET parent_id = ?, flags = ? WHERE id = ?",fvalues)
        if self.importable_db: # also store arc data if present
            no_arc_db = True
            avalues = [] # arc_data to be written to db
            for n in range(self._tot_arcs):
                arc = self._arcs[n]
                if arc.cycle > 0: # arc present
                    if no_arc_db: # create tables
                        self._arcs_DB()
                        no_arc_db = False
                    avalues.append((None,n,arc.cycle,arc.complete,\
                                   arc.neuron_id,arc.caller_fid,\
                                   arc.last_fid,arc.sphere._nid,arc.sphere._fid,\
                                   arc.arc_angle,arc.num_points,\
                                   arc.count,arc.next_point))
                    pvalues = [] # arc_points to be written to db
                    for i in range(arc.num_points):
                        p = self._arc_points[arc.index + i]
                        pvalues.append((None,n,p.x,p.y,p.z))
                    self._conn.cursor().executemany("INSERT into arc_points VALUES (?,?,?,?,?)",pvalues)
            if avalues: 
                self._conn.cursor().executemany("INSERT into arc_data VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",avalues)
            # and store mig_fronts if migrating fronts
            if self._mig_data > 0: # migrating fronts present
                mvalues = []
                self._mig_fronts_DB() # make the table
                for f in self._mig_fronts: # and populate it
                    mvalues.append((None,f._nid,f._fid))
                self._conn.cursor().executemany("INSERT into mig_fronts_data VALUES (?,?,?)",mvalues)
        self._conn.commit()
        self._conn.close() # close database file

    ### grid_lock brokering
    
    # Processor ids: 1 Admin, 2 - num_procsP2 other cores
    
    # grid_wlock values:
    #  < 0: locked for actual write-only, value == - processor id
    #    0: unlocked, initial state
    #  > 0: locked for writing (read-only possible), value == processor id
    #       this status may last for a long time
    
    # grid_rlock values:
    #    0: unlocked, initial state
    #  > 0: locked for read-only, value == processor id (short status)

    # self._gwlock_request and self._grlock_request values:
    #   -1: no request made, initial state
    #  >=0: request made, value == grid index
    
    # a method that monitors processor requests for grid locking and
    #   provides a lock if possible
    # the processor has to release the grid lock asap
    def _lock_broker(self):
        # first check for writing locks
        for pid in range(2,self._num_procsP2):
            if self._gwlock_request[pid] != 0: # request made
                gid = self._gwlock_request[pid]
                if gid > 0: # standard lock request
                    if (self._grid_wlock[gid] == 0) and (self._grid_rlock[gid] == 0):
                        # not locked at present for either writing or reading
                        self._gwlock_request[pid] = 0
                        self._grid_wlock[gid] = pid # lock it for process
                        if self._debug:
                            self._lw_gids.add(gid)
                else: # negative gid: immediate writing lock request
                    gid = -gid
                    if self._grid_wlock[gid] == pid: # should already be locked
                        self._gwlock_request[pid] = 0
                        self._grid_wlock[gid] = -pid # lock for immediate writing
                    else:
                        raise BugError("_lock_broker","immediate writing request for unlocked gid " + str(gid) + " by " + str(pid) + " for lock " + str(self._grid_wlock[gid]))
        # then check for reading locks
        for pid in range(2,self._num_procsP2):
            if self._grlock_request[pid] > 0: # request made
                gid = self._grlock_request[pid]
                if self._grid_rlock[gid] == 0: # not reading locked at present
                    if self._grid_wlock[gid] >= 0:
                        # no writing lock or not actively being used
                        self._grlock_request[pid] = 0
                        self._grid_rlock[gid] = pid # reading lock it for process
                        if self._debug:
                            self._lr_gids.add(gid)
                            
    # a method that checks whether all locks have been released at end of cycle
    def _lock_check(self):
        # first check for writing locks
        for gid in self._lw_gids:
            if self._grid_wlock[gid] > 0:
                raise BugError("_lock_check","cycle " + str(self.cycle) + " not unlocked write gid " + str(gid) + " by " + str(self._grid_wlock[gid]))
        # then check for reading locks
        for gid in self._lr_gids:
            if self._grid_rlock[gid] > 0:
                raise BugError("_lock_check","cycle " + str(self.cycle) + " not unlocked read gid " + str(gid) + " by " + str(self._grid_rlock[gid]))
        # clear for next cycle
        self._lw_gids = set() # gids write locked during this cycle
        self._lr_gids = set() # gids read locked during this cycle

# Admin_slaves: subclass of Admin_agent used for different subtasks
class Admin_slave(Admin_agent):

    def __init__(self,my_num,num_procs,verbose,master_constell):
        self.master = False # this is the slave
        self._num_procs = num_procs
        self._num_procsP2 = num_procs + 2
        self.my_id = my_num
        self.verbose = verbose
        self.master_clock = master_constell[5]
        self.cycle = 0
        self._debug = master_constell[24]
        if self._debug:
            self._lw_gids = set() # gids write locked during this cycle
            self._lr_gids = set() # gids read locked during this cycle
        self._grid_wlock = master_constell[8]
        self._grid_rlock = master_constell[32]
        self._gwlock_request = master_constell[25]
        self._grlock_request = master_constell[27]
        if my_num == -1: # slave running fetch, pre-fetch and lock broker
            self.pstart = 2 # _lock_broker ignores admin
            self._proc_status = master_constell[26]
            self._proc_status[1] = 1 # signal that this proc is online
            self._volume = master_constell[34]
            self._fronts = master_constell[0]
            self._actives = master_constell[1]
            act_range = master_constell[13]
            self._new_range = int(act_range * 1.5)
            self._new_AIDs = master_constell[2] # all needed
            self._n_starts = master_constell[40]
            self._gids = master_constell[19]
            self.constellation = Constellation(self,master_constell)
        elif my_num == -2: # slave running _lock_broker
            self.pstart = 1 # _lock_broker includes admin
                
    #  Admin_slave loop dedicated to fetching and prefetching for processes, also
    #    calls _lock_broker
    #  Master uses self._proc_status[1] to communicate start/end of cycle:
    #    1: ready state set by slave
    #    5: receive growing_fronts and migrating_fronts data in self._new_AIDs
    #    6: processed the growing_fronts and migrating_fronts data, set by slave
    #   10: new cycle started, set by Master
    #   11: start processing cycle slave, set by slave
    #   15: end processing cycle Master, set by Master if branch retractions
    #   20: cycle finished slave, set by slave
    #   25: cycle also finished by Master
    #   30: simulation loop finished, set by Master or detected by slave
    #  Slave1 uses self._proc_status[0] to transmit num_f_out to Master
    def _slave_loop1(self):
        if self.verbose >= 4:
            print ("Admin slave1 starting")
        # load balancing
        self._growing_fronts = [] # all fronts with is_growing() True
        self._migrating_fronts = [] # all fronts with is_migrating() True
        self._active_fronts = [] # all other fronts with is_active() True
        self._future_active = {} # dict by cycle fronts that need to be activated at that cycle
        gids = []
        while True: # endless loop
            status = self._proc_status[1]
            if status == 5: # receive initial data
                n = 0 # index into self._new_AIDs
                num_soma = self._new_AIDs[0].index # number of growing fronts
                num_soma2 = self._new_AIDs[1].index # migrating fronts
                num_soma3 = self._new_AIDs[2].index # active fronts
                if self.verbose >= 6:
                    print ("Admin slave1 receiving",num_soma-3,num_soma2-num_soma,num_soma3-num_soma2,"fronts")
                for n in range(3,num_soma):
                    id = self._new_AIDs[n].ID
                    soma = self._fronts[id._nid][id._fid]
                    self._growing_fronts.append(soma)
                    if self.verbose >= 6:
                        print ("Admin slave1 received growing",soma)
                for n in range(num_soma,num_soma2):
                    id = self._new_AIDs[n].ID
                    soma = self._fronts[id._nid][id._fid]
                    self._migrating_fronts.append(soma)
                    if self.verbose >= 6:
                        print ("Admin slave1 received migrating",soma)
                for n in range(num_soma2,num_soma3):
                    id = self._new_AIDs[n].ID
                    soma = self._fronts[id._nid][id._fid]
                    self._active_fronts.append(soma)
                    if self.verbose >= 6:
                        print ("Admin slave1 received active",soma)
                self._proc_status[1] = 6 # finished
                if self.verbose >= 6:
                    print ("Admin slave1 finished receiving fronts")
            elif status == 10: # loop through one cycle
                self.cycle = self.master_clock.value
                self._proc_status[1] = 11 # started
                if self.verbose >= 6:
                    print ("Admin slave1 starting cycle")
                retracted_f = [] # list of roots of branches to retract
                new_inactive = [] # list of fronts that become inactive next cycle
                next_growing = [] # self._growing_fronts for next cycle
                next_migrating = [] # self._migrating_fronts for next cycle
                next_active = [] # self._active_fronts for next cycle
                # potential duplication, but safer
                num_f_in = num_f_out = len(self._growing_fronts) + \
                                       len(self._migrating_fronts) + \
                                       len(self._active_fronts)
                self._proc_status[0] = num_f_out # send to master
                if num_f_in == 0:
                    self._proc_status[1] = 30 # ending simulation loop
                    continue
                elif num_f_in < self._num_procs:
                    max_procs = 2 + num_f_in
                    stop_procs = True
                else:
                    max_procs = self._num_procsP2
                    stop_procs = False
                start_AIDs = [0] * self._num_procsP2 # relative start of process block
                if self.verbose >= 6:
                    print ("Admin slave1 sending",num_f_in,"fronts",max_procs,stop_procs)
                start = time.time()
                # keep cycling till all fronts are processed
                # current balancing approach: tightly locked -> cores will have to
                #   wait for next front
                if self._num_procs > 1:
                    lock_gids_th = self._num_procsP2 # lock gids if num_f_in > lock_gids_th
                else: # don't bother locking on single processor run
                    lock_gids_th = num_f_in # never lock
                # control pre-fetching of fronts: not at end because some
                #   processes may take a lot of time and cannot execute the 
                #   prefetched front.
                reserve_th = self._num_procsP2 # get next front ready if num_f_in > lock_gids_th
                grow_index0 = 0 # start from beginning of list
                migr_index0 = 0 # start from beginning of list
                if self._debug: # keep track of ids sent and received
                    target = num_f_in
                    sent_id = []
                    receiv_id = []
                n_fails = 0 # number of failed fetch loops since last success
                if self.verbose >= 6:
                    last_fid = 0 # used to prevent endless repetition of same message
                while num_f_out: 
                    self._lock_broker() # essential duty, called many times
                    # find pid needing a front: at present on demand only
                    for pid in range(2,max_procs):
                        ns0 = self._n_starts[pid] + start_AIDs[pid]
                        aid = self._new_AIDs[ns0]
                        ns1 = aid.index
                        while ns1 > 0: # possibly data for multiple parent fronts
                            num_f_out -= 1 # finished one
                            # process front that was called
                            id = aid.ID
                            if self._debug: # keep track of ids sent and received
                                receiv_id.append(id)
                            if self.verbose >= 6:
                                print ("Admin slave1 received from",pid,id,ns0,ns1,num_f_in,num_f_out,n_fails)
                            front = self._fronts[id._nid][id._fid]
                            if front._gid < 0: # locked gids
                                # unlocked by process
                                front._gid = -front._gid # set back to unlocked
                                # reset to start of lists as gid has been freed
                                grow_index0 = 0 # start from beginning of list
                                migr_index0 = 0 # start from beginning of list
                            # store for next cycle if active, automatically in order called now
                            if front.is_migrating():
                                next_migrating.append(front)
                                if front.has_moved():
                                    any_f_moved = True
                            elif front.is_growing():
                                next_growing.append(front)
                            elif front.is_active():
                                next_active.append(front)
                            # process other new_AIDs
                            for n in range(ns0 + 1, self._n_starts[pid] + ns1):
                                self._lock_broker() # essential duty, called many times
                                status = self._new_AIDs[n].status
                                id = self._new_AIDs[n].ID
                                front = self._fronts[id._nid][id._fid]
                                if status == b'e':
                                    if front.is_growing():
                                        next_growing.append(front)
                                    elif front.is_active():
                                        next_active.append(front)
                                elif status == b't': # a trailing axon was created
                                    if front.is_active():
                                        next_active.append(front)
                                elif status == b'a': # enable front
                                    if front.is_migrating():
                                        next_migrating.append(front)
                                    elif front.is_growing():
                                        next_growing.append(front)
                                    elif front.is_active():
                                        next_active.append(front)
                                elif (status == b'i') or (status == b'g') or\
                                            (status == b'm'): # disable front
                                    front = self._fronts[id._nid][id._fid]
                                    # can be added to next_active later
                                    new_inactive.append(front)
                                    till_cycle = self._new_AIDs[n].index
                                    if till_cycle > self.cycle:
                                        # future activation at till_cycle requested
                                        if till_cycle in self._future_active:
                                            self._future_active[till_cycle].\
                                                        append([status,front])
                                        else:
                                            self._future_active[till_cycle] = \
                                                        [[status,front]]
                                elif status == b'r': # retracted front
                                    new_inactive.append(front) # sometimes kept active
                                elif status == b'd': # retract branch
                                    retracted_f.append(front)
                            start_AIDs[pid] = ns1 # update for next round
                            ns0 = self._n_starts[pid] + ns1
                            # check whether more data ready
                            aid = self._new_AIDs[ns0]
                            ns1 = aid.index
                        # Deal with end of cycle
                        if num_f_in == 0: # no more fronts to send
                            # check that both fetch and pre-fetch are done or in process
                            if (self._actives[pid]._nid == 0) and \
                                (self._actives[pid + self._num_procs]._nid == 0):
                                    self._actives[pid] = negat_ID # signal end of cycle
                            continue
                        # Check whether we should fetch or pre-fetch
                        if (self._actives[pid]._nid > 0) and (num_f_in <= reserve_th):
                            # previous fetch unused and no more pre-fetching
                            #   (works also for negat_ID in self._actives[pid])
                            continue
                        # check whether pre-fetch necessary
                        if self._actives[pid + self._num_procs]._nid > 0: 
                            continue # already done prefetch
                        # pid has no instructions
                        found = False # found a front for pid
                        # decide whether we should check for gid competition
                        if n_fails <= self._num_procs: # not too many fails
                            lock_gids = num_f_in > lock_gids_th
                        else:
                            lock_gids = False 
                            grow_index0 = 0 # start from beginning of list
                            migr_index0 = 0 # start from beginning of list
                        # do growing fronts first
                        for n in range(grow_index0,len(self._growing_fronts)):
                            front = self._growing_fronts[n]
                            if self.verbose >= 6:
                                print ("Admin slave1 trying grow",pid,front)
                            self._lock_broker() # essential duty, called many times
                            if lock_gids: # check whether free
                                gids = []
                                # check gid availibility: bypass _get_gids for speed
                                ngid = front._gid
                                failed = False
                                for i in range(ngid + 1, ngid + 1 + self._gids[ngid]):
                                    gid = self._gids[i]
                                    if self._grid_wlock[gid] != 0:
                                        if (self.verbose >= 6) and (front._fid != last_fid):
                                            last_fid = front._fid # do not repeat this one
                                            print ("Admin slave1 grow failed",pid,num_f_in,gid,self._grid_wlock[gid],front)
                                        grow_index0 = n # skip this one next pid
                                        failed = True # already locked
                                        break
                                    gids.append(gid)
                                if failed:
                                    continue # for front in self._growing_fronts
                            # use this front
                            if self._actives[pid]._nid > 0: # save in reserve
                                if self.verbose >= 6:
                                    print ("Admin slave1 sent growing reserve to",pid,front.get_id(),gids)
                                self._actives[pid + self._num_procs] = front.get_id()
                            else: # save in regular place
                                if self.verbose >= 6:
                                    print ("Admin slave1 sent growing to",pid,front.get_id(),gids)
                                self._actives[pid] = front.get_id()
                            if self._debug: # keep track of ids sent and received
                                sent_id.append(front.get_id())
                            num_f_in -= 1 # started one
                            found = True
                            if lock_gids: # now really lock them
                                for gid in gids: # lock gids for admin
                                    self._grid_wlock[gid] = pid # lock to process
                                front._gid = -ngid # mark as locked
                            self._lock_broker() # essential duty, called many times
                            del self._growing_fronts[n] # done this one
                            break
                        if found:
                            n_fails = 0 # reset
                            continue # for pid in range
                        # do migrating fronts next
                        for n in range(migr_index0,len(self._migrating_fronts)):
                            front = self._migrating_fronts[n]
                            if self.verbose >= 6:
                                print ("Admin slave1 trying mig",pid,front)
                            self._lock_broker() # essential duty, called many times
                            if lock_gids: # check whether free
                                gids = []
                                # check gid availibility: bypass _get_gids for speed
                                ngid = front._gid
                                failed = False
                                for i in range(ngid + 1, ngid + 1 + self._gids[ngid]):
                                    gid = self._gids[i]
                                    if self._grid_wlock[gid] != 0:
                                        if (self.verbose >= 6) and (front._fid != last_fid):
                                            last_fid = front._fid # do not repeat this one
                                            print ("Admin slave1 mig failed",pid,num_f_in,gid,self._grid_wlock[gid],front.get_id(),n_fails)
                                        migr_index0 = n # skip this one next pid
                                        failed = True # already locked
                                        break
                                    gids.append(gid)
                                if failed:
                                    continue # for front in self._migrating_fronts
                            # use this front
                            if self._actives[pid]._nid > 0: # save in reserve
                                if self.verbose >= 6:
                                    print ("Admin slave1 sent migrating reserve to",pid,front.get_id(),gids)
                                self._actives[pid + self._num_procs] = front.get_id()
                            else: # save in regular place
                                if self.verbose >= 6:
                                    print ("Admin slave1 sent migrating to",pid,front.get_id(),gids)
                                self._actives[pid] = front.get_id()
                            if self._debug: # keep track of ids sent and received
                                sent_id.append(front.get_id())
                            num_f_in -= 1 # started one
                            found = True
                            if lock_gids: # now really lock them
                                for gid in gids: # lock gids for admin
                                    self._grid_wlock[gid] = pid # lock to process
                                front._gid = -ngid # mark as locked
                            self._lock_broker() # essential duty, called many times
                            del self._migrating_fronts[n] # done this one
                            break
                        if found:
                            n_fails = 0 # reset
                            continue # for pid in range     
                        # do active fronts last
                        if self._active_fronts:
                            if self.verbose >= 6:
                                print ("Admin slave1 trying act",pid,front)
                            self._lock_broker() # essential duty, called many times
                            # use this front
                            fid = self._active_fronts.pop(0).get_id()
                            if self._actives[pid]._nid > 0: # save in reserve
                                self._actives[pid + self._num_procs] = fid
                                if self.verbose >= 6:
                                    print ("Admin slave1 sent active reserve to",pid,self._actives[pid + self._num_procs])
                            else: # save in regular place
                                self._actives[pid] = fid
                                if self.verbose >= 6:
                                    print ("Admin slave1 sent active to",pid,self._actives[pid])
                            if self._debug: # keep track of ids sent and received
                                sent_id.append(fid)
                            num_f_in -= 1 # started one
                            n_fails = 0 # reset
                            continue # for pid in range
                        # failed to find one
                        if self._actives[pid]._nid == 0: # failure of a fetch cycle
                            n_fails += 1
                            if n_fails > 2 * self._num_procs:
                                raise BugError("_slave_loop1","cannot allocate fronts to processors")
                    if stop_procs: # less active fronts than processes
                        self._lock_broker() # essential duty, called many times
                        for pid in range(max_procs,self._num_procsP2):
                           self._actives[pid] = negat_ID # signal end of cycle
                        stop_procs = False # do this only once 
                # do branch retractions
                if retracted_f: # cannot run together with _lock_broker
                    if self.verbose >= 6:
                        print ("Slave1 preparing to retract branches")
                    self._lock_broker()
                    # check whether admin completed cycle
                    while self._proc_status[1] < 15: # admin still processing
                        self._lock_broker()
                        time.sleep(PAUSE)
                    # check whether all processes completed cycle
                    for n in range(10000):
                        not_finish = 0
                        for pid in range(2,max_procs):
                            if self._proc_status[pid] != 20:
                                not_finish = pid
                        if not_finish == 0:
                            break # out of for n
                    if not_finish > 0:
                        raise BugError("_slave_loop1","process " + str(not_finish) + " does not reach end of cycle")
                    deleted_id = self._retract_branches(retracted_f,new_inactive,\
                                                    returnID=True)
                    self._lock_broker()
                    # transmit deleted fronts to master for db storing
                    num_dels = len(deleted_id) # range
                    if num_dels > 0:
                        #print (self.cycle,self.proc_status[1],"Admin slave1 retracted_f",num_dels)
                        n = 0 # index into self.new_AIDs which gets overwritten, not used now
                        for id in deleted_id:
                            self._new_AIDs[n] = ActiveFrontID(id,b'2',num_dels)
                            n += 1
                    else:
                        self._new_AIDs[0] = ActiveFrontID(empty_ID,b'0',0)
                self._lock_broker()
                # reset self._growing_fronts, self._migrating_fronts, self._active_fronts
                for front in new_inactive: # make inactive and remove from lists
                    self._lock_broker()
                    if front.is_growing():
                        front.clear_growing()
                        if front in next_growing:
                            next_growing.remove(front)
                    if front.is_migrating():
                        front.clear_migrating()
                        if front in next_migrating:
                            next_migrating.remove(front)
                    front._clear_active()
                    if front in next_active:
                        next_active.remove(front)
                self._growing_fronts = next_growing
                self._migrating_fronts = next_migrating
                self._active_fronts = next_active
                next = self.cycle + 1
                if next in self._future_active:
                    for item in self._future_active[next]:
                        status = item[0]
                        f = item[1]
                        f._set_active() # always make active
                        if status == b'i': # make active only
                            if f not in self._active_fronts:
                                self._active_fronts.append(f)
                        elif status == b'g': # make growing
                            f.set_growing()
                            if f not in self._growing_fronts:
                                self._growing_fronts.append(f)
                        elif status == b'm': # make migrating
                            f.set_migrating()
                            if f not in self._migrating_fronts:
                                self._migrating_fronts.append(f)
                        else:
                            raise BugError("_slave_loop1","unknown status " + str(status))
                    del self._future_active[next]
                if self._debug:
                    self._lock_check() # check that all locks have been cleared
                self._lock_broker()
                # for next cycle
                num_f_in = num_f_out = len(self._growing_fronts) + \
                                       len(self._migrating_fronts) + \
                                       len(self._active_fronts)
                self._proc_status[0] = num_f_out # send to master
                self._proc_status[1] = 20 # finished
                if self.verbose >= 6:
                    print ("Admin slave1 finished cycle")
            elif status == 20: # keep lock_broker running
                self._lock_broker()
            elif (status == 1) or (status == 6) or (status == 15) or (status == 25): 
                # wait for next instruction
                time.sleep(PAUSE)
            elif status == 30: # no active simulation loop
                time.sleep(0.03)
            else:
                raise BugError("_slave_loop1","unknown status " + str(status))
        
    #  Admin_slave loop dedicated to lock brokering
    def _slave_loop2(self):
        if self.verbose >= 4:
            print ("Admin slave2 starting")
        old_cycle = 0
        while True: # endless loop
            new_cycle = self.master_clock.value
            if new_cycle != old_cycle:
                if self._debug:
                    self._lock_check() # check that all locks have been cleared
                old_cycle = new_cycle
            if new_cycle > 0: # simulation_loop running, at present we do not detect end cycle
                self._lock_broker()
            elif new_cycle < 0:
                time.sleep(0.03)
            else: # new_cycle == 0
                time.sleep(PAUSE)
            
    # this version deals with write lock requests from admin1
    def _lock_broker(self):
        # first check for writing locks
        for pid in range(self.pstart,self._num_procsP2):
            if self._gwlock_request[pid] != 0: # request made
                gid = self._gwlock_request[pid]
                if gid > 0: # standard lock request
                    if (self._grid_wlock[gid] == 0) and (self._grid_rlock[gid] == 0):
                        # not locked at present for either writing or reading
                        self._gwlock_request[pid] = 0
                        self._grid_wlock[gid] = pid # lock it for process
                        if self._debug:
                            self._lw_gids.add(gid)
                else: # negative gid: immediate writing lock request
                    gid = -gid
                    if self._grid_wlock[gid] == pid: # should already be locked
                        self._gwlock_request[pid] = 0
                        self._grid_wlock[gid] = -pid # lock for immediate writing
                    else:
                        raise BugError("_lock_broker","immediate writing request for unlocked gid " + str(gid) + " by " + str(pid) + " for lock " + str(self._grid_wlock[gid]))
        # then check for reading locks
        for pid in range(2,self._num_procsP2):
            if self._grlock_request[pid] > 0: # request made
                gid = self._grlock_request[pid]
                if self._grid_rlock[gid] == 0: # not reading locked at present
                    if self._grid_wlock[gid] >= 0:
                        # no writing lock or not actively being used
                        self._grlock_request[pid] = 0
                        self._grid_rlock[gid] = pid # reading lock it for process
                        #print ("_lock_broker read locking",gid,pid)
                        if self._debug:
                            self._lr_gids.add(gid)
    def destruction(self,crash=False): # not implemented on slave
        return
        
def _admin_init(my_num,num_procs,verbose,master_constell):
    
    if verbose >= 5:
        print ("Process admin slave",abs(my_num),"online")
    admin2 = Admin_slave(my_num,num_procs,verbose,master_constell)
    if my_num == -1:
        admin2._slave_loop1()
    elif my_num == -2:
        admin2._slave_loop2()
    else:
        raise BugError("_admin_init","unknown my_num " + str(my_num))
    if verbose >= 6:
        print ("Process",my_num,"finishes")
