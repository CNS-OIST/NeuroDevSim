####################################################################################
#
#    NeuroDevSim: Neural Development Simulator
#    Copyright (C) 2019-2022 Okinawa Institute of Science and Technology Graduate
#    University, Japan.
#
#    See the file AUTHORS for details.
#    This file is part of NeuroDevSim. It contains fucntions that use NeuroDevSim
#    databases.
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

import sys
import os
import random
from scipy import stats
import sqlite3
import copy
import numpy as np
import scipy
from scipy.linalg import expm
from colorama import Fore

from neurodevsim.simulator import Point,dist3D_cyl_to_cyl,dist3D_point_to_cyl,\
                                  DataID,_key_to_DataID,BugError

# nds version 1.0.1

### PLOTTING FUNCTIONS: 
def nds_plot(db_name,pdf_out=True,max_cycle=-1,neurons=[],wire=False,azim=-60,\
             elev=30,box=False,no_axis=False,scale_axis=False,axis_ticks=True,\
             soma_black=True,color_scheme=0,color_data=None,neuron_colors=None,\
             prefix="",postfix="",show_retracted=False,verbose=1):
    """ Generate a 3D plot and save it to a pdf file.
    
    Parameters
    ----------
    db_name : string : name of NeuroDevSim database file.
    Optional :
    axis_ticks : boolean : show axis ticks, default True.
    azim : float : azimuth in degrees of camera, default -60.
    box : box format [[left, front, bottom], [right, back, top]] to plot, allows to zoom in, default full *sim_volume.*
    color_scheme : integer -1 - 3 : controls how colors change: 0: every neuron has a different color; 1: neurons of same type have same color, different types have different colors; 2: different branches in a neuron have different colors, based on branch_name; 3: color set by front attribute as defined in color_data, -1: use colors defined in neuron_colors, default 0.
    color_data : list : [front attribute name, min value, max value], data necessary for color_scheme 3, default None.
    elev : float : elevation in degrees of camera, default 30.
    max_cycle : integer : stop plotting at this cycle (inclusive), default -1: plot till end of simulation.
    neuron_colors : string : name of text file containing dictionary info by neuron name that specifies color to use, use nds_get_color_dict to obtain a valid file and then edit it, default None.
    neurons : list of string : only plot the neurons with names (wildcard) listed, default: empty list (plot all).
    no_axis : boolean : suppress drawing of axes, default False.
    pdf_out : boolean : ,save plot as a pdf file, if False plot is shown in a window, default True.
    prefix : string : attach string before database name to specify a directory in name of pdf file, default ''.
    postfix : string : attach string after database name in name of pdf file, default ''.
    scale_axis: boolean or list of 3 floats: list as [1.0,1.0,1.0] decrease one or more values to change relative scaling of axes, value for one axis should always be 1.0; default False.
    show_retracted : boolean : if True retracted dendrites are drawn in black, if False they are not drawn, default False.
    soma_black : boolean : all somata are black for increased contrast, default True.
    verbose : integer 0-1 : print information about all fronts plotted (1), default 1.
    wire : boolean : if True all fronts have same small radius, default False.
    """

    if not os.path.isfile(db_name):
        print ("Error in nsd_plot:",db_name,"not found")
        return
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # check color settings
    if (color_scheme < -1) or (color_scheme > 3):
        print ("Error in nsd_plot: invalid color_scheme value, range is -1 - 3")
    if color_scheme == 3:
        if not color_data:
            print ("Error in nsd_plot: color_data must be defined for color_scheme=3")
        elif not isinstance(color_data,list):
            print ("Error in nsd_plot: color_data should be a list")
        elif len(color_data) != 3:
            print ("Error in nsd_plot: color_data should be a list with 3 entries")
    if color_scheme == -1:
        try:
            f = open(neuron_colors,'r')
            color_data = neuron_colors
            #close(f)
        except:
            print ("Error in nsd_plot: could not open",neuron_colors)
    # Get general data
    try:
        cursor.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in nsd_plot: not a NeuroDevSim database")
        return
    nret = cursor.fetchone()
    version = nret['version']
    if (version < 93): 
        print ("Error in nsd_plot: wrong database version",version)
        return
    if max_cycle < 0:
        max_cycle = nret['num_cycles']
        if max_cycle == 0:  # database of a crashed simulation -> num_cycles not updated
            max_cycle = 1000000 # arbitrary large value
    # name of output file
    out_name = prefix + strip_db_name(db_name) + postfix
    show_retraction = not show_retracted
    if pdf_out:
        if verbose > 0:
            print ("pdf file:",out_name+".pdf")
        _draw_figure(conn,nret,out_name,1,0,max_cycle,wire,azim,elev,box,\
                     soma_black,no_axis,scale_axis,axis_ticks,color_scheme,\
                     show_retraction,verbose,color_data,300,neurons,version)
    else:
        _draw_figure(conn,nret,"",0,0,max_cycle,wire,azim,elev,box,\
                     soma_black,no_axis,scale_axis,axis_ticks,color_scheme,\
                     show_retraction,verbose,color_data,300,neurons,version)
                     
def nds_movie(db_name,min_cycle=-1,max_cycle=-1,neurons=[],wire=False,azim=-60.0,elev=30.0,\
            box=False,no_axis=False,scale_axis=False,axis_ticks=True,soma_black=True,\
            color_scheme=0,color_data=None,neuron_colors=None,prefix="",postfix="",\
            show_retraction=True,verbose=1,dpi=300):
    """ Generate a 3D movie and save it to a mp4 file.
    
    Parameters
    ----------
    db_name : string : name of NeuroDevSim database file.
    Optional :
    axis_ticks : boolean : show axis ticks, default True.
    azim : float : azimuth in degrees of camera, default -60.
    box : box format [[left, front, bottom], [right, back, top]] to plot, allows to zoom in, default full *sim_volume.*
    color_scheme : integer -1 - 3 : controls how colors change: 0: every neuron has a different color; 1: neurons of same type have same color, different types have different colors; 2: different branches in a neuron have different colors, based on branch_name; 3: color set by front attribute as defined in color_data, -1: use colors defined in neuron_colors, default 0.
    color_data : list : [front attribute name, min value, max value], data necessary for color_scheme 3, default None.
    dpi : integer : resolution of movie frames, default 300.
    elev : float : elevation in degrees of camera, default 30.
    max_cycle : integer : stop plotting at this cycle (inclusive), default -1: plot till end of simulation.
    min_cycle : integer : start plotting at this cycle, default -1: plot from begin of simulation.
    neuron_colors : string : name of text file containing dictionary info by neuron name that specifies color to use, use nds_get_color_dict to obtain a valid file and then edit it, default None.
    neurons : list of string : only plot the neurons with names (wildcard) listed, default: empty list (plot all).
    no_axis : boolean : suppress drawing of axes, default False.
    pdf_out : boolean : ,save plot as a pdf file, if False plot is shown in a window, default True.
    prefix : string : attach string before database name to specify a directory in name of pdf file, default ''.
    postfix : string : attach string after database name in name of pdf file.
    scale_axis: boolean or list of 3 floats: list as [1.0,1.0,1.0] decrease one or more values to change relative scaling of axes, value for one axis should always be 1.0; default False.
    show_retraction : boolean : retracted dendrites will disappear, they are intially drawn in black, default True.
    soma_black : boolean : all somata are black for increased contrast, default True.
    verbose : integer 0-1 : print information about all fronts plotted (1), default 1.
    wire : boolean : if True all fronts have same small radius, default False.
    """
    if not os.path.isfile(db_name):
        print ("Error in nds_movie:",db_name,"not found")
        return
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # check color settings
    if (color_scheme < 0) or (color_scheme > 3):
        print ("Error in nds_movie: invalid color_scheme value, range is 0-3")
    if color_scheme == 3:
        if not color_data:
            print ("Error in nds_movie: color_data must be defined for color_scheme=3")
        elif not isinstance(color_data,list):
            print ("Error in nds_movie: color_data should be a list")
        elif len(color_data) != 3:
            print ("Error in nds_movie: color_data should be a list with 3 entries")
    if color_scheme == -1:
        try:
            f = open(neuron_colors,'r')
            color_data = neuron_colors
            #close(f)
        except:
            print ("Error in nds_movie: could not open",neuron_colors)
    # Get volume and cycle data
    try:
        cursor.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in nds_movie: not a NeuroDevSim database")
        return
    nret = cursor.fetchone()
    version = nret['version']
    if (version < 93): 
        print ("Error in nds_movie: wrong database version",version)
        return
    if min_cycle < 0:
        min_cycle = 0 # start
    if max_cycle < 0:
        max_cycle = nret['num_cycles'] + 1
        if max_cycle == 0:  # database of a crashed simulation -> num_cycles not updated
            max_cycle = 1000000 # arbitrary large value
    # name of output file
    out_name = prefix + strip_db_name(db_name) + postfix
    if verbose > 0:
        print ("movie file:",out_name+".mp4")
    _draw_figure(conn,nret,out_name,2,min_cycle,max_cycle,wire,azim,elev,box,soma_black,\
                    no_axis,scale_axis,axis_ticks,color_scheme,show_retraction,verbose,\
                    color_data,dpi,neurons,version)

def nds_interact(db_name,neuron_id,front_id,cycle,focus=10,azim=-60,elev=30,\
                 radius_scale=2.0,sphere_scale=1.0,verbose=0):
    """ Make a notebook plot of a NeuroDevSim database in an interactive session.
    
    Standard use is to focus on a small cube around a provided reference front which will be colored red. If the front information is not provided (``neuron_id==0 and front_id==0``) the entire simulation is shown.
    
    Parameters
    ----------
    db_name : string : name of NeuroDevSim database file.
    neuron_id : integer : neuron_id value of a reference front present in *db_name*.
    front_id : integer : front_id value of a reference front present in *db_name*.
    cycle : integer : plot till this cycle.
    Optional :
    focus : float : distance in µm to plot around reference front, default 10 µm.
    azim : float : azimuth in degrees of camera, default -60.
    elev : float : elevation in degrees of camera, default 30.
    radius_scale : float : increase radius of all cylindrical fronts to improve visibility, default 2.
    sphere_scale : float : change radius of all spherical fronts to improve visibility, default 1.
    verbose : integer 0-1 : print information about all fronts plotted (1), default 0.
    
    Returns
    -------
    ax : subplot object
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as manimation
    from mpl_toolkits.mplot3d import Axes3D
    if not os.path.isfile(db_name):
        print ("Error in nds_interact:",db_name,"not found")
        return
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Get volume and cycle data
    try:
        cursor.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in nds_interact: not a NeuroDevSim database")
        return
    nret = cursor.fetchone()
    version = nret['version']
    if (version < 93): 
        print ("Error in nds_interact: wrong database version",version)
        return
    max_cycle = nret['num_cycles']
    if max_cycle == 0:  # database of a crashed simulation -> num_cycles not updated
        max_cycle = 1000000 # arbitrary large value
    xmin = nret['xmin']
    ymin = nret['ymin']
    zmin = nret['zmin']
    xmax = nret['xmax']
    ymax = nret['ymax']
    zmax = nret['zmax']
    # info about extra tables
    sub_table = nret['substrate']
    mig_table = nret['migration']
    # load migration_data and do initial analysis
    if mig_table > 0:
        mig_xyz = {} # dict by key cycle containing lists of xyz
        mig_orig_xyz = [] # original or final xyz of migrating soma (used if mig_xyz[][] == None)
        db_to_dict = {} # dict by key database migration column to mig_xyz column
        mcol = 0 # counts columns in mig_xyz
        last_cycle = 0
        mig_first_cycle = []
        for i in range(mig_table):
            try:
                cursor.execute("select * from migration_data" + str(i + 1))
            except Exception:
                print ("Error in nds_to_swc: expected migration_data" + str(i + 1) + " table")
                return
            mrets = cursor.fetchall()
            mig_first_cycle.append(mrets[0]['cycle']) # python starts at row 0
            # turn mrets data into a list of xyz
            num_cols = len(mrets[0])
            for row in mrets:
                mcycle = row['cycle']
                if mcycle > last_cycle:
                    last_cycle = mcycle
                if mcycle in mig_xyz:
                    mrow = mig_xyz[mcycle]
                else:
                    mrow = [] # new entry for mig_xyz
                for ncol in range(2,len(mrets[0]),3):
                    key = i * 1800 + ncol
                    #print (cycle,key)
                    if key not in db_to_dict:
                        db_to_dict[key] = mcol
                        mcol += 1
                    if row[ncol]:
                        xyz = Point(row[ncol],row[ncol+1],row[ncol+2])
                        mrow.append(xyz)
                    else:
                        mrow.append(None)
                mig_xyz[mcycle] = mrow
    # set colors: red and black are reserved
    colors = ['tab:blue','tab:green','tab:orange','tab:cyan','m','y','tab:brown',   'tab:purple','tab:pink','tab:gray','tab:olive']
    c_mapping = {} # map color to neuron_id
    c = 0 # index into colors
    # get all neuron names and set types and colors
    cursor.execute("select * from neuron_data")
    nrets = cursor.fetchall()
    names = {} # dict by neuron_id of all names
    lines = {} # store lines and spheres drawn by index front.index
    to_remove = {} # dictionary by time of removal of lines to be removed
    for row in nrets :
        name = row['name']
        nid = row['neuron_id']
        names[nid] = name
    # get all front_data
    cursor.execute("select * from front_data order by birth")
    frets = cursor.fetchall()
    # set data structures for spheres
    u = np.linspace(0, 2 * np.pi, num=20)
    v = np.linspace(0, np.pi, num=20)
    x_outer = np.outer(np.cos(u),np.sin(v))
    y_outer = np.outer(np.sin(u),np.sin(v))
    z_outer = np.outer(np.ones([20]),np.cos(v))
    # prepare to plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.azim = azim
    ax.elev = elev
    if (neuron_id > 0) and (front_id > 0): # focus around reference front
        all_fronts = False
        # find front
        failed = True
        for row in frets:
            if (row['neuron_id'] == neuron_id) and (row['front_id'] == front_id):
                frow = row
                failed = False
                break
        if failed:
            print ("Error in nds_interact: reference front",neuron_id,front_id,"not found")
            return
        # set axis: make sure all are equal except at borders
        orig = Point(frow['orig_x'],frow['orig_y'],frow['orig_z'])
        mcol = frow['migration']
        if mcol:
            if cycle > last_cycle:
                mcycle = last_cycle
            else:
                mcycle = cycle
            #print (cycle,mcol,db_to_dict[mcol],mig_xyz[mcycle][db_to_dict[mcol]])
            #mig_orig_xyz.append(orig)
            try:
                status, orig = _mig_soma_xyz(db_to_dict[mcol],mcycle,mig_xyz,\
                                            mig_orig_xyz,False)
            except:
                print ("_mig_soma_xyz failed",mcol,mcycle)
            end = orig
        else:
            end = Point(frow['end_x'],frow['end_y'],frow['end_z'])
        dir = end - orig
        mid = orig + dir / 2.
        xsize = end.x - orig.x
        ysize = end.y - orig.y
        zsize = end.z - orig.z
        #print (orig,end,mid,xsize,ysize,zsize)
        # find largest
        if xsize > ysize:
            if xsize > zsize:
                size = xsize + focus
            else:
                size = zsize + focus
        else:
            if ysize > zsize:
                size = ysize + focus
            else:
                size = zsize + focus
        size_2 = int(size / 2.) + 1
        axmin = max(xmin,mid.x - size_2)
        aymin = max(ymin,mid.y - size_2)
        azmin = max(zmin,mid.z - size_2)
        axmax = min(xmax,mid.x + size_2)
        aymax = min(ymax,mid.y + size_2)
        azmax = min(zmax,mid.z + size_2)
        #print (size_2,axmin,aymin,azmin,axmax,aymax,azmax)
    else: # use entire simulation volume
        all_fronts = True
        axmin = xmin
        aymin = ymin
        azmin = zmin
        axmax = xmax
        aymax = ymax
        azmax = zmax
    ax_volume = [[axmin,aymin,azmin],[axmax,aymax,azmax]]
    #print ("ax_volume",ax_volume)
    ax.set_xlim([axmin,axmax])
    ax.set_ylim([aymin,aymax])
    ax.set_zlim([azmin,azmax])
    # plot all the fronts that are (partially) inside the ax_volume
    if verbose:
        print ("color  cycle neuron                         front_id swc  orig                  end")
    for row in frets:
        birth = row['birth']
        if birth > cycle: # finish now
            break # out of for row loop
        death = row['death']
        if (death >= 0) and (death <= cycle): # deleted
            continue
        nid = row['neuron_id']
        fid = row['front_id']
        ename = names[nid]
        bname = row['branch']
        key = str(nid) + "_" + str(fid)
        shape = row['shape']
        if shape == 2:
            radius = row['radius'] * radius_scale
        else:
            radius = row['radius'] * sphere_scale
        orig = Point(row['orig_x'],row['orig_y'],row['orig_z'])
        mcol = row['migration']
        if mcol:
            mig_orig_xyz.append(orig)
            status, orig = _mig_soma_xyz(db_to_dict[mcol],cycle,mig_xyz,\
                                            mig_orig_xyz,True)
            end = orig
        else:
            end = Point(row['end_x'],row['end_y'],row['end_z'])
        if all_fronts or point_in_volume(orig,ax_volume) or \
            point_in_volume(end,ax_volume):
            if fid == 5003:
                sph_orig = orig
            if fid == front_id:
                col = 'tab:red'
            else:
                if nid not in c_mapping:
                    col = colors[c%len(colors)]
                    c_mapping[nid] = col
                    c = c + 1
                else:
                    col = c_mapping[nid]
            if verbose:
                out = col[4:].ljust(7) + str(birth).ljust(6) + ename.ljust(31)
                out += str(fid).ljust(9) + str(row['swc_type']).ljust(4)
                print (out,orig,end)
                if verbose == 2 and (fid != 5003):
                    print (" ".ljust(57),(orig-sph_orig).toPol_point(),\
                            (end-sph_orig).toPol_point())
            _draw_front(ax,lines,key,col,None,shape,row['swc_type'],False,orig,
                end,radius,birth,row['death'],to_remove,x_outer,y_outer,z_outer,\
                None,interact=True)
    return ax
    
def nds_get_color_dict(db_name,file_name,max_cycle=-1,neurons=[],color_scheme=0,verbose=1):
    """ Save the color table by neuron name to a text file.
    
    Parameters
    ----------
    db_name : string : name of NeuroDevSim database file.
    file_name : string : name of output text file.
    Optional :
    color_scheme : integer 0 - 1 : controls how colors change: 0: every neuron has a different color; 1: neurons of same type have same color, different types have different colors, default 0.
    verbose : integer 0-1 : print information about all fronts plotted (1), default 1.
    """

    if not os.path.isfile(db_name):
        print ("Error in nds_get_color_dict:",db_name,"not found")
        return
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # check color settings
    if (color_scheme < 0) or (color_scheme > 1):
        print ("Error in nds_get_color_dict: invalid color_scheme value, range is 0-1")
    # Get general data
    try:
        cursor.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in nds_get_color_dict: not a NeuroDevSim database")
        return
    nret = cursor.fetchone()
    version = nret['version']
    if (version < 93):
        print ("Error in nds_get_color_dict: wrong database version",version)
        return
    # define colors
    colors = ['tab:red','tab:blue','tab:green','tab:orange','tab:cyan','m','y',\
              'tab:brown','tab:purple','tab:pink','tab:gray','tab:olive']
    c_mapping = {} # map color to name of neuron or dendrite
    c = 0 # index into colors
    # get all neuron names and set types and colors
    cursor.execute("select * from neuron_data")
    rets = cursor.fetchall()
    names = {} # dict by neuron_id of all names
    name_types = {} # store neuron types by index neuron_name
    lines = {} # store lines and spheres drawn by index front.index
    c=0
    for row in rets:
        name = row['name']
        if neurons: # only plot specific neurons
            skip = True
            for neu_name in neurons:
                if name.startswith(neu_name):
                    skip = False
                    break
            if skip: # do not plot this one
                continue
        nid = row['neuron_id']
        names[nid] = name
        if version < 93.2:
            type = row['array_id']
        else:
            type = row['type_id']
        name_types[name] = type
        if color_scheme == 0: # color neurons different
            color = colors[c%len(colors)]
            c_mapping[name] = color
            c += 1
        else: # color neuron types different
            if type not in c_mapping:
                color = colors[c%len(colors)]
                c_mapping[type] = color
                c += 1
            else:
                color = c_mapping[type]
        if color_scheme < 2:
            if verbose > 0:
                if color.startswith('tab'):
                    print ("plotting:",name,"with color",color[4:])
                elif color == 'm':
                    print ("plotting:",name,"with color magenta")
                else:
                    print ("plotting:",name,"with color yellow")
    # output text file
    if not file_name.endswith(".txt"):
        file_name += ".txt"
    try:
        writer = open(file_name,"w")
    except:
        print ("Error in nds_get_color_dict: could not open",file_name,"as output")
        return
    for row in rets:
        name = row['name']
        if color_scheme == 0:
            to_write = name + " " + c_mapping[name] + "\n"
        else:
            to_write = name + " " + c_mapping[name_types[name]] + "\n"
        writer.write(to_write)
    writer.flush()
    
def nds_plot_data(db_name,attribute,num_plots=-1,select=[],\
                    min_cycle=0,max_cycle=-1):
    """ Generate a line plot of an attribute versus time.
    
    To select plotting of specific subsets use the *select* optional parameter. Data about available neurons and fronts can be obtained with the *nds_list_data* method. Alternatively, the *num_plots* optional parameter can be used to restrict the number of fronts plotted.
    
    To restrict the range of cycles plotted use the *min_cycle* and *max_cycle* optional parameters.
    
    Parameters
    ----------
    db_name : string : name of NeuroDevSim database file.
    attribute : string : name of attribute to plot. 
    Optional :
    max_cycle : integer -1 or >0 : stop plotting at this cycle, default -1: plot till end of simulation.
    min_cycle : integer >0 : start plotting at this cycle, default 0: plot from beginning.
    num_plots : integer -1 or >0 or [integer,integer]: restrict number of plots to value indicated. When a list is used it is interpreted as a range. Default: -1, plot all data.
    select : [DataID,] : list of neurons or fronts to plot defined by their DataID, default []: plot all entries in database table.
    """

    import matplotlib.pyplot as plt
    
    plt.plot(subplots=True)

    if not os.path.isfile(db_name):
        print ("Error in nds_plot_data:",db_name,"not found")
        return
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Get general data
    try:
        cursor.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in nds_plot_data: not a NeuroDevSim database")
        return
    nret = cursor.fetchone()
    version = nret['version']
    if (version < 93): 
        print ("Error in nds_plot_data: wrong database version",version)
        return
    if max_cycle < 0:
        max_cycle = nret['num_cycles']
    if min_cycle < 0:
        print ("Error in nds_plot_data: negative min_cycle",min_cycle)
        return
    if min_cycle >= max_cycle:
        print ("Error in nds_plot_data: min_cycle >= max_cycle",min_cycle, max_cycle)
        return
    if isinstance(num_plots,list):
        if num_plots[0] < 0:
            print ("Error in nds_plot_data: invalid num_plots: start of range should be >= 0",num_plots[0])
            return
        if num_plots[1] <= num_plots[0]:
            print ("Error in nds_plot_data: invalid num_plots: end of range should be larger than start",num_plots[1])
            return
        start_plot = num_plots[0]
        end_plot = num_plots[1]
    else:
        if (num_plots < -1) or (num_plots == 0):
            print ("Error in nds_plot_data: invalid num_plots",num_plots)
            return
        start_plot = 0
        if num_plots == -1:
            end_plot = 1000000 # very large number
        else:
            end_plot = num_plots
    subset = len(select) > 0
    table = attribute + "_data"
    try:
        cursor.execute("select * from " + table)
    except Exception:
        print ("Error in nds_plot_data: no data table for ",attribute)
        return
    drets = cursor.fetchall()
        
    # organize data per neuron or front, dictionary contains entries as list:
    #   [[cycles],[values]]
    #   for each cycle only the highest absolute value is stored.
    front_data = {}
    for dret in drets:
        front = DataID(dret['neuron_id'],dret['front_id']) # can also be a neuron
        key = front._key()
        cycle = dret['cycle']
        if (cycle < min_cycle) or (cycle > max_cycle):
            continue    # outside of range
        if subset and (front not in select): # skip this entry
            continue
        value = dret[attribute]
        if key in front_data:
            data = front_data[key]
            try: # test whether already data for this cycle
                cyc_ind = data[0].index(cycle) # find index of cycle
                if abs(value) > abs(data[1][cyc_ind]):
                    front_data[key][1][cyc_ind] = value # store larger value
            except: # no data for this cycle
                front_data[key][0].append(cycle)
                front_data[key][1].append(value)
        else: # make new entry
            front_data[key] = [[cycle],[value]]
            
    # next extract plotting data and plot
    n_plots = 0
    for key in front_data.keys():
        data = front_data[key]
        if len(data[0]) == 1: # single data point
            continue
        if n_plots < start_plot:
            n_plots += 1
            continue # skip these ones        print ("plot",n_plots, start_plot, end_plot)
        plt.plot(data[0],data[1],label=key)
        n_plots += 1
        if n_plots >= end_plot:
            break # done plotting
    plt.title(attribute + ' versus cycle')
    plt.xlabel('Cycle')
    plt.ylabel(attribute)
    plt.legend()
    plt.show()

# draw basic figure and save for nm_plot or nm_movie, controlled by outtype
#    0: draw in window
#    1: save pdf file
#    2: save as movie
#    3: save both pdf file and movie
#   actual drawing is done in _draw_frames
def _draw_figure(conn,nret,out_name,outtype,start,last,wire,azim,elev,box,\
                    soma_black,no_axis,scale_axis,axis_ticks,color_scheme,\
                    show_retraction,verbose,color_data,dpi,neurons,\
                    version):
    import matplotlib
    if outtype > 0:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as manimation
    from mpl_toolkits.mplot3d import Axes3D
    
    cursor = conn.cursor()
    # Get volume and cycle data
    if box:
        xmin,ymin,zmin = box[0]
        xmax,ymax,zmax = box[1]
    else:
        xmin = nret['xmin']
        ymin = nret['ymin']
        zmin = nret['zmin']
        xmax = nret['xmax']
        ymax = nret['ymax']
        zmax = nret['zmax']

    if outtype == 2: # make a movie
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title=out_name, artist='Matplotlib',
                comment='NeuroDevSim simulation')
        writer = FFMpegWriter(fps=5, metadata=metadata)
    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if no_axis:
        ax.set_axis_off()
    else:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        if not axis_ticks:
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            ax.axes.zaxis.set_ticks([])
        if scale_axis:
            if len(scale_axis) != 3:
                print (Fore.RED + "Error! ignoring incorrect scale_axis option, should be: [xscale,yscale,zscale]",Fore.RESET)
            else:
                scale_axis.append(1.0)
                ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax),np.diag(scale_axis))
    ax.azim = azim
    ax.elev = elev
    if verbose > 0:
        print ("plotting with azim =",ax.azim,"and elev =",ax.elev)
    #ax.set_aspect('equal')
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.set_zlim([zmin,zmax])

    # info about extra tables
    sub_table = nret['substrate']
    mig_table = nret['migration']
    if outtype >= 2: # write movie
        with writer.saving(fig,out_name+".mp4",dpi):
            _draw_frames(cursor,outtype,fig,ax,writer,start,last,wire,azim,elev,box\
                    ,soma_black,no_axis,scale_axis,color_scheme,show_retraction,\
                    verbose,color_data,sub_table,mig_table,neurons,version)
    else:
        _draw_frames(cursor,outtype,fig,ax,None,start,last,wire,azim,elev,box,\
                    soma_black,no_axis,scale_axis,color_scheme,show_retraction,\
                    verbose,color_data,sub_table,mig_table,neurons,version)

    conn.close() # close database file
    if outtype == 0:
        fig.canvas.draw()
        plt.show()  # show window
    elif (outtype == 1) or (outtype == 3):
        plt.savefig(out_name+".pdf")  # save to pdf file
        plt.close()
    else:
        return      # movie

# draws or deletes the front
def _draw_front(ax,lines,key,col,col2,shape,swc_type,soma_black,orig,end,radius,\
                birth,death,to_remove,x_outer,y_outer,z_outer,color_obs,interact=False):
    if shape == 1: # spherical
        # plot the soma
        if (swc_type == 1) and soma_black:
            col = 'k'
        x = orig.x + (radius * x_outer)
        y = orig.y + (radius * y_outer)
        z = orig.z + (radius * z_outer)
        if (death < 0) or interact:
            if col2:
                lines[key] = (col2,ax.plot_surface(x,y,z,rstride=2,cstride=2,\
                                color=col,linewidth=0,antialiased=False,shade=False))
                lines[key][1].set_color(col2)
                if color_obs:
                    color_obs.append(key)
            else:
                lines[key] = (col,ax.plot_surface(x,y,z,rstride=2,cstride=2,\
                                color=col,linewidth=0,antialiased=False,shade=False))
        else: # retracted front
            lines[key] = ("k",ax.plot_surface(x,y,z,rstride=2,cstride=2,\
                color="k",linewidth=0,antialiased=False,shade=False))
            if death not in to_remove:
                to_remove[death] = []
            to_remove[death].append(key)
    else: # cylindrical
        if (death < 0) or interact:
            if col2:
                lines[key] = (col2,ax.plot([orig.x,end.x],[orig.y,end.y],\
                              zs=[orig.z,end.z],linewidth=2*radius,color=col))
                lines[key][1][0].set_color(col2)
                if color_obs:
                    color_obs.append(key)
            else:
                lines[key] = (col,ax.plot([orig.x,end.x],[orig.y,end.y],\
                              zs=[orig.z,end.z],linewidth=2*radius,color=col))
        else: # retracted front
            if col2:
                lines[key] = (col2,ax.plot([orig.x,end.x],[orig.y,end.y],\
                              zs=[orig.z,end.z],linewidth=2*radius,color=col))
                lines[key][1][0].set_color(col2)
                if color_obs:
                    color_obs.append(key)
            else:
                lines[key] = ("k",ax.plot([orig.x,end.x],[orig.y,end.y],\
                          zs=[orig.z,end.z],linewidth=2*radius,color="k"))
            if swc_type == 12: # filipod, remove one cycle later
                death += 1
            if death not in to_remove:
                to_remove[death] = []
            to_remove[death].append(key)

# colors plot according to attribute data at specific cycle
def _draw_attribute(cursor,color_attrib,cf_name,ax,lines,cycle,colors,color_min,color_scale,color_obs,color_nid):
    import mpl_toolkits.mplot3d.art3d as mplt
    if cycle == 0:
        return
    for key in color_obs:
        under = key.rfind('_')
        if under < 0 :
            print ("Error in nsd_plot: wromg key",key)
            return
        nid = int(key[0:under])
        fid = int(key[under + 1:])
        if cf_name: # get value from table:
            try:
                #print (key,nid,fid,cycle)
                cursor.execute("select * from " + cf_name + " where neuron_id=? and front_id=? and cycle=?",(nid,fid,cycle))
                arow = cursor.fetchone()
                if arow:
                    value = arow[color_attrib]
                    col = colors(max(0.,min(1.,(value - color_min) * color_scale)))
                else:
                    if nid in color_nid:
                        col = colors(0.)
                    else:
                        col = 'k'
            except:
                if nid in color_nid:
                    col = colors(0.)
                else:
                    col = 'k'
                #print ('except',key,nid,fid,value,col)
        else:
            try:
                #print ("no cf_name",key,nid,fid)
                cursor.execute("select * from front_data  where neuron_id=? and front_id=?",(nid,fid))
                arow = cursor.fetchone()
                value = arow[color_attrib]
                col = colors(max(0.,min(1.,(value - color_min) * color_scale)))
            except:
                #pass
                col = colors(0.)
                print ("Bug in nsd_plot: " + color_attrib + " not found to plot")

        item = lines[key][1]
        if isinstance(item,mplt.Poly3DCollection): # a sphere
            item.set_color(col)
        else: # a line
            item[0].set_color(col)

# do all checking and drawing related to soma migration
def _draw_migration(outtype,writer,mig_soma,prev_soma,birth,last,start,mrets,\
                mig_first_cycle,mig_max_row,ax,lines,soma_black,\
                color_scheme,to_remove,x_outer,y_outer,z_outer,cursor,\
                color_attrib,cf_name,colors,col_min,col_scale,color_obs,color_nid):
    mig_now = [] # list of [key,mr_ind,mrow,mr_col,radius] that need to be plotted now
    # check all migrating soma whether they still migrate this cycle
    to_del = [] # keys to delete
    for key in mig_soma.keys():
        mcol = mig_soma[key][0]
        mr_ind = mcol // 1800
        mr_col = mcol % 1800
        mrow = birth - mig_first_cycle[mr_ind]
        if mrow < 0:
            continue
        #print (key,mr_ind,mig_first_cycle[mr_ind],birth,mcol,mig_max_row[mcol])
        if (mrow > mig_max_row[mcol]) or (birth==last): # no data for this and later cycles or last cycle plotted
            if outtype < 2: # plot soma in final location
                if mrow > 0:
                    mig_now.append([key,mr_ind,mrow - 1,mr_col,mig_soma[key][1]])
                else: # never moved
                    print ("Error in nsd_plot: mig_min_row error")
                    return
            to_del.append(key)
        elif outtype >= 2: # store in mig_now
            mig_now.append([key,mr_ind,mrow,mr_col,mig_soma[key][1]])
    for key in to_del:
        del mig_soma[key]
    # movies: remove previous migrating somata and draw frame: always execute this
    if outtype >= 2: # movies
        # remove moved soma drawing from previous cycle: they have
        #  'hidden' gaps in image
        to_del = [] # keys to delete
        for key in prev_soma.keys():
            if prev_soma[key]:
                try:
                    prev_soma[key].remove()
                except:
                    print ("Error in nsd_plot: cannot remove soma for cycle",prev_birth)
            if key not in mig_soma: # final remove
                to_del.append(key)
        for key in to_del:
            del prev_soma[key]
        # write a frame of previous cycle (if not before starting frame)
        if birth > start:
            if color_scheme == 3: # first recolor according to concentration
                _draw_attribute(cursor,color_attrib,cf_name,ax,lines,birth-1,colors,\
                                col_min,col_scale,color_obs,color_nid)
            writer.grab_frame()
    # now draw all migrating somata
    empty = Point(0.,0.,0.)
    for entry in mig_now:
        #print(entry)
        key = entry[0]
        mr_ind = entry[1]
        mrow = entry[2]
        mr_col = entry[3]
        #print (key,mr_ind,mrow,mr_col,mrets[mr_ind][mrow][mr_col],mrets[mr_ind][mrow][mr_col+1],mrets[mr_ind][mrow][mr_col+2])
        if mrets[mr_ind][mrow][mr_col]: # real entry
            if key in lines and lines[key][1]:
                prev_soma[key] = lines[key][1] # store previous location to remove later
            orig = Point(mrets[mr_ind][mrow][mr_col],mrets[mr_ind][mrow][mr_col+1],\
                        mrets[mr_ind][mrow][mr_col+2])
            _draw_front(ax,lines,key,lines[key][0],None,1,1,soma_black,orig,empty,
                        entry[4],birth,-1,to_remove,x_outer,y_outer,z_outer,color_obs)
        else:
            prev_soma[key] = None # do not remove last position
# does the actual drawing for nm_plot and nm_movie, needs to be separate routine
def _draw_frames(cursor,outtype,fig,ax,writer,start,last,wire,azim,elev,box,\
                 soma_black,no_axis,scale_axis,color_scheme,\
                 show_retraction,verbose,color_data,sub_table,mig_table,\
                 neurons,version):
    # set colors
    color_obs = None
    if color_scheme < 3:
        colors = ['r','b','g','tab:orange','c','m','y','tab:red','tab:blue',\
                  'tab:green','tab:brown','tab:purple','tab:pink','tab:gray','tab:olive']
        c_mapping = {} # map color to name of neuron or dendrite
        c = 0 # index into colors
        # initialize unused parameters
        color_attrib = None
        cf_name = None
        color_nid = None
        color_min = None
        color_scale = None
    else:
        color_attrib = str(color_data[0])
        # is this a user saved attribute?
        cf_name = color_attrib + "_data"
        try:
            cursor.execute("select * from " + cf_name)
            arows = cursor.fetchall()
            # make a set of neuron_id stored in attribute table
            color_nid = set()
            for row in arows:
                color_nid.add(int(row['neuron_id']))
        except Exception:
            cf_name = None # is front attribute
        color_min = color_data[1]
        color_scale = 1.0 / (color_data[2] - color_min)
        import matplotlib.cm as cm
        colors = cm.get_cmap('rainbow')
        if (outtype >= 2): # movie -> keep list of keys for drawn objects that change color
            color_obs = []
    # set data structures for spheres
    u = np.linspace(0, 2 * np.pi, num=20)
    v = np.linspace(0, np.pi, num=20)
    x_outer = np.outer(np.cos(u),np.sin(v))
    y_outer = np.outer(np.sin(u),np.sin(v))
    z_outer = np.outer(np.ones([20]),np.cos(v))
    # deal with substrate
    if sub_table:
        # plot all substrate:
        try:
            cursor.execute("select * from substrate_data order by id")
        except Exception:
            print ("Error in nds_plot: expected substrate_data table")
            return
        subs = cursor.fetchall()
        subs_row = 0
        for row in subs:
            if color_scheme < 3:
                name = row['name']
                if name not in c_mapping:
                    c_mapping[name] = colors[c%len(colors)]
                    c = c + 1
                col = color=c_mapping[name]
            else:
                col = 'tab:blue'
            if row['birth'] == last: # stop plotting
                break
            ax.scatter(row['x'],row['y'],row['z'],color=col,marker='+')
    # load migration_data and do initial analysis
    if mig_table > 0:
        mrets = []
        mig_min_row = {} # dictionary by key first column: first row with coordinates
        mig_max_row = {} # dictionary by key first column: last row with coordinates
        mig_first_cycle = [] # cycle of first row in each table
        for i in range(mig_table):
            try:
                cursor.execute("select * from migration_data" + str(i + 1))
            except Exception:
                print ("Error in nds_to_swc: expected migration_data" + str(i + 1) + " table")
                return
            mr = cursor.fetchall()
            mrets.append(mr)
            mig_first_cycle.append(mr[0]['cycle']) # python starts at row 0
            # analyze all soma: get first and last valid entry
            num_cols = (len(mr[0]) - 2) // 3
            ncol = 2 # first x column
            key = i * 1800 + 2 # column number used in database
            max_rows = len(mr)
            # fill mig_min_row and mig_max_row
            for c in range(num_cols):
                for nrow in range(max_rows):
                    x = mr[nrow][ncol]
                    if x:
                        if key not in mig_min_row: # first valid entry
                            mig_min_row[key] = nrow
                        if key in mig_max_row: # later entry -> overwrite
                            mig_max_row[key] = nrow
                    if (not x) and (key in mig_min_row) and \
                        (key not in mig_max_row): # previous row was last valid entry
                        mig_max_row[key] = nrow - 1
                # take care of edge cases
                if key not in mig_min_row: # not a single valid entry
                    mig_min_row[key] = None
                    mig_max_row[key] = None
                elif key not in mig_max_row: # valid till end
                    mig_max_row[key] = max_rows - 1
                #print (key,mig_min_row[key],mig_max_row[key])
                ncol += 3
                key += 3
    else:
        mrets = None
    # get all neuron names and set types and colors
    cursor.execute("select * from neuron_data")
    rets = cursor.fetchall()
    names = {} # dict by neuron_id of all names
    lines = {} # store lines and spheres drawn by index front.index
    if color_scheme == -1:
        f = open(color_data,'r') # color_data contains neuron_colors
        # read the data into c_mapping
        flines = f.readlines()
        for line in flines:
            space = line.rfind(' ')
            if space < 0:
                print ("Error in nsd_plot: neuron_colors file contains line without separating space:",line)
                return
            name = line[0 : space]
            color = line[space + 1 : -1]
            #print (name,color)
            c_mapping[name] = color
        #close(f)
        for row in rets:
            name = row['name']
            if name not in c_mapping:
                print ("Error in nsd_plot: incomplete neuron_colors file, no data for:",name)
                return
            nid = row['neuron_id']
            names[nid] = name
    else:
        name_types = {} # store neuron types by index neuron_name
        c=0
        for row in rets:
            name = row['name']
            if neurons: # only plot specific neurons
                skip = True
                for neu_name in neurons:
                    if name.startswith(neu_name):
                        skip = False
                        break
                if skip: # do not plot this one
                    continue
            nid = row['neuron_id']
            names[nid] = name
            if version < 93.2:
                type = row['array_id']
            else:
                type = row['type_id']
            name_types[name] = type
            if color_scheme < 3:
                if color_scheme != 1: # color neurons different
                    color = colors[c%len(colors)]
                    c_mapping[name] = color
                    c += 1
                else: # color neuron types different
                    if type not in c_mapping:
                        color = colors[c%len(colors)]
                        c_mapping[type] = color
                        c += 1
                    else:
                        color = c_mapping[type]
                if color_scheme < 2:
                    if verbose > 0:
                        if color.startswith('tab'):
                            print ("plotting:",name,"with color",color[4:])
                        elif color == 'b':
                            print ("plotting:",name,"with color bright blue")
                        elif color == 'c':
                            print ("plotting:",name,"with color cyan")
                        elif color == 'g':
                            print ("plotting:",name,"with color bright green")
                        elif color == 'm':
                            print ("plotting:",name,"with color magenta")
                        elif color == 'r':
                            print ("plotting:",name,"with color bright red")
                        else:
                            print ("plotting:",name,"with color yellow")
                else:
                    soma_black = False
                    if verbose > 0:
                        print ("plotting:",name,"with variable color")
            else: # color_scheme 3
                if verbose > 0:
                    if cf_name:
                        if nid in color_nid:
                            print ("plotting:",name,"with variable color")
                        else:
                            print ("plotting:",name,"with color black")
                    else:
                        print ("plotting:",name,"with color purple")
    # get all front_data and draw row by row
    cursor.execute("select * from front_data order by birth")
    rets = cursor.fetchall()
    to_remove = {} # dictionary by time of removal of lines to be removed
    mig_soma = {} # dictionary by key of all currently migrating soma mcol
    prev_soma = {} # dictionary by key of migrated somata lines that need to be removed
    new_cycle = -1
    prev_birth = -1
    col = 'k' # default value for color_scheme == 3
    col2 = None # only changed by color_scheme == 3
    for row in rets:
        birth = row['birth']
        if birth > last: # finish now
            break # out of for row loop
        neuron_id = row['neuron_id']
        if neuron_id in names: # plot this neuron
            ename = names[neuron_id]
            bname = row['branch']
            front_id = row['front_id']
            key = str(neuron_id) + "_" + str(front_id)
            shape = row['shape']
            swc_type = row['swc_type']
            # set radius
            if wire and shape == 2:
                radius = 0.1
            else:
                radius = row['radius']
            mcol = row['migration']
            if mcol: # migrating soma
                # check whether it really migrated
                if (mcol not in mig_min_row) or (mcol not in mig_max_row):
                    print ("Error in nsd_plot: migration column mismatch",mcol)
                    return
                if mig_min_row[mcol] == None: # no data for this neuron
                    mcol = 0
                else:
                    mig_soma[key] = [mcol,radius]
                mr_ind = mcol // 1800
            # get color
            if color_scheme <= 0: # color neurons different
                col = c_mapping[ename]
            elif color_scheme == 1: # color neuron types different
                col = c_mapping[name_types[ename]]
            elif color_scheme == 2: # color varies with branch_name
                if swc_type != 1:
                    if not (bname in c_mapping):
                        c_mapping[bname] = colors[c%len(colors)]
                        c = c + 1
                    col = c_mapping[bname]
                else:
                    col = c_mapping[ename]
            elif color_scheme == 3:
                col = 'k' # will be changed
                if cf_name: # get value from table:
                    if (outtype >= 2):
                        cyc = birth
                    else: # pdf
                        cyc = last
                    try:
                        cursor.execute("select * from " + cf_name + " where neuron_id=? and front_id=? and cycle=?",(neuron_id,front_id,cyc))
                        arow = cursor.fetchone()
                        value = arow[color_attrib]
                        col2 = colors(max(0.,min(1.,(value - color_min) * color_scale)))
                    except:
                        if neuron_id in color_nid:
                            col2 = colors(0.)
                        else:
                            col2 = 'k'
                else:
                    try:
                        value = row[color_attrib]
                        col2 = colors(max(0.,min(1.,(value - color_min) * color_scale)))
                    except:
                        #pass
                        col2 = colors(0.)
                        print ("Bug in nsd_plot: " + color_attrib + " not found to plot")
            else:
                print ("Bug in nsd_plot: wrong color_scheme",color_scheme)
                return
        # if transition to new cycle: take care of migration and draw movie frames
        if birth > prev_birth:
            for cycle in range(prev_birth + 1, birth + 1): # in case of absent cycles
                # plot migrating soma
                if mig_table > 0:
                    _draw_migration(outtype,writer,mig_soma,prev_soma,cycle,last,start,mrets,\
                                    mig_first_cycle,mig_max_row,ax,lines,soma_black,\
                                    color_scheme,to_remove,x_outer,y_outer,z_outer,cursor,\
                                    color_attrib,cf_name,colors,color_min,color_scale,color_obs,color_nid)
                # movies: draw frame
                elif (outtype >= 2) and (cycle >= start): # movies
                    # write a frame of previous cycle (if not before starting frame)
                    if color_scheme == 3: # first recolor according to concentration
                        _draw_attribute(cursor,color_attrib,cf_name,ax,lines,birth-1,colors,\
                                        color_min,color_scale,color_obs,color_nid)
                    writer.grab_frame()
                if show_retraction and (cycle < birth): # remove for intermediate cycles
                    if cycle in to_remove:
                        for key in to_remove[cycle]:
                            try:
                                ax.lines.remove(lines[key][1][0])
                            except:
                                print ("Error in nsd_plot: cannot remove line for cycle",cycle)
                                continue
                        del to_remove[cycle]        # birth corresponds to current cycle
        if (birth % 5 == 0) and (birth != new_cycle) and (verbose > 0):
            print ('cycle >',birth,'<')
        new_cycle = birth # update new_cycle
        if neuron_id in names: # plot this neuron
            # check whether to plot migrating soma before movement for movies:
            if (mcol == 0) or ((outtype >= 2) and \
                            (birth - mig_first_cycle[mr_ind] < mig_min_row[mcol])):
                orig = Point(row['orig_x'],row['orig_y'],row['orig_z'])
                end = Point(row['end_x'],row['end_y'],row['end_z'])
                _draw_front(ax,lines,key,col,col2,shape,row['swc_type'],soma_black,orig,
                            end,radius,birth,row['death'],to_remove,x_outer,y_outer,z_outer,color_obs)
            else:
                lines[key] = (col,None)
        # do we need to retract something?
        if show_retraction:
            if birth in to_remove:
                #print ("removing",cycle)
                for key in to_remove[birth]:
                    try:
                        ax.lines.remove(lines[key][1][0])
                    except:
                        print ("Error in nsd_plot: cannot remove line for cycle",birth)
                        continue
                del to_remove[birth]
        # update prev_birth
        prev_birth = birth
    # no more new fronts, process later migrations and removals
    if (len(mig_soma) > 0) or show_retraction:
        while (birth <= last):
            birth += 1
            if mig_table > 0:
                _draw_migration(outtype,writer,mig_soma,prev_soma,birth,last,start,mrets,\
                                mig_first_cycle,mig_max_row,ax,lines,soma_black,\
                                color_scheme,to_remove,x_outer,y_outer,z_outer,cursor,\
                                color_attrib,cf_name,colors,color_min,color_scale,color_obs,color_nid)
            if birth % 5 == 0:
                print ('cycle >',birth,'<')
            if birth in to_remove:
                for key in to_remove[birth]:
                    try:
                        ax.lines.remove(lines[key][1][0])
                    except:
                        print ("Error in nsd_plot: cannot remove line for cycle",birth)
                        continue
                del to_remove[birth]
            if (outtype >= 2) and (birth >= start): # write a frame
                if color_scheme == 3: # first recolor according to concentration
                    _draw_attribute(cursor,color_attrib,cf_name,ax,lines,birth-1,colors,\
                                    color_min,color_scale,color_obs,color_nid)
                writer.grab_frame()
    # do we need to remove more somata at the end?
    for keys in prev_soma.keys():
        try:
            prev_soma[key].remove()
        except:
            print ("Error in nsd_plot: cannot remove soma for cycle",prev_birth)

### DATABASE REPORTING FUNCTIONS

def nds_compare_files(db_name1,db_name2,max_cycle=-1,\
                      swc_types=[],verbose=1,print_all=True):
    """ Compare two NeuroDevSim databases to each other.
    
    Outputs a list of all differences between the two files. Does not check branch names or shape.
    Present version ignores soma migration.
    
    Parameters
    ----------
    db_name1 : string : name of NeuroDevSim database file.
    db_name2 : string : name of NeuroDevSim database file.
    Optional :
    max_cycle : integer > 0 : stop output at this cycle, default: all cycles (-1).
    swc_types : list of integer : only analyze differences for specified swc_types, default: all swc_types.
    verbose : integer 0-2 : control output, default 1: print output filename.
    print_all : boolean : print all differences between fronts or only first difference, default True.
    """
    if not os.path.isfile(db_name1):
        print ("Error in nm_swc_files:",db_name1,"not found")
        return
    if not os.path.isfile(db_name2):
        print ("Error in nm_swc_files:",db_name2,"not found")
        return
    conn1 = sqlite3.connect(db_name1)
    conn1.row_factory = sqlite3.Row
    cursor1 = conn1.cursor()
    conn2 = sqlite3.connect(db_name2)
    conn2.row_factory = sqlite3.Row
    cursor2 = conn2.cursor()
    # Check versions
    try:
        cursor1.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in nds_to_swc: not a NeuroDevSim database")
        return
    nret1 = cursor1.fetchone()
    version = nret1['version']
    if (version < 93):
        print ("Error in nds_to_swc: wrong database version",db_name1,version)
        return
    try:
        cursor2.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in nds_to_swc: not a NeuroDevSim database")
        return
    nret2 = cursor2.fetchone()
    version = nret2['version']
    if (version < 93):
        print ("Error in nds_to_swc: wrong database version",db_name2,version)
        return
    # Get all neuron names
    cursor1.execute("select * from neuron_data")
    rets1 = cursor1.fetchall()
    names1 = {} # dict by name of all neuron_ids
    ids1 = {} # dict by neuron_ids of all names
    c=0
    for row in rets1:
        names1[row['name']] = row['neuron_id']
        ids1[row['neuron_id']] = row['name']
    cursor2.execute("select * from neuron_data")
    rets2 = cursor2.fetchall()
    names2 = {} # dict by name of all neuron_ids
    ids2 = {} # dict by neuron_ids of all names
    c=0
    for row in rets2:
        names2[row['name']] = row['neuron_id']
        ids2[row['neuron_id']] = row['name']
    match = True
    for name in names1:
        if name not in names2:
            print ("Neuron",name,"not in",db_name2)
            match = False
    for name in names2:
        if name not in names1:
            print ("Neuron",name,"not in",db_name1)
            match = False
    if (verbose > 1) and match:
        print ("Same neurons present in both databases")
    print1 = not print_all
    # Compare each neuron in the data
    for name in names1.keys():
        if name not in names2:
            continue
        #if not name.startswith("gr"):
            #continue
        cursor1.execute("select * from front_data where neuron_id=? order by birth",(names1[name],) )
        rets1 = cursor1.fetchall()
        max1 = len(rets1)
        cursor2.execute("select * from front_data where neuron_id=? order by birth",(names2[name],) )
        rets2 = cursor2.fetchall()
        max2 = len(rets2)
        ind2 = 0
        matches2 = [] # list of ind2 that have been matched
        if max1 != max2:
            match = False
            print (name,"different number of fronts",max1,max2)
        elif (verbose > 1):
            print ("checking",name,max1,max2)
        """
        for ind1 in range(max1):
            row1 = rets1[ind1]
            row2 = rets2[ind1]
            print ("db1",row1['neuron_id'],row1['front_id'],row1['parent_id'],\
                    row1['swc_type'],row1['orig_x'],row1['orig_y'],row1['orig_z'],\
                    row1['end_x'],row1['end_y'],row1['end_z'],row1['birth'])
            print ("db2",row2['neuron_id'],row2['front_id'],row2['parent_id'],\
                    row2['swc_type'],row2['orig_x'],row2['orig_y'],row2['orig_z'],\
                    row2['end_x'],row2['end_y'],row2['end_z'],row2['birth'])
        break
        """
        # get birth indexes for rets2:
        birth2 = 0
        births2 = {} # dict by birth containing starting index
        for ind2 in range(max2):
            row2 = rets2[ind2]
            birth = row2['birth']
            if birth > birth2:
                births2[birth] = ind2
                birth2 = birth
        for ind1 in range(max1):
            row1 = rets1[ind1]
            if swc_types and (row1['swc_type'] not in swc_types):
                continue # ignore
            row2 = rets2[ind2]
            fid1 = row1['front_id']
            birth1 = row1['birth']
            if row1 == row2: # perfectly matching rows
                matches2.append(ind2)
                ind2 += 1
                continue
            elif (birth1 in births2): # try to find matching row2
                # fid may be different so we need to compare on properties
                # assume that same swc_type, origin and birth is a probable homolog
                # but note that this may be true for multiple children of same
                # parent...
                if (birth1 + 1) in births2:
                    bmax = births2[birth1 + 1]
                else:
                    bmax = max2
                swc1 = row1['swc_type']
                or1x = row1['orig_x']
                or1y = row1['orig_y']
                or1z = row1['orig_z']
                targets = []
                for ind2 in range(births2[birth1],bmax):
                    row2 = rets2[ind2]
                    if (swc1 == row2['swc_type']) and (or1x == row2['orig_x']) \
                        and (or1y == row2['orig_y']) and (or1z == row2['orig_z']):
                        targets.append(ind2)
                #print (name,births2[birth1],bmax,targets)
                if targets: # found some candidates
                    if len(targets) == 1: # single candidate -> compare
                        row2 = rets2[targets[0]]
                        fid2 = row2['front_id']
                        matches2.append(targets[0])
                        if _print_diffs(name,fid1,fid2,row1,row2,True,print1):
                            match = False
                    else: # more candidates -> look for matching end
                        for ind2 in targets:
                            row2 = rets2[ind2]
                            if (row1['end_x'] == row2['end_x']) and \
                               (row1['end_y'] == row2['end_y']) and \
                               (row1['end_z'] == row2['end_z']): # match
                                fid2 = row2['front_id']
                                matches2.append(ind2)
                                if _print_diffs(name,fid1,fid2,row1,row2,False,print1):
                                    match = False
                else:
                    match = False
                    print ("cycle",birth1,"no match found for front",name,fid1,"with swc_type",\
                            swc1,"in",db_name2)
        for ind2 in range(max2):
            if ind2 not in matches2:
                row2 = rets2[ind2]
                if swc_types and (row2['swc_type'] not in swc_types):
                    continue # ignore
                match = False
                print ("cycle",birth1,"no match found for front",name,row2['front_id'],\
                       "with swc_type",row2['swc_type'],"in",db_name1)
    if (verbose > 1) and match:
        print ("Same fronts present in both databases")

def nds_summary(db_name):
    """ Prints summary of available neuron data from a NeuroDevSim simulation database.
    
    Parameters
    ----------
    db_name: string : name of raw SQL NeuroDevSim output database file.
    """
    if not os.path.isfile(db_name):
        print ("Error in nds_summary:",db_name,"not found")
        return
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Get volume data
    try:
        cursor.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in nds_summary: not a NeuroDevSim database")
        return
    nret = cursor.fetchone()
    version = nret['version']
    if (version < 93): 
        print ("Error in nds_summary: wrong database version",version)
        return
    version = float(version)/100.0
    print ("NeuroDevSim %.3f" % (version),"simulation:")
    print ("  simulation volume: [%5.0f, %5.0f, %5.0f] - [%5.0f, %5.0f, %5.0f])" %\
        (nret['xmin'],nret['ymin'],nret['zmin'],nret['xmax'],nret['ymax'],nret['zmax']))
    # get all neuron names
    cursor.execute("select * from neuron_data")
    rets = cursor.fetchall()
    neuron_data = [] # list of [names,num_fronts,num_synapses)
    c=0
    for row in rets:
        neuron_data.append([row['name'],row['num_fronts'],row['num_synapses']])
    num = len(neuron_data)
    if num > 1:
        print (" ",nret['num_cycles'],"cycles,",num,"neurons:")
    else:
        print (" ",nret['num_cycles'],"cycles, 1 neuron:")
    for data in neuron_data:
        if data[2] == 0: # no synapses
            print ("    ",data[0],":",data[1],"fronts")
        else:
            print ("    ",data[0],":",data[1],"fronts",",",data[2],"synapses")

def nds_cycles(db_name,max_cycle=-1,by_type=False):
    """ Prints summary of how many fronts were made each cycle from a NeuroDevSim simulation database.
    
    Parameters
    ----------
    db_name: string : name of raw SQL NeuroDevSim output database file.
    Optional:
    by_type : boolean : split totals by neuron_type, default False.
    max_cycle : integer > 0 : last cycle to print data for, default -1 (all cycles).
    """
    if not os.path.isfile(db_name):
        print ("Error in nds_cycles:",db_name,"not found")
        return
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Get volume data
    try:
        cursor.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in nds_cycles: not a NeuroDevSim database")
        return
    nret = cursor.fetchone()
    version = nret['version']
    if (version < 90): 
        print ("Error in nds_cycles: wrong database version",version)
        return
    if max_cycle < 0: # data will be output till max_cycle (included)
        max_cycle = nret['num_cycles']
    pversion = float(version)/100.0
    print ("NeuroDevSim %.3f" % (pversion),"simulation:")
    if by_type: # get neuron_type data
        types = [] # list of all types
        idtypes = [-1] # neuron_type by neuron_id, first not used
        if version >= 93:
            cursor.execute("select * from neuron_data")
        else:
            cursor.execute("select * from names_data")
        nrets = cursor.fetchall()        
        for row in nrets:
            if version < 93.2:
                type = row['array_id']
            else:
                type = row['type_id']
            if type not in types:
                types.append(type)
            idtypes.append(type - 1)
        n_types = len(types)
        space = "  "
    else:
        n_types = 1
        space = " "
    # get all fronts
    if version >= 93:
        cursor.execute("select * from front_data")
    else:
        cursor.execute("select * from neuron_data")
    rets = cursor.fetchall()
    cycle = 0
    nf = [0] * n_types # number of fronts by type
    type = 0 # default for by_types==False
    bug = False
    for row in rets:
        cur_cycle = row['birth']
        if (cycle > 0) and (cur_cycle == 0):
            if not bug:
                text = "cycle " + str(cycle) + ": "
                for n in range(n_types):
                    text += str(nf[n]) + space
                    nf[n] = 0 # reset
                print (text,"new fronts") # print data previous cycle
                if by_type:
                    type = idtypes[row['neuron_id']]
                nf[type] = 1
                cur_cycle = cycle + 1 # deal with all neurons born at cycle 0 bug
                cycle = cycle + 1
                bug = True
            else:
                cur_cycle = cycle # deal with all neurons born at cycle 0 bug
            continue
        else:
            bug = False
        if cycle == cur_cycle:
            if by_type:
                type = idtypes[row['neuron_id']]
            nf[type] += 1
        else: # new cycle
            text = "cycle " + str(cycle) + ": "
            for n in range(n_types):
                text += str(nf[n]) + space
                nf[n] = 0 # reset
            print (text,"new fronts") # print data previous cycle
            cycle = cur_cycle
            if by_type:
                type = idtypes[row['neuron_id']]
            nf[type] = 1
            if cycle > max_cycle:
                break
    
def nds_neuron(db_name,neuron_name,by_swc_type=False,front_ids=False,sizes=False):
    """ Prints information about a single neuron from a NeuroDevSim simulation database.
    
    By default prints database neuron_id and number of fronts, which can be sorted by *swc_type*. Can also print all front_ids.
    
    Parameters
    ----------
    db_name: string : name of raw SQL NeuroDevSim output database file.
    neuron_name: string : name of a neuron in the database, can be a wildcard.
    Optional:
    by_swc_type : boolean : print number of fronts for each swc_type, default False.
    front_ids : boolean : print also all front ids with swc type and birth/death, default False.
    sizes : boolean : print also front sizes, requires ``front_ids=True``, default False.
    """
    if not os.path.isfile(db_name):
        print ("Error in nds_neuron:",db_name,"not found")
        return
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Get volume data
    try:
        cursor.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in nds_neuron: not a NeuroDevSim database")
        return
    nret = cursor.fetchone()
    version = nret['version']
    if (version < 88): 
        print ("Error in nds_neuron: wrong database version",version)
        return
    # get all neuron names
    if version >= 93:
        cursor.execute("select * from neuron_data")
    else:
        cursor.execute("select * from names_data")
    rets = cursor.fetchall()
    names = {} # dict by neuron_id of all names
    for row in rets:
        name = row['name']
        if not name.startswith(neuron_name):
            continue
        nid = row['neuron_id']
        if version >= 93:
            cursor.execute("select * from front_data where neuron_id=?",(nid,) )
        else:
            cursor.execute("select * from neuron_data where neuron_id=?",(nid,) )
        frets = cursor.fetchall()
        print (name,": neuron_id",nid,",",len(frets),"fronts")
        if by_swc_type:
            swc_total = {} # dictionary by swc_type of total # fronts
            for row in frets:
                swc = str(row['swc_type'])
                if swc in swc_total:
                    swc_total[swc] += 1
                else:
                    swc_total[swc] = 1
            for key in swc_total.keys():
                print ("  swc_type",key,":",swc_total[key],"fronts")
        if front_ids:
            for row in frets:
                out = "  front_id " + str(row['front_id']).ljust(6)
                out += ": swc " + str(row['swc_type']).ljust(4)
                if sizes:
                    orig = Point(row['orig_x'],row['orig_y'],row['orig_z'])
                    end = Point(row['end_x'],row['end_y'],row['end_z'])
                    rad = row['radius']
                    if row['shape'] == 1: # sphere
                        length = 2. * rad
                    else:
                        length = (end - orig).length()
                    out += "radius %5.2f length %6.2f" % (rad,length)
                out += " made in cycle " + str(row['birth']).ljust(4)
                death = row['death']
                if death > 0:
                    out += " retracted in cycle " + str(death).ljust(4)
                print (out)

def nds_stats(data,max_range,leader="      ",mean=True,histo=True,ranges=True,bins=20,\
              min_range=0,plot=False,title=""):
    """ Prints or plots statistics for data list.
    
    By default prints mean +/- std and a histogram with 10 *bins* for given range.
    
    Parameters
    ----------
    data : list : list of data (integer or float).
    max_range : integer > 0 : maximum of range to be used for histogram.
    Optional:
    leader : string : printed first for every line, default: "      ".
    mean : boolean : print mean and standard deviation, default True.
    histo : boolean : print histogram with step_size, default True.
    ranges : boolean : print above histogram, default True.
    bins : integer > 0 : number of bins to be used for histogram, default 20.
    min_range : integer > 0 : minimum of range to be used for histogram, default 0.
    plot : boolean : plot instead of print, default False.
    title : string : title of plot (if plot==True), default "".
    """
    if plot:
        import matplotlib
        import matplotlib.pyplot as plt
    if not isinstance(data,list):
        print ("Error in data: data should be a list")
        return    
    if max_range <= 0:
        print ("Error in nds_stats: max_range should be larger than 0")
        return
    if min_range >= max_range:
        print ("Error in nds_stats: min_range should be smaller than max_range")
        return
    if bins <= 0:
        print ("Error in nds_stats: bins should be larger than 0")
        return
    if mean:
        print (leader,"Mean +/- std: {:4.2f}".format(np.mean(data)),"+/-","{:4.2f}".format(np.std(data)))
    if histo:
        if plot:
            fig=plt.figure(figsize=(4,3))
            plt.hist(data,bins=bins,range=(min_range,max_range))
            #plt.legend(loc=0)
            plt.title(title)
            plt.show()
        else:
            h,r = np.histogram(data,bins=bins,range=(min_range,max_range))
            if ranges:
                print (leader,"Histogram:",r)
                print (leader,"          ",h)
            else:
                print (leader,"Histogram:",h)

def nds_children(db_name,max_children=2,verbose=1):
    """ Checks whether any front except a soma has too many children in NeuroDevSim simulation database. Default is maximum 2 children.
    
    Parameters
    ----------
    db_name: string : name of raw SQL NeuroDevSim output database file.
    Optional:
    max_children : integer > 0 : maximum children allowed, default 2.
    verbose : 0 or 1 : print summary (0) or all problem fronts (1), default 1.
    """
    if not os.path.isfile(db_name):
        print ("Error in nds_children:",db_name,"not found")
        return
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Get volume data
    try:
        cursor.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in nds_children: not a NeuroDevSim database")
        return
    nret = cursor.fetchone()
    version = nret['version']
    if (version < 88): 
        print ("Error in nds_children: wrong database version",version)
        return
    pversion = float(version)/100.0
    print ("NeuroDevSim %.3f" % (pversion),"simulation:")
    # get all fronts
    if version >= 93:
        cursor.execute("select * from front_data")
    else:
        cursor.execute("select * from neuron_data")
    rets = cursor.fetchall()
    parents = {} # dictionary by _key containing number of children, -1 for somata
    too_many = [] # list of _keys of parents with too many children
    # first populate parents dictionary
    for row in rets:
        key = str(row['neuron_id']) + "_" + str(row['front_id'])
        swc = row['swc_type']
        if swc == 1: # soma
            parents[key] = -1 # do not count
        else:
            parents[key] = 0 # will count
    # next count children
    for row in rets:
        swc = row['swc_type']
        if swc == 1: # soma: no children
            continue 
        # find parent
        par_key = str(row['neuron_id']) + "_" + str(row['parent_id'])
        if par_key not in parents:
            print ("ERROR: parent",par_key,"not found for",row['neuron_id'],row['front_id'],swc)
        else:
            if parents[par_key] >= 0:
                parents[par_key] += 1
                if parents[par_key] > max_children:
                    if par_key not in too_many:
                        too_many.append(par_key)
    if too_many:
        print (len(too_many),"out of",len(rets),"fronts have too many children")
        if verbose > 0:
            too_many.sort()
            for key in too_many:
                print ("Parent",key,"has",parents[key],"children")
    else:
        print ("no problem children found")
        

def nds_front_id(db_name,front_id,sizes=False):
    """ Prints information about a single front from a NeuroDevSim simulation database.
    
    By default prints database neuron_id and number of fronts. Can also print all front_ids.
    
    Parameters
    ----------
    db_name: string : name of raw SQL NeuroDevSim output database file.
    front_id: integer : front_id of a front in the database.
    Optional:
    sizes : boolean : print also front sizes, default False.
    """
    if not os.path.isfile(db_name):
        print ("Error in nds_front_id:",db_name,"not found")
        return
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Get volume data
    try:
        cursor.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in nds_front_id: not a NeuroDevSim database")
        return
    nret = cursor.fetchone()
    version = nret['version']
    if (version < 93): 
        print ("Error in nds_front_id: wrong database version",version)
        return
    # get all fronts with front_id
    cursor.execute("select * from front_data where front_id=?",(front_id,) )
    frets = cursor.fetchall()
    for row in frets:
        nid = row['neuron_id']
        cursor.execute("select * from neuron_data where neuron_id=?",(nid,) )
        nrow = cursor.fetchone()
        name = nrow['name']
        out = name + " neuron_id " + str(nid).ljust(5)
        out += " front_id " + str(row['front_id']).ljust(6)
        out += ": swc " + str(row['swc_type']).ljust(4)
        if sizes:
            orig = Point(row['orig_x'],row['orig_y'],row['orig_z'])
            end = Point(row['end_x'],row['end_y'],row['end_z'])
            rad = row['radius']
            if row['shape'] == 1: # sphere
                length = 2. * rad
            else:
                length = (end - orig).length()
            out += "radius %5.2f length %6.2f" % (rad,length)
        out += " made in cycle " + str(row['birth']).ljust(4)
        death = row['death']
        if death > 0:
            out += " retracted in cycle " + str(death).ljust(4)
        print (out)

def nds_list_data(db_name,attribute,num_data=-1):
    """ Prints summary of data available for attribute in a NeuroDevSim database.
    
    Parameters
    ----------
    db_name : string : name of NeuroDevSim database file.
    attribute : string : name of attribute to report on. 
    Optional:
    num_data : integer -1 or >0 or [integer,integer]: restrict number of fronts listed to value indicated. When a list is used it is interpreted as a range. Default: -1, output all fronts.
    """
    
    if not os.path.isfile(db_name):
        print ("Error in nds_list_data:",db_name,"not found")
        return
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Get general data
    try:
        cursor.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in nds_list_data: not a NeuroDevSim database")
        return
    nret = cursor.fetchone()
    version = nret['version']
    if (version < 93): 
        print ("Error in nds_list_data: wrong database version",version)
        return
    if isinstance(num_data,list):
        if num_data[0] < 0:
            print ("Error in nds_list_data: invalid num_plots: start of range should be >= 0",num_data[0])
            return
        if num_data[1] <= num_data[0]:
            print ("Error in nds_list_data: invalid num_plots: end of range should be larger than start",num_data[1])
            return
        start_data = num_data[0]
        end_data = num_data[1]
    else:
        if (num_data < -1) or (num_data == 0):
            print ("Error in nds_list_data: invalid num_plots",num_data)
            return
        start_data = 0
        if num_data == -1:
            end_data = 1000000 # very large number
        else:
            end_data = num_data
    # load table
    table = attribute + "_data"
    try:
        cursor.execute("select * from " + table)
    except Exception:
        print ("Error in nds_list_data: no data table for ",attribute)
        return
    drets = cursor.fetchall()
    # load table info
    cursor.execute("select * from attributes where name=?",(table,))
    tret = cursor.fetchone()
    # organize data per neuron or front, dictionary contains entries as list:
    #   [first_cycle,last_cycle,num_data_points,min_value,max_value]
    #    for each cycle only one num_data_point is counted.
    front_data = {}
    num_f = 0
    for dret in drets:
        front = DataID(dret['neuron_id'],dret['front_id']) # can also be a neuron
        key = front._key()
        cycle = dret['cycle']
        value = dret[attribute]
        if key in front_data:
            data = front_data[key]
            if cycle > data[1]: # new cycle
                data[1] = cycle # last_cycle
                data[2] += 1    # num_data_points
            if value < data[3]:
                data[3] = value # min_value
            elif value > data[4]:
                data[4] = value # max_value
        else: # make new entry
            front_data[key] = [cycle,cycle,1,value,value]
            num_f += 1
    # count relevant data
    for key in front_data.keys():
        data = front_data[key]
    print (db_name,table,"contains data for",num_f,"fronts or neurons:")
    n_data = 0
    for key in front_data.keys():
        if n_data < start_data:
            n_data += 1
            continue # skip these ones
        data = front_data[key]
        if data[2] == 1:
            if tret['type'] == 'real': # format
                print (_key_to_DataID(key),": 1 data point for cycle",data[0],"value  {:4.2f}".format(data[3]))
            else:
                print (_key_to_DataID(key),": 1 data point for cycle",data[0],"value"  ,data[3])
        else:
            if tret['type'] == 'real': # format
                print (_key_to_DataID(key),":",data[2], "data points for cycles",data[0],"-",data[1],", range {:4.2f}".format(data[3]),"- {:4.2f}".format(data[4]))
            else:
                print (_key_to_DataID(key),":",data[2], "data points for cycles",data[0],"-",data[1],", range",data[3],"-",data[4])
        n_data += 1
        if n_data >= end_data:
            break # done listing

### DATABASE OUTPUT FUNCTIONS

def nds_swc_files(db_name,postfix="",max_cycle=-1,verbose=1):
    """ Generate swc files from a NeuroDevSim database.
    
    The swc file format is defined in Cannon et al. J. Neurosci. Methods 84, 49–54 (1998).
    
    Parameters
    ----------
    db_name : string : name of NeuroDevSim database file.
    Optional :
    postfix : string : attach string after database name in name of pdf file.
    max_cycle : integer > 0 : stop output at this cycle, default: all cycles (-1).
    verbose : integer 0-2 : control output, default 1: print output filename.
    """
    if not os.path.isfile(db_name):
        print ("Error in nm_swc_files:",db_name,"not found")
        return
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Check version
    try:
        cursor.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in nds_to_swc: not a NeuroDevSim database")
        return
    nret = cursor.fetchone()
    version = nret['version']
    if (version < 93):
        print ("Error in nds_to_swc: wrong database version",version)
        return
    # Get relevant parameters
    if max_cycle < 0: # data will be output till max_cycle (included)
        max_cycle = nret['num_cycles']
    if nret['migration'] > 0: # load migration_data
        mig_table = nret['migration']
        mrets = []
        mig_max_row = []
        mig_first_cycle = []
        for i in range(mig_table):
            try:
                cursor.execute("select * from migration_data" + str(i + 1))
            except Exception:
                print ("Error in nds_to_swc: expected migration_data" + str(i + 1) + " table")
                return
            mr = cursor.fetchall()
            mrets.append(mr)
            mig_first_cycle.append(mr[0]['cycle']) # python starts at row 0
            mig_max_row.append(len(mr) - 1)
    else:
        mrets = None
    # Get all neuron names
    cursor.execute("select * from neuron_data")
    rets = cursor.fetchall()
    names = {} # dict by name of all neuron_ids
    c=0
    for row in rets :
        names[row['name']] = row['neuron_id']
    # Generate file name prefix
    prefix=""
    if db_name.startswith(".."):    #relative path
        prefix = db_name.split("/")
        if verbose > 1:
            print (".. prefix: ", prefix)
    elif db_name.startswith("/"): #complete path -> copy
        prefix = "/".join(db_name.split("/")[:-1])
        if verbose > 1:
            print (".. prefix: ", prefix)
    else:
        prefix = "/".join(db_name.split(".")[0].split("/")[:-1])
        if verbose > 1:
            print ("prefix: ", prefix)
    # Make swc files for each neuron in the data
    for name in names.keys():
        cursor.execute("select * from front_data where neuron_id=? order by birth",(names[name],) )
        rets = cursor.fetchall()
        if len(prefix) > 0:
            if verbose > 0:
                print ("writing to file: ", prefix+"/"+name+".swc")
            writer = open(prefix+"/"+name+".swc","w")
        else:
            if verbose > 0:
                print ("writing to file: ", name+".swc")
            writer = open(name+".swc","w")
        to_write = "# swc file generated by NeuroDevSim version " + str(version/100) + "\n"
        writer.write(to_write)
        index_mapping = {} # maps fid into swc file index
        if rets[0]['parent_id'] != -1:
            print ("Error in nds_to_swc: no root segment found")
            return
        row = rets[0]
        soma_id = row['front_id']
        # insert the soma + first segments
        radius = row['radius']
        if row['branch'] == "soma": # real soma
            mcol = row['migration']
            if mcol > 0: # soma migrated
                if not mrets:
                    print ("Error in nds_to_swc: expected migration_data1 table")
                    return
                mr_ind = mcol // 1800
                mrow = birth - mig_first_cycle[mr_ind]
                if mrow > mig_max_row:
                    mrow = mig_max_row
                xyz = _get_mig_xyz(mrets,mrow,mcol)
                if not xyz: # no migration data for this neuron
                    xyz = (row['orig_x'],row['orig_y'],row['orig_z'])
            else:
                xyz = (row['orig_x'],row['orig_y'],row['orig_z'])
            if verbose > 1:
                print ("constructing soma with radius:", radius)
            # approximate sphere
            soma_str = "1 1 %.2f %.2f %.2f %.2f -1\n" % (xyz[0],xyz[1],xyz[2],radius)
            soma_str += "2 1 %.2f %.2f %.2f %.2f 1\n" % (xyz[0],xyz[1]-radius,xyz[2],radius)
            soma_str += "3 1 %.2f %.2f %.2f %.2f 1\n" % (xyz[0],xyz[1]+radius,xyz[2],radius)
            writer.write(soma_str)
            index_mapping[soma_id] = 1 # connect children to spherical center
            new_index = 4
        else: # axon
            if row['branch'] != "axon":
                print ("Error in nds_to_swc: unknown root segment: not soma or axon")
                return
            xyz0 = (row['orig_x'],row['orig_y'],row['orig_z'])
            xyz = (row['end_x'],row['end_y'],row['end_z'])
            # output start and end points
            to_write = "1 2 %.2f %.2f %.2f %.2f -1\n" % (xyz0[0],xyz0[1],xyz0[2],radius)
            to_write += "2 2 %.2f %.2f %.2f %.2f 1\n" % (xyz[0],xyz[1],xyz[2],radius)
            index_mapping[soma_id] = 2 # connect children to cylindrical end point
            new_index = 3
        # now write rest of neuron
        for row in rets[1:]:
            if row['birth'] > max_cycle:
                continue # ignore data beyond max_cycle
            fid = row['front_id']
            xyz = (row['end_x'],row['end_y'],row['end_z'])
            swc_type = row['swc_type']
            radius = row['radius']
            pid = row['parent_id']
            if pid in index_mapping:
                if verbose > 1:
                    print ("processing front",fid)
                to_write = "%i %i %.2f %.2f %.2f %.2f %i\n" % \
                  (new_index,swc_type,xyz[0],xyz[1],xyz[2],radius,index_mapping[pid])
                index_mapping[fid] = new_index
                writer.write(to_write)
                new_index += 1
            else:
                print ("Error in nds_to_swc: could not find parent",pid,"for",fid,"index",new_index)
                return
    writer.flush()

def nds_output_sizes(db_name,neuron_name="",number_fronts=False,\
                number_swc_type=False,length=True):
    """ Outputs a list containing specific sizes from a NeuroDevSim database.
    
    Four output options are available:
    - default : length of all fronts is returned as a list.
    - neuron_name specified : data is printed and returned for all fronts of a single neuron.
    - number_fronts : total number of fronts is printed for each neuron.
    - number_swc_type : total number of fronts of each swc_type is printed for each neuron.
    
    Parameters
    ----------
    db_name: string : name of raw SQL NeuroDevSim output database file.
    Optional:
    neuron_name: string : name of a neuron in the database, can be a wildcard, print and return only data for this neuron, default not specified (all neurons).
    number_fronts : boolean : print only total number of fronts for each neuron, default False.
    number_swc_type : boolean : print only total number of fronts of each swc_type for each neuron, default False.
    length : boolean : return front lengths (True) or radii (False), default True.
    
    Returns
    -------
    list of lengths or radii : [float, ]
    """
    if not os.path.isfile(db_name):
        print ("Error in nds_output_sizes:",db_name,"not found")
        return
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Get volume data
    try:
        cursor.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in nds_output_sizes: not a NeuroDevSim database")
        return
    nret = cursor.fetchone()
    version = nret['version']
    if (version < 90): 
        print ("Error in nds_output_sizes: wrong database version",version)
        return
    # get all neuron names
    if version >= 93:
        cursor.execute("select * from neuron_data")
    else:
        cursor.execute("select * from names_data")
    rets = cursor.fetchall()
    names = {} # dict by neuron_id of all names
    use_name = len(neuron_name) > 0
    result = []
    for row in rets:
        name = row['name']
        if use_name and (not name.startswith(neuron_name)):
            continue
        nid = row['neuron_id']
        if version >= 93:
            cursor.execute("select * from front_data where neuron_id=?",(nid,))
        else:
            cursor.execute("select * from neuron_data where neuron_id=?",(nid,))
        frets = cursor.fetchall()
        if number_swc_type:
            swc_total = {} # dictionary by swc_type of total # fronts
        for row in frets:
            if number_swc_type:
                swc = str(row['swc_type'])
                if swc in swc_total:
                    swc_total[swc] += 1
                else:
                    swc_total[swc] = 1
            rad = row['radius']
            if length:
                orig = Point(row['orig_x'],row['orig_y'],row['orig_z'])
                end = Point(row['end_x'],row['end_y'],row['end_z'])
                if row['shape'] == 1: # sphere
                    flength = 2. * rad
                else:
                    flength = (end - orig).length()
                result.append(flength)
            else:
                result.append(rad)
        if number_fronts or number_swc_type:
            print (name,": neuron_id",nid,",",len(frets),"fronts")
        if number_swc_type:
            for key in swc_total.keys():
                print ("  swc_type",key,":",swc_total[key],"fronts")
    return result
    
def nds_output_data(db_name,front_id,attribute,min_cycle=0,max_cycle=-1):
    """ Output values of an attribute versus time for a front or neuron.
    
    At present only attributes stored with ``admin.attrib_to_db`` can be selected. 
    To restrict the range of cycles plotted use the *min_cycle* and *max_cycle* optional parameters.
    
    Parameters
    ----------
    db_name : string : name of NeuroDevSim database file.
    front_id : dataID : selects which front to plot, can also be neuron_id
    attribute : string : name of attribute to output. 
    Optional :
    max_cycle : integer -1 or >0 : stop plotting at this cycle, default -1: plot till end of simulation.
    min_cycle : integer >0 : start plotting at this cycle, default 0: plot from beginning.
    
    Returns:
    list of (cycle,attribute_value) : list
    """

    if not os.path.isfile(db_name):
        print ("Error in nds_output_data:",db_name,"not found")
        return
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Get general data
    try:
        cursor.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in nds_output_data: not a NeuroDevSim database")
        return
    nret = cursor.fetchone()
    version = nret['version']
    if (version < 93): 
        print ("Error in nds_output_data: wrong database version",version)
        return
    if max_cycle < 0:
        max_cycle = nret['num_cycles']
    if min_cycle < 0:
        print ("Error in nds_output_data: negative min_cycle",min_cycle)
        return
    if min_cycle >= max_cycle:
        print ("Error in nds_output_data: min_cycle >= max_cycle",min_cycle, max_cycle)
        return
    table = attribute + "_data"
    try:
        cursor.execute("select * from " + table)
    except Exception:
        print ("Error in nds_output_data: no data table for ",attribute)
        return
    drets = cursor.fetchall()
        
    # find data for neuron or front
    front_data = []
    for dret in drets:
        fid = DataID(dret['neuron_id'],dret['front_id']) # can also be a neuron
        #print (fid,front_id)
        if fid != front_id:
            continue
        cycle = dret['cycle']
        if (cycle < min_cycle) or (cycle > max_cycle):
            continue    # outside of range
        value = dret[attribute]
        front_data.append((cycle,value))
    return front_data
    
def nds_output_neurons(db_name,neuron_type):
    """ Output information about all neurons of a specific type.
    
    Parameters
    ----------
    db_name : string : name of NeuroDevSim database file.
    neuron_type : integer : index into neuron_types list provided to ``Admin_agent``.
    
    Returns:
    dictionary by neuron_id : list [[neuron_name,num_fronts,num_retracted,num_synapses],]
    """
    if not os.path.isfile(db_name):
        print ("Error in nds_output_neurons:",db_name,"not found")
        return
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Get version data
    try:
        cursor.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in nds_output_neurons: not a NeuroDevSim database")
        return
    nret = cursor.fetchone()
    version = nret['version']
    if (version < 92): 
        print ("Error in nds_output_neurons: wrong database version",version)
        return
    # check neuron_type
    cursor.execute("select * from neuron_types")
    trets = cursor.fetchall()
    num_types = len(trets)
    if (neuron_type < 0) or (neuron_type >= num_types):
        print ("Error in nds_output_neurons: neuron_type should be in range 0 -" ,num_types)
        return
    neuron_type += 1 # as used in database
    # get all neuron data
    if version >= 93.2:
        cursor.execute("select * from neuron_data where type_id=?",(neuron_type,))
    elif version >= 93.0:
        cursor.execute("select * from neuron_data where array_id=?",(neuron_type,))
    else:
        cursor.execute("select * from names_data where array_id=?",(neuron_type,))
    nrets = cursor.fetchall()
    neurons = {}
    for row in nrets:
        neurons[row['neuron_id']] = [row['name'],row['num_fronts'],row['num_retracted'],row['num_synapses']]
    return neurons
    
def nds_output_migration(db_name,last_pos=False):
    """ Output migration history for each migrating somata in database.
    
    Parameters
    ----------
    db_name : string : name of NeuroDevSim database file.
    Optional :
    last_pos : boolean : output only final position, default: False.
    
    Returns:
    dictionary by neuron_id : dictionary of [[cycle,Point],] (last_pos=False) or [cycle,Point] (last_pos=True)
    """
    if not os.path.isfile(db_name):
        print ("Error in nds_output_migration:",db_name,"not found")
        return
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # check version
    try:
        cursor.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in nds_output_migration: not a NeuroDevSim database")
        return
    nret = cursor.fetchone()
    version = nret['version']
    if (version < 92): 
        print ("Error in nds_output_migration: wrong database version",version)
        return
    # get info about all migrating somata    
    if version >= 93:
        cursor.execute("select * from front_data")
    else:
        cursor.execute("select * from neuron_data")
    frets = cursor.fetchall()
    mig_somata = {} # dict by migration column of neuron_id
    for row in frets:
        mig_col = row['migration']
        if mig_col > 0:
            mig_somata[mig_col] = row['neuron_id']
    # load migration tables
    n_mig_tables = nret['migration']
    mig_history = {} # dict by col_key containing [[coordinate,][cycle,]]
    first = True
    for i in range(n_mig_tables):
        try:
            cursor.execute("select * from migration_data" + str(i + 1))
        except Exception:
            raise TypeError(str(db_name) + ": expected migration_data" + str(i + 1) + " table")
        mrets = cursor.fetchall()
        num_cols = len(mrets[0])
        # turn mrets data into a dictionary by neuron_id of list of (cycle,xyz)
        for row in mrets:
            cycle = row['cycle']
            for ncol in range(2,num_cols,3):
                if row[ncol]:
                    mig_col = i * 1800 + ncol
                    if mig_col in mig_somata:
                        neuron_id = mig_somata[mig_col]
                    else:
                        raise BugError("nds_output_migration","missing migration column")
                    data = [cycle,Point(row[ncol],row[ncol+1],row[ncol+2])]
                    if neuron_id in mig_history:
                        mig_history[neuron_id].append(data)
                    else:
                        mig_history[neuron_id] = [data]
    if last_pos: # turn mig_history into dictionary of last positions
        mig_last = {}
        for key in mig_history.keys():
            mig_last[key] = mig_history[key][-1]
        return mig_last
    else:
        return mig_history

def nds_dict_to_list(d,index):
    """ Extracts data from a dictionary.
    
    Parameters
    ----------
    dict : string : name of NeuroDevSim dictionary produced by nds_output_neurons or nds_output_migration.
    index : integer >= 0 : index in list for which data should be returned.
    
    Returns : list
    """
    if not isinstance(d,dict):
        print ("Error in nds_dict_to_list: dict should be dictionary")
        return    
    if index < 0:
        print ("Error in nds_dict_to_list: index should be >= 0")
        return
    data = []
    for key in d.keys():
        data.append(d[key][index])
    return data


### DATABASE CHECK FUNCTIONS

# check whether any front collisions occurred during the simulation
def check_collision(db_name,verbose=1):
    """ Checks whether any undetected collisions are present in a NeuroDevSim database.
    
    Parameters
    ----------
    db_name : string : name of NeuroDevSim database file.
    Optional :
    verbose : integer 0-3 : control output, default 1.
    """
    np.set_printoptions(precision=2)
    if not os.path.isfile(db_name):
        print ("Error in database_collision:",db_name,"not found")
        exit()
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    # Check version
    try:
        cursor.execute("select * from neurodevsim") # Should be only one row
    except Exception:
        print ("Error in database_collision: not a NeuroDevSim database")
        exit()
    nret = cursor.fetchone()
    version = nret['version']
    if (version < 93):
        print ("Error in database_collision: wrong database version",version)
        exit()
    # Get relevant parameters
    max_cycle = nret['num_cycles']
    if nret['migration'] > 0: # load migration_data
        mig_table = nret['migration']
        mig_xyz = {} # dict by key cycle containing lists of xyz
        mig_orig_xyz = [] # original or final xyz of migrating soma (used if mig_xyz[][] == None)
        db_to_dict = {} # dict by key database migration column to mig_xyz column
        first = True
        mcol = 0 # counts columns in mig_xyz
        #mig_first_cycle = []
        for i in range(mig_table):
            try:
                cursor.execute("select * from migration_data" + str(i + 1))
            except Exception:
                print ("Error in nds_to_swc: expected migration_data" + str(i + 1) + " table")
                return
            mrets = cursor.fetchall()
            #mig_first_cycle.append(mr[0]['cycle']) # python starts at row 0
            # turn mrets data into a list of xyz
            num_cols = len(mrets[0])
            for row in mrets:
                mrow = [] # new entry for mig_xyz
                for ncol in range(2,num_cols,3):
                    key = i * 1800 + ncol
                    cycle = row['cycle']
                    if first:
                        db_to_dict[key] = mcol
                    if row[ncol]:
                        xyz = Point(row[ncol],row[ncol+1],row[ncol+2])
                        mrow.append(xyz)
                    else:
                        mrow.append(None)
                    mcol += 1
                mig_xyz[cycle] = mrow
                first = False
    else:
        mrets = None
    # Get all neuron names
    cursor.execute("select * from neuron_data")
    rets = cursor.fetchall()
    id_names = {} # dict by neuron_ids of all names
    c=0
    for row in rets :
        id_names[row['neuron_id']] = row['name']
    # get one big table from front_data
    cursor.execute("select * from front_data order by birth")
    rets = cursor.fetchall()
    # convert this to a list of data with np_arrays:
    data = [] # all non migrating fronts
    msoma = [] # all migrating somata
    index = 0
    mindex = 0
    cycle_index = {} # cycle based dictionary into data
    mcycle_index = {} # cycle based dictionary into msoma
    prev_cycle = -1
    for row in rets:
        cycle = row['birth']
        # fill cycle_index arrays
        if cycle > prev_cycle:
            for n in range(prev_cycle + 1, cycle + 1):
                cycle_index[n] = index
                mcycle_index[n] = mindex
            prev_cycle = cycle
        elif cycle < prev_cycle:
            print ("Error in database_collision: front_data not ordered by birth")
            exit()
        nid = row['neuron_id']
        name = id_names[nid]
        fid = row['front_id']
        shape = row['shape']
        parent = row['parent_id']
        # order of data and msoma lists:
        # 0: cycle
        # 1: neuron id
        # 2: front id
        # 3: parent id
        # 4: shape
        # 5: radius
        # 6: death
        # 7: swc_type (negative for migrating soma)
        # 8: xyz0 (for data) or migration column (for msoma)
        # 9: xyz (only data)
        # 10-11 are filled later in data : #10 grandparent id, #11 parent length
        #xyz0 = Point(row['from_x'],row['from_y'],row['from_z'])
        #xyz = Point(row['to_x'],row['to_y'],row['to_z'])
        xyz0 = Point(row['orig_x'],row['orig_y'],row['orig_z'])
        xyz = Point(row['end_x'],row['end_y'],row['end_z'])
        if row['migration'] == 0:
            if (verbose > 2) and (parent == -1):
                print ("adding soma",fid,"to data")
            data.append([cycle,nid,fid,parent,shape,\
                    row['radius'],row['death'],row['swc_type'],xyz0,xyz])
            index += 1
        else:
            if shape != 1:
                print ("Error in database_collision: migrating cylinder")
                exit()
            msoma.append([cycle,nid,fid,parent,shape,\
                    row['radius'],row['death'],-1,db_to_dict[row['migration']]])
            mig_orig_xyz.append(xyz0)
            mindex += 1
    maxi = len(data)
    if mrets:
        mig_orig0_xyz = copy.copy(mig_orig_xyz) # mig_orig_xyz will be changed
    # extend data with grandparent etc
    for i1 in range(maxi):
        fid = data[i1][2]
        gpid = data[i1][3]
        if data[i1][4] == 1: # sphere
            length = 2 * data[i1][5]
        else:
            length = (data[i1][8] - data[i1][9]).length()
        for i2 in range(i1 + 1, maxi):
            if data[i2][3] == fid: # is child of i1
                data[i2].append(gpid) # grandparent id
                data[i2].append(length) # parent length
    # fill end of cycle_index arrays
    for c in range(cycle + 1, max_cycle + 2):
        cycle_index[c] = index
        mcycle_index[c] = mindex
    # iterate through all non-migrating fronts and do all possible pairwise comparisons
    num_coll = 0 # number of collisions
    if verbose > 2:
        print ("Checking fixed fronts")
    for i1 in range(maxi):
        # prepare for entering data[i2][10] and data[i2][11]
        fid = data[i1][2]
        gpid = data[i1][3]
        if data[i1][4] == 1: # sphere
            length = 2 * data[i1][5]
        else:
            length = (data[i1][8] - data[i1][9]).length()
        for i2 in range(i1 + 1, maxi):
            if data[i1][6] >= 0: # was retracted -> only compare with structures that existed before retraction
                if data[i2][0] >= data[i1][6]: # 2 was born after 1 retraction
                    continue
            if (len(data[i2]) == 10) and (data[i2][3] == fid): # is child of i1
                print ("late update")
                data[i2].append(gpid) # grandparent id
                data[i2].append(length) # parent length
            # retraction of 2 doesn't matter because comparison is in order of birth:
            #   1 is always born before or together with 1
            #print ("comparing",i1,data[i1][1],data[i1][2],i2,data[i2][1],data[i2][2])
            num_coll += _compare_2fronts(data[i1],data[i2],verbose)
    # check all migrating soma: among themselves and to concurrent front data
    if mrets:
        maxm = len(msoma)
        my_coll = {} # dictionary by key i1 with list of colliding i2
        old_stat_i1 = [False] * maxm # track migration status: goes False -> True (-> False)
        old_stat_i2 = [False] * maxm # track migration status
        for cycle in range(max_cycle + 1):
            if verbose > 2:
                print ("Checking migrating fronts, cycle:",cycle)
            # compare to each other
            for i1 in range(mcycle_index[cycle + 1]):
                new_stat, xyz0_1 = _mig_soma_xyz(msoma[i1][8],cycle,mig_xyz,\
                                                 mig_orig_xyz,True)
                if (old_stat_i1[i1] == True) and (new_stat == False): # stopped migrating
                    stopped_i1 = True
                else:
                    stopped_i1 = False
                    old_stat_i1[i1] = new_stat # update for next comparison
                for i2 in range(i1 + 1, mcycle_index[cycle + 1]):
                    #if (msoma[i1][2] == 7) and (msoma[i2][2] == 15024):
                        #print (cycle,"comparing",msoma[i1][2],msoma[i1][6],msoma[i2][2],msoma[i2][6])
                    new_stat, xyz0_2 = _mig_soma_xyz(msoma[i2][8],cycle,mig_xyz,\
                                                     mig_orig_xyz,False)
                    if (old_stat_i2[i2] == True) and (new_stat == False): # stopped migrating
                        stopped_i2 = True
                    else:
                        stopped_i2 = False
                        old_stat_i2[i2] = new_stat # update for next comparison
                    if stopped_i1 or stopped_i2: # one stopped migrating
                        continue
                    num_coll += _compare_2fronts(data[i1],data[i2],verbose)
            # compare to other fronts present at this cycle
            mig_orig_xyz = mig_orig0_xyz # reset to original
            if maxi > 0: # only if regular fronts present
                for i1 in range(mcycle_index[cycle + 1]):
                    new_stat, xyz0_1 = _mig_soma_xyz(msoma[i1][8],cycle,mig_xyz,\
                                                     mig_orig_xyz,True)
                    for i2 in range(cycle_index[cycle + 1]):
                        if (data[i2][6] >= 0) and (cycle >= data[i2][6]):
                            # i2 retracted before this cycle
                            continue
                        if data[i2][7] == 12: # filipod
                            # do not compare migrating soma with its filipod
                            continue
                        if data[i2][7] == 2: # axon
                            if (data[i2][0] == cycle):
                                # do not compare migrating soma child trailing axon
                                continue
                            stat, xyz = _mig_soma_xyz(msoma[i1][8],data[i2][0],\
                                                      mig_xyz,mig_orig_xyz,False)
                            if np.array_equal(xyz,xyz0_1): # soma stopped moving
                                continue
                        if (i1 in my_coll) and (i2 in my_coll[i1]):
                            # already found a collision between these two
                            continue
                        if _compare_2fronts(data[i1],data[i2],verbose):
                            if i1 not in my_coll:
                                my_coll[i1] = [i2]
                                num_coll += 1
                            elif i2 not in my_coll[i1]:
                                my_coll[i1].append(i2)
                                num_coll += 1

    if num_coll > 0:
        print (Fore.RED + str(num_coll) + " collisions detected for",len(rets),"points",Fore.RESET)
    else:
        print (Fore.BLUE + "No collisions detected for",len(rets),"points",Fore.RESET)
                    
### GENERAL UTILITY FUNCTIONS

# extract relevant data from the migration_data table: mrow may contain None
# returns [x,y,z] or None
def _get_mig_xyz(mrets,mrow,mcol):
    mr_ind = mcol // 1800
    mr_col = mcol % 1800
    x = mrets[mr_ind][mrow][mr_col]
    # search for last row with data in case it stopped migratig before max_cycle
    while not x:
        mrow -= 1
        if mrow < 1:
            break
        x = mrets[mr_ind][mrow][mr_col]
    if mrow > 0:
        xyz = (mrets[mr_ind][mrow][mr_col],mrets[mr_ind][mrow][mr_col+1],mrets[mr_ind][mrow][mr_col+2])
        return xyz
    else:
        return None

# strip directory and '.db' from database name
def strip_db_name(db_name):
    """ removes directories and '.db' from NeuroDevSim database name.
    
    Parameters
    ----------
    db_name : string : name of NeuroDevSim database file.

    Returns
    -------
    name : string
    """
    if '/' in db_name:
        fname = db_name.rsplit(sep='/',maxsplit=1)[1]
    else:
        fname = db_name
    fname = fname.split(".db")[0]
    return fname

# is point in a volume
def point_in_volume(point,volume):
    """ Returns whether *point*` is inside the given *volume* (including its borders).
        
    Parameters
    ----------
    point : ``Point`` object.
    volume: list of 2 lists: coordinates specifying volume in µm as: [[left, front, bottom], [right, back, top]]

    Returns
    -------
    is inside volume : boolean.
    """
    if point.x < volume[0][0]:
        return False
    if point.y < volume[0][1]:
        return False
    if point.z < volume[0][2]:
        return False
    if point.x > volume[1][0]:
        return False
    if point.y > volume[1][1]:
        return False
    if point.z > volume[1][2]:
        return False
    return True

class _FakeFront(object):
    def __init__(self,p0,p1):
        self.orig = p0
        self.end = p1

# do actual front comparison and return OK (0) or collision (1)
def _compare_2fronts(f1,f2,verbose):
    min_dist = f1[5] + f2[5] # rad1 + rad2
    # make same exceptions as main code
    if (f1[1] == f2[1]): # same neuron only
        #if parent1 == front2 or parent2 == front1 or parent1 == parent2:
        if (not ((f1[3] == -1) and (f2[3] == -1))) and (f1[1] == f2[1]) and\
           ((f1[3] == f2[2]) or (f2[3] == f1[2]) or (f1[3] == f2[3])):
            return 0 # do not measure parent to child or between siblings
        # do not measure to grandparent if parent has shorter length than min_dist
        if len(f2) == 12: # be safe
            if (f2[10] == f1[2]) and (f2[11] <= min_dist):
                return 0
    #if shape1 > 1 and shape2 > 1:
    if (f1[4] > 1) and (f2[4] > 1):
        D,p1,p2 = dist3D_cyl_to_cyl(f1[9],f1[8],_FakeFront(f2[9],f2[8]),points=True)
        if np.array_equal(f1[8], p1) or np.array_equal(f1[9], p1) or \
            np.array_equal(f2[8], p2) or np.array_equal(f2[9], p2):
            return 0
    elif (f1[4] == 1) and (f2[4] > 1):
        D,p1 = dist3D_point_to_cyl(f1[8],f2[9],f2[8],points=True)
        if np.array_equal(f1[8], p1) or  \
            np.array_equal(f2[8], p1) or np.array_equal(f2[9], p1):
             return 0
    elif (f1[4] > 1) and (f2[4] == 1):
        D,p1 = dist3D_point_to_cyl(f2[6],f1[9],f1[8],points=True)
        if np.array_equal(f2[8], p1) or  \
            np.array_equal(f1[8], p1) or np.array_equal(f1[9], p1):
             return 0
    else:
        D = (f1[8] - f2[8]).length()
    if D < min_dist:
        if verbose:
            if (abs(f1[7]) != 1) or (abs(f2[7]) != 2):
                print ("Collision between",f1[1],":",f1[2],"(p",f1[3],", swc",abs(f1[7]),") and ",f2[1],":",f2[2],"(p",f2[3],", swc",abs(f2[7]),")",": {:4.2f}".format(D)," {:4.2f}".format(min_dist))
            if verbose > 1:
                if (f1[4] > 1) and (f2[4] > 1):
                    print (f1[0],f1[8],f1[9]," {:4.2f}".format(f1[5]),"+",f2[0],f2[8],f2[9]," {:4.2f}".format(f2[5]),"coll at:",p1,p2)
                elif (f1[4] == 1) and (f2[4] > 1):
                    print (f1[0],f1[8]," {:4.2f}".format(f1[5]),"+",f2[0],f2[8],f2[9]," {:4.2f}".format(f2[5]),"coll at:",p1)
                elif (f1[4] > 1) and (f2[4] == 1):
                    print (f1[0],f1[8],f1[9]," {:4.2f}".format(f1[5]),"+",f2[0],f2[8]," {:4.2f}".format(f2[5]),"coll at:",p1)
                else:
                    print (f1[8],f1[0]," {:4.2f}".format(f1[5]),"+",f2[0],f2[8]," {:4.2f}".format(f2[5]))
        return 1
    else:
        return 0

# returns migrating soma status (migrating (True) or not (False)) and xyz0 for cycle
def _mig_soma_xyz(mcol,cycle,mig_xyz,mig_orig_xyz,first):
    if cycle in mig_xyz:
        xyz = mig_xyz[cycle][mcol]
        if xyz:
            if first: # update only in i1 loop
                mig_orig_xyz[mcol] = xyz # store for possible future use
            return True,xyz
        elif mcol < len(mig_orig_xyz):            
            return False,mig_orig_xyz[mcol]                
    # stopped migrating previously and this was not stored
    for c in range(cycle - 1,1,-1):
        if c in mig_xyz:
            xyz = mig_xyz[c][mcol]
            if xyz:
                return False,xyz

# prints possible differences between two 'identical' rows
# returns True if different, False otherwise
def _print_diffs(name,fid1,fid2,row1,row2,comp_end,print1):
    diff = False
    if comp_end:
        if (row1['end_x'] != row2['end_x']) or (row1['end_y'] != row2['end_y']) or \
            (row1['end_z'] != row2['end_z']):
            end1 = "[{:.2f}".format(row1['end_x']) + ", {:.2f}".format(row1['end_y']) + \
                   ", {:.2f}".format(row1['end_z']) + "]"
            end2 = "[{:.2f}".format(row2['end_x']) + ", {:.2f}".format(row2['end_y']) + \
                   ", {:.2f}".format(row2['end_z']) + "]"
            print (name,fid1,fid2,"different end:",end1,end2)
            if print1:
                return True
            else:
                diff = True
    if row1['radius'] != row2['radius']:
        print (name,fid1,fid2,"different radius:","{:.2f}".format(row1['radius']),\
                "{:.2f}".format(row2['radius']))
        if print1:
            return True
        else:
            diff = True
    if row1['death'] != row2['death']:
        print (name,fid1,fid2,"different death:",row1['death'],row2['death'])
        return True
    return diff
    
