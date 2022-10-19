.. _plots-label:

Plotting the simulation
=======================
NeuroDevSim can plot during simulations when run in jupyter notebooks by setting `plot=True` during `Admin_agent` initialization. Results can also be plotted afterwards from the stored database using the `nds_plot` or `nds_movie` commands. All these methods have common settings that are explained here.

Color settings
--------------
There are several parameters that control how structures in the simulation are colored, selecting from 13 standard colors. By default all somata are black and each neuron has a different color, but because of the small number of colors available several neurons will have the same color in networks with many neurons. The following settings control how colors are used in `Admin_agent`, `nds_plot` and `nds_movie`:

**color_scheme :** integer : has 4 possible settings:

- 0 : default, all neurons have different colors, limited by number of colors available. Colors are asigned in order of soma creation.
- 1 : color determined by *neuron_types*. All neurons of the same type have the same color, different types have different colors, limited by number of colors available. Useful in network simulations with many different neuron types.
- 2 : color determined by *branch_name*. Fronts with different branch_names have different colors, limited by number of colors available. Useful when simulating a single neuron.
- 3 : a continuous color scale is used (matplotlib 'rainbow') to color a scalar attribute. Additional information must be provided in **color_data**. All fronts that do not have the selected attribute are colored black. An example can be found in the :ref:`synapsenote-label`.

**color_data :** list with 3 entries : used only for `color_scheme==3`:

- attribute : string : the front attribute to be used for color selection.
- min value : float : mimimum value used for color scale (purple color).
- max value : float : maximum value used for color scale (red color).

**soma_black :** boolean : every soma is always colored black (default), used for *color_scheme* 0-2.

In addition, `nds_plot` and `nds_movie` also have a `color_scheme==-1` setting. This allows the user to specify a specific color for each neuron in the simulation in a text file, which is defined in **neuron_colors**. The easiest way to do this is to first call the `nds_get_color_dict` method on the database, this will output a text file containing for each neuron a line with its name and the color selected for `color_scheme==0` (optionally `color_scheme==1`). The user can then edit this text file, being careful to specify colors correctly as shown in the table below, and then run `nds_plot` or `nds_movie` with `color_scheme==-1`.

**neuron_colors :** string : name of text file containing dictionary info by neuron name that specifies color to use for `color_scheme==-1` in `nds_plot` or `nds_movie`. Use `nds_get_color_dict` on the database to obtain a valid text file and then edit it.

Available colors and their names
++++++++++++++++++++++++++++++++

These color names should be used in the **neuron_color** text file.

=======  ===========
color    name used
=======  ===========
black    'k'
blue     'tab:blue'
brown    'tab:brown'
cyan     'tab:cyan'
gray     'tab:gray'
green    'tab:green'
magenta  'm'
olive    'tab:olive'
orange   'tab:orange'
pink     'tab:pink'
purple   'tab:purple'
red      'tab:red'
yellow   'y'
=======  ===========

View settings
-------------
Properly setting orientation of the camera with **azim** and **elev** can improve visibility of relevant phenomena and in complex simulations it can also be helpful to zoom in with **box**. Finally visibility of small structures can be enhanced with **radius_scale** or **sphere_scale**.

**azim :** float : azimuth in degrees of camera, default -60.

**box :** list [[left, front, bottom], [right, back, top]]: subvolume to plot, allows to zoom in, default full *sim_volume.*

**elev :** float : elevation in degrees of camera, default 30.

**radius_scale :** float : change thickness of cylindrical fronts, default 1. size equals 2 * radius.

**sphere_scale :** float : change size of spherical fronts, default 1. size equals radius.

Axes settings
-------------
One can suppress axes with **no_axis** or change relative scaling of axes with **scale_axis**. The latter can be quite important as Matplotlib plots by default a cubic volume, so if not all axes have identical length the default setting will create a distorted plot with some axes compressed. This may, however, squeeze the axis ticks to unlegible text and in that case it is better to turn them off with **axis_ticks**.

**axis_ticks :** boolean : show axis ticks, default True.

**no_axis :** boolean : suppress drawing of axes, default False.

**scale_axis :** boolean or list of 3 floats : list as [1.0,1.0,1.0] decrease one or more values to change relative scaling of axes, value for largest axis should be close to 1.0; default False. Examples in :ref:`realnote-label`.


