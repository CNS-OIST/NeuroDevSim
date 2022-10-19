from neurodevsim.processing import nds_plot,nds_movie

# color_scheme=3 examples, these use databases where the extra attribute was stored
#   make a pdf file
nds_plot("output/store_signal.db",prefix="./plots and movies/",color_scheme=3,color_data=['signal',0.,100.],elev=90,azim=0,axis_ticks=False)
#   make a movie
nds_movie("output/cmigration.db",prefix="./plots and movies/",color_scheme=3,color_data=['local_attractor',0.,4000.])

# color_scheme=-1 example: specify neuron colors so that all blocking somata are grey, colors are stored in bmigration_colors.txt
#   make a pdf file
nds_plot("output/bmigration.db",prefix="./plots and movies/",soma_black=False,color_scheme=-1, neuron_colors="output/bmigration_colors.txt")
#   make a movie
nds_movie("output/bmigration.db",prefix="./plots and movies/",soma_black=False,color_scheme=-1, neuron_colors="output/bmigration_colors.txt")

