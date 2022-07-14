from neurodevsim.processing import nds_plot,nds_movie

# color_scheme=3 examples, these use databases where the extra attribute was stored
#   make a pdf file
nds_plot("output/store_signal.db",prefix="./plots and movies/",color_scheme=3,color_data=['signal',0.,100.],elev=90,azim=0,axis_ticks=False)
#   make a movie
nds_movie("output/cmigration.db",prefix="./plots and movies/",color_scheme=3,color_data=['local_attractor',0.,4000.])
