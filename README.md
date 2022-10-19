NeuroDevSim  
----------------  

**NeuroDevSim** is a Neural Development Simulator for Linux of MacOS environments. It is a parallel and phenomenological computational framework to grow large numbers of  neuronal morphologies (and resultant microcircuits) simultaneously according to growth-rules expressed in terms of interactions with the environment and internal variables:  

- **Growth of 3D neuronal morphologies**: neurons grow starting from a soma that can sprout dendrites and axons. Growth is embodied in *fronts* which mimic the functionality of growth cones: they can elongate, branch or terminate.    
- **Cell migration**: somata can migrate before sprouting dendrites. They can make a trailing axon during this migration.  
- **Microcircuits**: neurons are generated together in a simulated volume. With the addition of connections rules and synapses circuits emerge.  
- **Interactions**: growth can be influenced by environmental cues. Most obvious is guidance through repulsion or attraction by other neurons or chemical cues. Existing structures can block growth: physical overlap between structures is not allowed.  

Access the **Documentation** at https://cns-oist.github.io/NeuroDevSim/index.html or locally as <http:docs/build/html/index.html>

This software is described in the following preprint, please cite it when using NeuroDevSim:  
E. De Schutter: Efficient simulation of neural development using shared memory parallelization. https://www.biorxiv.org/content/10.1101/2022.10.17.512465v1

