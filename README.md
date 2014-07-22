ctr
===

Running the ctr scripts requires the following software:

  - Nest (http://www.nest-initiative.org/Software:Installation)
  - Python (http://www.python.org/download)
  - NumPy (http://www.scipy.org/Download)
  - SciPy (http://www.scipy.org/Download)
  - Matplotlib (http://matplotlib.sourceforge.net/) 

Installing Nest:
----------------

Download nest-x.y.z.tar.gz and follow these instructions:
  

```
tar -xzvf nest-x.y.z.tar.gz
mkdir nest-x.y.z-build
cd nest-x.y.z
./bootstrap.sh
./configure --prefix=/path/to/nest-x.y.z-build
make
make install
```

Main file "sim.py"
-----------------

This file reproduces out of the box the following figures from the manuscript "Communication
through resonance in spiking neuronal networks":

- Fig. 1b
- Fig. 3a
- Fig. 4a
- Fig. 5b
- Fig. S6a

To generate the figures you can type something like this:

```
python sim.py --fig_name figure4a
```

Figure names: figure1b, figure3a, figure4a, ... and so on. The program provides the option to adjust 
some simulation parameters when executing the script from the terminal. To see a list of available
parameters you can type:

```
python sim.py --help
```

When executing the scripts as indicated above, many of these parameters will be overwritten. To avoid 
this, you will have to modify the script. Alternatively, you can execute the script without adding 
the --fig_name argument e.g.:

```
python sim.py --activity 20 --sigma 3. --with-vm
```

This will run a simulation with the specified parameters and save the results to ".gdf" and ".dat" files 
in your local directory. It will not generate any figure by default. 



