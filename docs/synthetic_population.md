# Synthetic population

GradABM-JUNE uses the synthetic population from the JUNE model. To convert a JUNE synthetic "world" to a PyTorch geometric graph that GradABM-JUNE can recognize, we can use the `scrips/make_data.py` script.

## 1. Create digital population with JUNE.

The first step is to create the digital twin with the JUNE code. We first need to install JUNE (https://github.com/idas-durham/june). Then we can create our own world by using the `create_world.py` script: https://github.com/IDAS-Durham/JUNE/blob/master/example_scripts/create_world.py .

After running the script, the world should have been save in an HDF5 file.

## 2. Convert the JUNE world into a GradABM-JUNE graph.

By using https://github.com/arnauqb/GradABM-JUNE/blob/main/example_scripts/make_data.py we can do

```python
python make_data.py /path/to/world.hdf5
```

This will save a pickle file with the PyTorch graph, that can then be loaded by GradABM-JUNE.
