# Util Files

[alex.a40.def](./alex.a40.def) contains an Apptainer container definition for testing and deployment on the A40 partition of the Alex cluster located at NHR@FAU.

It can be build with
```bash
apptainer build gpu-programming-approaches.sif alex.a40.def
```

and executed on a compute node for up to two hours with
```bash
srun --partition=a40 --nodes=1 --gres=gpu:a40:1 --time 02:00:00 build gpu-programming-approaches.sif
```

which starts a *Jupyter Lab* instance that can be connected to from within the university network or via ssh port forwarding.
Code for starting a *Jupyter Hub* via BatchSpawner is also included in the run section of the container definition, but currently commented out.
