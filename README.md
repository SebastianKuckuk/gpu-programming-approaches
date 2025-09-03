# GPU Programming Approaches

This repository collects material for the full-day workshop *GPU Programming Approaches*.

## Sections

The tutorial starts with an [introductory presentation](./material/introduction-to-gpu-programming-techniques.pdf) with the following content:
* Basics of GPU architecture
* GPU programming on an abstract level
* Popular approaches to GPU programming

It is followed by more technical content presented along two alternative paths.

### Path One - Comparative Overview (Breadth First)

* [Data Handling](./material/data-handling.ipynb)
  * How to allocate and move data
* [Parallel Computation](./material/parallel-computation.ipynb)
  * How to launch threads on the GPU
  * How to map between work to be done and parallel threads
* [next steps](./material/next-steps.ipynb)
  * Further considerations for choosing between GPU programming approaches
  * Basics of performance evaluation
  * Reductions on GPUs

### Path Two - A Series of Introductions (Depth First)

* [Introduction](./material/introduction.ipynb)
  * Summary: abstract level GPU programming
  * Overview: example applications
* Exploration of different GPU programming approaches - generally the order can be mixed as required
  * [OpenMP Target Offloading](./material/omp-target.ipynb)
  * [OpenACC](./material/openacc.ipynb)
  * [CUDA/ HIP](./material/cuda-hip.ipynb)
  * [Modern C++](./material/modern-cpp.ipynb)
  * [Thrust](./material/thrust.ipynb)
  * [Kokkos](./material/kokkos.ipynb)
  * [SYCL](./material/sycl.ipynb)
* [next steps](./material/next-steps.ipynb)
  * Further considerations for choosing between GPU programming approaches
  * Basics of performance evaluation
  * Reductions on GPUs

### Programming Challenge

The workshop concludes with multi-leveled a [programming challenge](./material/programming-challenge.ipynb).

## Further Learning

Interested in deepening your knowledge about GPU programming?
Check out this selection of the diverse courses portfolio offered by NHR@FAU:
* CUDA
  * [Fundamentals of Accelerated Computing with CUDA C/C++](https://hpc.fau.de/teaching/tutorials-and-courses/#collapse_4)
  * [Accelerating CUDA C++ Applications with Multiple GPUs](https://hpc.fau.de/teaching/tutorials-and-courses/#collapse_7)
  * [Scaling CUDA C++ Applications to Multiple Nodes](https://hpc.fau.de/teaching/tutorials-and-courses/#collapse_8)
* Thrust (and some CUDA): [Fundamentals of Accelerated Computing with Modern CUDA C++](https://hpc.fau.de/teaching/tutorials-and-courses/#collapse_3)
* OpenMP (incl. target offloading): [Introduction to Parallel Programming with OpenMP](https://hpc.fau.de/teaching/tutorials-and-courses/#collapse_14)
* OpenACC: [Fundamentals of Accelerated Computing with OpenACC](https://hpc.fau.de/teaching/tutorials-and-courses/#collapse_6)
* [GPU Performance Engineering](https://hpc.fau.de/teaching/tutorials-and-courses/#collapse_9)
