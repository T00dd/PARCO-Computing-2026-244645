# Benchmark of SpMV with OpenMP

This repository contains an analysis of the Sparse Matrix-Vector Multiplication (SpMV) algorithm. 
The implementation is in C and parallelized with MPI. 

The main goal is to measure the efficiency and the scalability of the algorithm on an HPC cluster through Strong and Weak Scaling tests with different types of matrices.

### Matrices used for the test

The strong scaling benchmark is executed on a set of sparse matrices, automatically downloaded from the SuiteSparse Matrix Collection (`sparse.tamu.edu`):

- `twotone` (ATandT) 
- `Transport` (Janna)
- `cage14` (vanHeukelum)
- `torso1` (Norris)
- `memchip` (Freescale)

While the weak scaling benchmark is executed on synthetic matrices generated ad-hoc with the `matrices_creator.py` python script. These matrices are specifically designed to scale the workload proportionally with the number of processes, ensuring that the local memory footprint per MPI rank remains constant.

The script `matrices_creator.py` must be executed only in a local environment.

> [!NOTE]
> The generation process may take several minutes to create the full set of matrices required for the test.

#### Dependencies

To run the script, ensure your local environment has:

1. Python 3
2. `scipy`
3. `numpy`

## Project structure
```
.  
├── data/
│   ├── syntethic matrices (for weak scaling)
│   └── matrices for strong scaling (after the execution of the .sh)
├── include/  
│   ├── mmio.h  
│   └── timer.h  
├── plots/  
│   └── (contains plots from plotting scripts) 
├── results/  
│   └── (contains CSV output for weak and strong scaling)  
├── scripts/  
│   ├── start_job.pbs  
│   ├── benchmark.sh  
│   ├── local_bench_test.sh  
│   ├── matrices_creator.py 
│   ├── strong_scaling_plot.py  
│   └── weak_scaling_plot.py  
├── src/
│   ├── ghost_entries.c   
│   ├── main.c  
│   └── mmio.c  
├── .gitignore  
├── README.md  
└── report.pdf  
```

## Requirements and execution enviroments

The project can be run in 2 different enviroments:

#### Cluster (recommended for benchmarks)

This is the intended execution mode for running the complete performance analysis, leveraging the resources of an HPC (High-Performance Computing) system.

- Scheduler: A PBS (Portable Batch System), which is required to interpret the spmv_benchmark.pbs script and use the qsub command.

- Module System: to load specific software dependencies (e.g., module load gcc91, module load mpich-3.2.1--gcc-9.1.0).

- Internet Access on Compute Nodes: The compute nodes must have wget access to sparse.tamu.edu to download the matrices.

#### Local (recommended for testing or development)

You can compile and run the benchmark on your personal machine to test for correctness, develop new features, or perform smaller-scale measurements.
In this case, you cannot launch the `.pbs` script directly. Instead, you must check the requirments:

1. A `mpicc` compiler
2. Download Utility: `wget`
3. Archive Utility: `tar`

## How to execute the benchmark

The compilation and download are entirely managed by the `benchmark.sh` script

#### Automatic data download 

You do not need to download the matrices manually. The PBS script automatically checks if each matrix exist in the `data/` folder at the beginning of each execution. If the matrices are not found, the script will automatically download and extract them.

#### Job submission

###### CLUSTER:

To launch the job in the cluster (that will execute all the combinations of number of processes for each matrix, measuring the comunication and multiplication time), navigate to the `scripts/` folder and use `qsub`:

```bash
qsub start_job.pbs
```
The job will be submitted to the `short_cpuQ` queue (maximum 6 hours). The script is set to execute the program on 4 nodes with 32 gb of dedicated RAM each. Every node has 64 cores.

###### LOCAL:

To launch the job in locally, navigate to the `scripts/` folder make sure that the file is executable and execute the `.sh`:

```bash
chmod +x local_bench_test.sh
./local_bench_test.sh
```

## Outputs

The produced CSV files in the results/ folder include:

- Size: number of MPI processes used

- Time Communication: time spent in the ghost entries exchange (ms)

- Time Multiplication: time spent in the local SpMV calculation (ms)

- GFLOPS: floating-point operations per second

## Plotting scripts

The plotting scripts located in the `scripts/` folder (`perf_plotting.py` and `time_plotting.py`) must not be executed on the cluster.
They are intended only for local execution, since they rely on graphical libraries (Matplotlib, Seaborn) that are typically not available or not supported on HPC compute nodes.

When executed locally, each script automatically generates all the figures and saves them in the `plots/` directory. This folder is created if it does not already exist.

Before running the plotting scripts, make sure your local environment includes the following dependencies:

1. Python 3
2. `pandas`
3. `matplotlib`
4. `seaborn`
5. `numpy`

