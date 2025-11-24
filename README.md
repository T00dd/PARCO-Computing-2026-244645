# Benchmark of SpMV with OpenMP

This repository contains an analysis of the Sparse Matrix-Vector Multiplication (SpMV) algorithm. 
The implementation is in C and parallelized with OpenMP. 

The main goal is to measure the impact of different strategies of scheduling (offered by OpenMP) 
on the efficiency of the parallelization, with different number of threads and types of matrix structures.

### Matrices used for the test

The benchmark is executed on a set of sparse matrices, automatically downloaded from the SuiteSparse Matrix Collection (`sparse.tamu.edu`):

- `twotone` (ATandT) 
- `Transport` (Janna)
- `cage14` (vanHeukelum)
- `torso1` (Norris)
- `memchip` (Freescale)

## Project structure
.
├── include/
│   ├── mmio.h
│   └── timer.h
├── results/
│   └── (contains CSV output for time and hardware profiling)
├── scripts/
│   ├── start_job.pbs
│   ├── start_job.sh
│   ├── perf_plotting.py
│   └── time_plotting.py
├── src/
│   ├── main.c
│   └── mmio.c
├── .gitignore
├── README.md
└── report.pdf

## Requirements and execution enviroments

The project can be run in 2 different enviroments:

#### Cluster (recommended for benchmarks)

This is the intended execution mode for running the complete performance analysis, leveraging the resources of an HPC (High-Performance Computing) system.

- Scheduler: A PBS (Portable Batch System), which is required to interpret the spmv_benchmark.pbs script and use the qsub command.

- Module System: to load specific software dependencies (e.g., module load gcc91, module load perf).

- Internet Access on Compute Nodes: The compute nodes must have wget access to sparse.tamu.edu to download the matrices.

#### Local (recommended for testing or development)

You can compile and run the benchmark on your personal machine to test for correctness, develop new features, or perform smaller-scale measurements.
In this case, you cannot launch the `.pbs` script directly. Instead, you must check the requirments:

1. A C Compiler with OpenMP (e.g., `gcc`)
2. Download Utility: `wget`
3. Archive Utility: `tar`
4. `Perf`: utility for hardware profiling

## How to execute the benchmark

The compilation and download are entirely managed by the PBS script

#### Automatic data download 

You do not need to download the matrices manually. The PBS script automatically checks if each matrix exist in the `data/` folder at the beginning of each execution. If the matrices are not found, the script will automatically download and extract them.

#### Job submission (standard mode)

###### CLUSTER:

To launch the job in the cluster (that will execute all the combinations of number of threads, schedules and chunck sizes for each matrix measuring only the multiplication time), navigate to the `scripts/` folder and use `qsub`:

```bash
qsub start_job.pbs
```
The job will be submitted to the `short_cpuQ` queue (maximum 6 hours), with 32 gb of dedicated RAM and 64 cores.

###### LOCAL:

To launch the job in locally, navigate to the `scripts/` folder and execute the `.sh`:

```bash
./start_job.sh
```
#### Job submission (perf mode)

###### CLUSTER:

To activate hardware profiling (L1/LLC cache misses etc.) on the cluster, you need to pass the `PERF` variable to the PBS script with the `-v` flag:

```bash
qsub -v PERF=1 start_job.pbs 
```
###### LOCAL:

To activate the perf mode locally, just use the `-p` flag with the .sh script:

```bash
./start_job.sh -p
```
## Outputs

All benchmark outputs are saved in the `results/` folder:

- `log.out` and `log.err`: Logs of the output and standard error of the job (only if executed in the cluster).

- `time_results.csv`: (standard mode) contains the execution time (in milliseconds) for each matrix combination.

- `perf_results.csv`: (perf mode) contains the hardware profiling data (L1/LLC cache misses etc.) for each matrix combination.

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

