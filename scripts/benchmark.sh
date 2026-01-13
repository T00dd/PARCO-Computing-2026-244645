#!/bin/bash

cd "$PBS_O_WORKDIR/../src"
mkdir -p ../data
mkdir -p ../results

PARALLEL_MODE=0


RESULTS_FILE_WEAK="../results/time_results_weak.csv"
RESULTS_FILE_STRONG="../results/time_results_strong.csv"


matrix_download(){

	local NAME=$1
	local URL=$2
	local TARGET_DIR="../data"
	local CHECK_FILE="$TARGET_DIR/$NAME.mtx"

	if [ -f "$CHECK_FILE" ]; then
		echo "Matrix $NAME found! No download needed"
	else
		echo "Matrix not founded. Starting to download..."

		wget -q "$URL" -O "$TARGET_DIR/$NAME.tar.gz"

		if [ $? -ne 0 ]; then
			echo "ERROR: download $NAME failed!"
			exit 1
		fi

		echo "Starting extraction of $NAME..."
		tar -xzf "$TARGET_DIR/$NAME.tar.gz" -C "$TARGET_DIR"
		        
	    if [ -f "$TARGET_DIR/$NAME/$NAME.mtx" ]; then
		  	mv "$TARGET_DIR/$NAME/$NAME.mtx" "$TARGET_DIR/"
		    rm -rf "$TARGET_DIR/$NAME" 
		    rm "$TARGET_DIR/$NAME.tar.gz" 
		    echo "Matrix $NAME ready to use."
		else
		    echo "ERROR: File .mtx not found after extraction: $NAME"
		    exit 1
		fi
		
	fi
	
}

echo "Checking matrices..."

matrix_download "twotone" "https://sparse.tamu.edu/MM/ATandT/twotone.tar.gz"
matrix_download "HV15R" "https://sparse.tamu.edu/MM/Fluorem/HV15R.tar.gz"
matrix_download "cage15" "https://sparse.tamu.edu/MM/vanHeukelum/cage15.tar.gz"
matrix_download "torso1" "https://sparse.tamu.edu/MM/Norris/torso1.tar.gz"
matrix_download "memchip" "https://sparse.tamu.edu/MM/Freescale/memchip.tar.gz"

MATRICES=(	"../data/twotone.mtx"
			"../data/HV15R.mtx"
			"../data/cage15.mtx"
			"../data/torso1.mtx"
			"../data/memchip.mtx"	)

echo "Compiling with MPI..."


mpicc -O3 -fopenmp -I../include -o spmv_mpi_benchmark ../src/main.c ../src/mmio.c ../src/ghost_entries.c -lm
if [ $? -ne 0 ]; then 
    echo "Compilation failed!"; 
    exit 1; 
fi
echo "Compilation completed!"




echo "matrix,size,time_comunication_ms,time_multiplication_ms,gflops,avg_nnz,global_min_nnz,global_max_nnz,avg_comm,global_min_comm,global_max_comm" > "$RESULTS_FILE_WEAK"
echo "matrix,size,time_comunication_ms,time_multiplication_ms,gflops,avg_nnz,global_min_nnz,global_max_nnz,avg_comm,global_min_comm,global_max_comm" > "$RESULTS_FILE_STRONG"



echo "Starting Strong Scaling..."

	for matrix in "${MATRICES[@]}"; do
		for procs in 1 2 4 8 16 32 64 128; do

			NUM_NODES=$(cat $PBS_NODEFILE | sort -u | wc -l )
			PPN=$((procs / NUM_NODES))
			if [ $PPN -eq 0 ]; then PPN=1; fi

			echo "Strong Scaling: $procs processes"
			mpirun -np $procs --map-by node --bind-to core numactl --interleave=all ./spmv_mpi_benchmark "$matrix" "-ss"
		done
	done

echo "Starting Weak Scaling..."

for procs in 1 2 4 8 16 32 64 128; do
	WEAK_MATRIX="../data/matrix_weak_${procs}.mtx"

	NUM_NODES=$(cat $PBS_NODEFILE | sort -u | wc -l )
	PPN=$((procs / NUM_NODES))
	if [ $PPN -eq 0 ]; then PPN=1; fi

	if [ -f "$WEAK_MATRIX" ]; then
		echo "Weak Scaling: $procs processes with $WEAK_MATRIX"
		mpirun -np $procs --map-by node --bind-to core numactl --interleave=all ./spmv_mpi_benchmark "$WEAK_MATRIX" "-ws"
	else
		echo "Warning: $WEAK_MATRIX not found, skipping."
	fi
done


echo "Benchmarks completed!"