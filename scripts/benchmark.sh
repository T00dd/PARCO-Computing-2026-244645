#!/bin/bash

cd "$PBS_O_WORKDIR/../src"
mkdir -p ../data
mkdir -p ../results

PARALLEL_MODE=0

if [ "$PARAL" -eq 1 ]; then
	PARALLEL_MODE=1
	RESULTS_FILE_WEAK="../results/time_results_weak_parall.csv"
	RESULTS_FILE_STRONG="../results/time_results_strong_parall.csv"
else
	RESULTS_FILE_WEAK="../results/time_results_weak.csv"
	RESULTS_FILE_STRONG="../results/time_results_strong.csv"

fi

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
matrix_download "Transport" "https://sparse.tamu.edu/MM/Janna/Transport.tar.gz"
matrix_download "cage14" "https://sparse.tamu.edu/MM/vanHeukelum/cage14.tar.gz"
matrix_download "torso1" "https://sparse.tamu.edu/MM/Norris/torso1.tar.gz"
matrix_download "memchip" "https://sparse.tamu.edu/MM/Freescale/memchip.tar.gz"

MATRICES=(	"../data/twotone.mtx"
			"../data/Transport.mtx"
			"../data/cage14.mtx"
			"../data/torso1.mtx"
			"../data/memchip.mtx"	)

echo "Compiling with MPI and OpenMP..."


mpicc -O3 -fopenmp -I../include -o spmv_mpi_benchmark ../src/main.c ../src/mmio.c ../src/ghost_entries.c -lm
if [ $? -ne 0 ]; then 
    echo "Compilation failed!"; 
    exit 1; 
fi
echo "Compilation completed!"




echo "matrix,size,time_comunication_ms,time_multiplication_ms,mflops" > "$RESULTS_FILE_WEAK"
echo "matrix,size,time_comunication_ms,time_multiplication_ms,mflops" > "$RESULTS_FILE_STRONG"



echo "Starting Strong Scaling..."

if [ "$PARALLEL_MODE" -eq 1 ]; then
	for matrix in "${MATRICES[@]}"; do
		for procs in 1 2 4 8 16 32 64 128; do
			echo "Strong Scaling: $procs processes"
			mpirun -np $procs ./spmv_mpi_benchmark "$matrix" "-ss" "-on"
		done
	done
else
	for matrix in "${MATRICES[@]}"; do
		for procs in 1 2 4 8 16 32 64 128; do
			echo "Strong Scaling: $procs processes"
			mpirun -np $procs ./spmv_mpi_benchmark "$matrix" "-ss"
		done
	done
fi

echo "Starting Weak Scaling..."

if [ "$PARALLEL_MODE" -eq 0 ]; then
	for procs in 1 2 4 8 16 32 64 128; do
		WEAK_MATRIX="../data/matrix_weak_${procs}.mtx"
		if [ -f "$WEAK_MATRIX" ]; then
			echo "Weak Scaling: $procs processes with $WEAK_MATRIX"
			mpirun -np $procs ./spmv_mpi_benchmark "$WEAK_MATRIX" "-ws"
		else
			echo "Warning: $WEAK_MATRIX not found, skipping."
		fi
	done
fi

echo "Benchmarks completed!"