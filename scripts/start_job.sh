#!/bin/bash

cd ../
cd src/

mkdir -p ../data
mkdir -p ../results

PERF_MODE=0
PERF_RESULTS_FILE="../results/perf_results.csv"

while getopts "p" opt; do
  case $opt in
    p)
      echo "Flag -p rilevato: Attivazione modalità PERF"
      PERF_MODE=1 
      ;;
    \?)
      echo "Errore: Opzione -$OPTARG non valida." >&2
      exit 1
      ;;
  esac
done

if [ "$PERF" -eq 1 ]; then
	PERF_MODE=1
fi

#download matrices

matrix_download(){

	local NAME=$1
	local URL=$2
	local TARGET_DIR="../data"
	local CHECK_FILE="$TARGET_DIR/$NAME.mtx"

	if [ -f "$CHECK_FILE" ]; then
		echo "Matrix $NAME found! No download needed"
	else
		echo "Matrix not founded. Starting to download..."

		wget -q --show-progress "$URL" -O "$TARGET_DIR/$NAME.tar.gz"

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



echo "Compiling and executing benchmark..."


gcc -O3 -fopenmp -I../include -o spmv_benchmark main.c mmio.c
if [ $? -ne 0 ]; then
	echo "ERR: compilation failed!"
	exit 1
fi
echo "Compilation completed!"

#definition of all the combinations

MATRICES=(	"../data/twotone.mtx"
			"../data/Transport.mtx"
			"../data/cage14.mtx"
			"../data/torso1.mtx"
			"../data/memchip.mtx"	)
			
THREADS=(1 2 4 8 16 32 64)
SCHEDULES=(static dynamic guided)
CHUNKS=(1 10 100 1000 10000)

RESULTS_FILE="../results/"




echo "Starting benchmark..."

if [ "$PERF_MODE" -eq 0 ]; then

	echo "matrix, threads, schedules type, chunk size, time (ms)" > "$RESULTS_FILE/time_results.csv"


	for matrix in "${MATRICES[@]}"; do
		for threads in "${THREADS[@]}"; do
			for schedule in "${SCHEDULES[@]}"; do
				for chunk in "${CHUNKS[@]}"; do
					if [ "$threads" -eq 1 ] && { [ "$schedule" != "static" ] || [ "$chunk" -ne 1 ]; }; then
						continue
					fi
									
					echo "Running 10 times: $matrix, $threads, $schedule, $chunk"
					
					
						./spmv_benchmark "$matrix" "$threads" "$schedule" "$chunk"
					                    
					    if [ $? -ne 0 ]; then
					    	echo "ERRORE durante l'esecuzione (Run $i): $matrix, $threads, $schedule, $chunk"
					    fi
					
				done
			done
		done
	done

else

	echo "matrix,threads,schedule,chunk,L1_dcache_loads,L1_dcache_load_misses,LLC_loads,LLC_misses,time_ms" > "$PERF_RESULTS_FILE"

	for matrix in "${MATRICES[@]}"; do

		matrix_name=$(basename "$matrix" .mtx)

		for threads in "${THREADS[@]}"; do
			for schedule in "${SCHEDULES[@]}"; do
				for chunk in "${CHUNKS[@]}"; do

					if [ "$threads" -eq 1 ] && { [ "$schedule" != "static" ] || [ "$chunk" -ne 1 ]; }; then
						continue
					fi
									
					echo "Running perf 5 times: $matrix, $threads, $schedule, $chunk"
					
					for ((i=0; i<5; i++)); do
							PERF_OUTPUT=$(perf stat -x, -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-misses \
                            ./spmv_benchmark "$matrix" "$threads" "$schedule" "$chunk" "$TEMP_TIME_FILE" 2>&1)					    
						

						PERF_STATS=$(echo "$PERF_OUTPUT" | awk -F, '
                            /L1-dcache-loads/       { val1 = $1 }
                            /L1-dcache-load-misses/ { val2 = $1 }
                            /LLC-loads/             { val3 = $1 }
                            /LLC-misses/            { val4 = $1 }
                            END { printf "%s,%s,%s,%s", val1, val2, val3, val4 }
                        ')

						echo "$matrix_name,$threads,$schedule,$chunk,$PERF_STATS,$EXEC_TIME" >> "$PERF_RESULTS_FILE"

						if [ $? -ne 0 ]; then
					    	echo "ERRORE durante l'esecuzione perf (Run $i): $matrix, $threads, $schedule, $chunk"
					    fi
					done

					rm -f "$TEMP_TIME_FILE"
				done
			done
		done
	done

fi

echo "Benchmark completed. Time results and Perf results saved in $RESULTS_FILE"