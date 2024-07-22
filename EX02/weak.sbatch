#!/bin/bash

#SBATCH --job-name=weak_scalability
#SBATCH --nodes=4
#SBATCH --time=60
#SBATCH --account=dssc
#SBATCH --partition=THIN
#SBATCH --exclusive
#SBATCH --output=weak_scalability.out
module load openMPI/4.1.5/gnu

X_MIN=-2.0
X_MAX=2.0
Y_MIN=-2.0
Y_MAX=2.0
WIDTH=4096
HEIGHT=256 # this will be multiplied by the number of processes/threads
MAX_ITERATIONS=1000

output_file="./output/MPI_weak_scalability.csv"
echo "np,height,time" > $output_file

# MPI
for np in $(seq 2 2 96)
do
    seconds=$(mpirun -np ${np} --map-by socket --bind-to socket -x OMP_NUM_THREADS=1 ./mandelbrot $X_MIN $X_MAX $Y_MIN $Y_MAX $WIDTH $((HEIGHT * np)) $MAX_ITERATIONS)
    echo "${np},$((HEIGHT * np)),${seconds}" >> $output_file
done


output_file="./output/OMP_weak_scalability.csv"
echo "n_threads,height,time" > $output_file

# OpenMP
for n_threads in $(seq 1 2 23)
do
    seconds=$(mpirun -np 2 --map-by socket --bind-to socket -x OMP_NUM_THREADS=${n_threads} ./mandelbrot $X_MIN $X_MAX $Y_MIN $Y_MAX $WIDTH $((HEIGHT * n_threads)) $MAX_ITERATIONS)
    echo "${n_threads},$((HEIGHT * n_threads)),${seconds}" >> $output_file
done