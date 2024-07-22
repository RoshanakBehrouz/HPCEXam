#!/bin/bash

#SBATCH --nodes=2
#SBATCH --time=120
#SBATCH --account=dssc
#SBATCH --partition=THIN
#SBATCH --exclusive
#SBATCH --job-name=bcast_benchmark
#SBATCH --output=bcast_benchmark.out
module load openMPI/4.1.5/gnu

for np in $(seq 2 2 48) # 2 4 6 ... 48
do
    for alg in 0 1 2 5 # default, basic linear, chain, binary tree
    do
        mpirun -np $np --map-by core --mca coll_tuned_use_dynamic_rules true --mca coll_tuned_bcast_algorithm $alg ./osu/osu_bcast --full --iterations 10000 --warmup 1000 --message-size 1:1048576 -f csv >> ./output/bcast-np$np-a$alg.csv
    done
done
