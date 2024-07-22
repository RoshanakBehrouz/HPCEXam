module load openMPI/4.1.5/gnu
mpic++ -o mandelbrot mandelbrot.cpp -Wall -Wextra -O3 -march=native -fopenmp -lm