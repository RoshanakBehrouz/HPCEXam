#include <iostream>
#include <fstream>
#include <complex>
#include <mpi.h>
#include <omp.h>
#include <vector>

int compute_c(double x, double y, int max_iterations) {
    std::complex<double> z(0, 0);
    std::complex<double> c(x, y);
    int iter = 0;
    while (iter < max_iterations && std::abs(z) < 2.0) {
        z = z * z + c;
        iter++;
    }

    if (iter == max_iterations) {
        return 0;
    }

    return (255 * iter / max_iterations);
}

void save_pgm(const std::vector<short int>& image, int width, int height, const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return;
    }

    file << "P2\n" << width << " " << height << "\n255\n";

    for (long unsigned int i = 0; i < image.size(); i++) {
        file << image[i] << " ";
        if ((i + 1) % width == 0) {
            file << std::endl;
        }
    }

    file.close();
}


int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0] << " <x_min> <x_max> <y_min> <y_max> <width> <height> <max_iterations>" << std::endl;
        return 1;
    }

    double arg_x_min = std::stod(argv[1]);
    double arg_x_max = std::stod(argv[2]);
    double arg_y_min = std::stod(argv[3]);
    double arg_y_max = std::stod(argv[4]);
    int arg_width = std::stoi(argv[5]);
    int arg_height = std::stoi(argv[6]);
    int arg_max_iterations = std::stoi(argv[7]);

    int image_n_pixels = arg_width * arg_height;

    MPI_Init(&argc, &argv);
    int rank, num_processes, num_workers;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    num_workers = num_processes - 1;
    if (num_workers < 1) {
        std::cerr << "At least 1 worker is required" << std::endl;
        MPI_Finalize();
        return 1;
    }

    int n_pixels_per_worker = (image_n_pixels + num_workers - 1) / num_workers; // ceils image_n_pixels/num_workers

    if (rank == 0) { // master
        double start_time = MPI_Wtime();
        std::vector<short int> image(image_n_pixels, 0);
        for (int i = 1; i <= num_workers; i++) {
            std::vector<short int> computed_iter(n_pixels_per_worker);
            MPI_Recv(computed_iter.data(), n_pixels_per_worker, MPI_SHORT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int j = 0; j < n_pixels_per_worker; j++) {
                if (computed_iter[j] == -1) {
                    break;
                }
                int idx = i + j * num_workers - 1;
                if (idx < image_n_pixels) {
                    image[idx] = computed_iter[j];
                }
            }
        }
        double end_time = MPI_Wtime();
        std::cout << end_time - start_time << std::endl;
        save_pgm(image, arg_width, arg_height, "output.pgm");
    } else { // worker
        std::vector<short int> computed_iter(n_pixels_per_worker, -1);

        #pragma omp parallel for schedule(dynamic, 1024)
        for (int i=0; i<n_pixels_per_worker; i++) {
            int idx = rank + i * num_workers - 1;
            if (idx < image_n_pixels) {
                int pixel_x = idx % arg_width;
                int pixel_y = idx / arg_width;
                double x = arg_x_min + pixel_x * (arg_x_max - arg_x_min) / arg_width;
                double y = arg_y_min + pixel_y * (arg_y_max - arg_y_min) / arg_height;
                computed_iter[i] = compute_c(x, y, arg_max_iterations);
            }
        }

        MPI_Send(computed_iter.data(), n_pixels_per_worker, MPI_SHORT, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}