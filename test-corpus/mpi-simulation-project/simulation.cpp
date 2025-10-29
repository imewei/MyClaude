
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1000000;
    std::vector<double> data(N / size);

    #pragma omp parallel for
    for (int i = 0; i < data.size(); ++i) {
        data[i] = rank * (N / size) + i;
    }

    double local_sum = 0.0;
    #pragma omp parallel for reduction(+:local_sum)
    for (int i = 0; i < data.size(); ++i) {
        local_sum += data[i];
    }

    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Global sum: " << global_sum << std::endl;
    }

    MPI_Finalize();
    return 0;
}
