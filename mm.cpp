// PRL Project 2: Mesh multiplication
// Author: Jakub Bláha (xblaha36)

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kRoot = 0;

/// @brief Reads a matrix from a file, returns true on success, false on failure.
/// @param path path to the file.
/// @param first_is_rows whether the first number in the file is the number of rows of the final
/// matrix. If true, the first number in the file is the number of rows of the final matrix. If
/// false, then the first number in the file is the number of columns of the final matrix.
/// @param data a reference to the vector where the data from the file will be loaded into.
/// @param rows reference to a variable, which will contain the number of rows of the loaded
/// matrix after the function finishes execution.
/// @param cols reference to a variable, which will contain the number of cols of the loaded
/// matrix after the function finishes execution.
/// @return true if the loading was successful, false if not.
bool ReadMatrix(const std::string& path, bool first_is_rows, std::vector<int>& data, int& rows,
                int& cols) {
    // Try to open input file
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Error: cannot open " << path << std::endl;
        return false;
    }

    // Read the header and automatically convert to int
    int header;
    if (!(in >> header) || header <= 0) {
        std::cerr << "Error: invalid number of rows or cols in " << path << std::endl;
        return false;
    }

    // Clear vector with numbers
    data.clear();

    // Load numbers from file
    int v;
    while (in >> v) data.push_back(v);

    // Get the total number of numbers
    const int total = static_cast<int>(data.size());

    // First matrix (first_is_rows = True) has the shape M x N, where M is the number of rows
    // First matrix has M on the first line of the file, which is also
    // the number of rows of the final matrix.

    // Second matrix has the shape N x K, where K is the number of cols
    // Second matrix has K on the first line of the file, which is also
    // the number of cols of the final matrix.

    // If the header number is ROWS of the final matrix, then by dividing the total number of
    // numbers by the number of ROWS, we get N, which is the number of cols of the FIRST matrix
    if (first_is_rows) {
        rows = header;

        if (rows == 0 || total % rows != 0) {
            std::cerr << "Error: " << path << " value count not divisible by row count"
                      << std::endl;
            return false;
        }

        cols = total / rows;
    }

    // If the header number is COLS of the final matrix, then by dividing the total number of
    // numbers by the number of COLS, we get N, which is the number of rows of the SECOND matrix
    else {
        cols = header;

        if (cols == 0 || total % cols != 0) {
            std::cerr << "Error: " << path << " value count not divisible by column count"
                      << std::endl;
            return false;
        }

        rows = total / cols;
    }

    if (rows <= 0 || cols <= 0) {
        std::cerr << "Error: empty matrix in " << path << std::endl;
        return false;
    }

    return true;
}

}  // namespace

int main(int argc, char** argv) {
    // Initialization
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int m = 0, n = 0, k = 0;
    std::vector<int> A, B;

    int abort_flag = 0;

    // If root process, then load matrices from files and make sure
    // they were loaded successfully
    if (world_rank == kRoot) {
        int a_rows, a_cols, b_rows, b_cols;

        const bool ok_a = ReadMatrix("mat1.txt", true, A, a_rows, a_cols);
        const bool ok_b = ReadMatrix("mat2.txt", false, B, b_rows, b_cols);

        if (!ok_a || !ok_b) {
            abort_flag = 1;
        } else if (a_cols != b_rows) {
            std::cerr << "Error: dimension mismatch (A is " << a_rows << "x" << a_cols << ", B is "
                      << b_rows << "x" << b_cols << ")" << std::endl;
            abort_flag = 1;
        } else {
            m = a_rows;
            n = a_cols;
            k = b_cols;
        }
    }

    // Tell all processes if the loading was successfull or not
    MPI_Bcast(&abort_flag, 1, MPI_INT, kRoot, MPI_COMM_WORLD);

    // Every process exits if the loading of matrices was not successful
    if (abort_flag) {
        MPI_Finalize();
        return 1;
    }

    // Root process sets dimensions
    int dims[3];
    if (world_rank == kRoot) {
        dims[0] = m;
        dims[1] = n;
        dims[2] = k;
    }

    // Broadcast dimensions to other processes
    MPI_Bcast(dims, 3, MPI_INT, kRoot, MPI_COMM_WORLD);

    m = dims[0];
    n = dims[1];
    k = dims[2];

    // Make sure that the number of processes matches the matrix dimensions
    if (world_size != m * k) {
        if (world_rank == kRoot) {
            std::cerr << "Error: expected " << m * k << " processes, got " << world_size
                      << std::endl;
        }

        MPI_Finalize();
        return 1;
    }

    // MPI cartesian topology helpers
    int cart_dims[2] = {m, k};
    int periods[2] = {0, 0};

    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, cart_dims, periods, /*reorder=*/0, &cart);

    // Get rank and coords in each process
    int my_rank = 0;
    MPI_Comm_rank(cart, &my_rank);

    int coords[2];
    MPI_Cart_coords(cart, my_rank, 2, coords);

    const int i = coords[0];
    const int j = coords[1];

    // Get ranks of neighbor processes
    int left, right, up, down;
    MPI_Cart_shift(cart, /*direction=*/1, /*disp=*/1, &left, &right);
    MPI_Cart_shift(cart, /*direction=*/0, /*disp=*/1, &up, &down);

    // The "border" processes will store the rows/columns, allocate memory for that
    std::vector<int> a_row, b_col;
    if (j == 0) a_row.resize(n);
    if (i == 0) b_col.resize(n);

    // Data distribution to the allocated vectors
    // root process is either storing the values in its own vector
    // or sending the values to other processes, since the root process
    // is the only one, which has the data stored.
    if (world_rank == kRoot) {
        // Distribute rows/left border
        // for each row... "on the left edge"
        for (int ii = 0; ii < m; ++ii) {
            int dst_coords[2] = {ii, 0};
            int dst_rank;

            MPI_Cart_rank(cart, dst_coords, &dst_rank);

            // Just copy the values to the vector in case we are the root process
            if (dst_rank == my_rank) {
                for (int s = 0; s < n; ++s) a_row[s] = A[ii * n + s];
            }

            // In case we are not the root process, we need to send the values
            // to the correct process
            else {
                MPI_Send(&A[ii * n], n, MPI_INT, dst_rank, /*tag=*/0, cart);
            }
        }

        // We will load the values here into this buffer before sending each column
        std::vector<int> col_buf(n);

        // for each column... "on the top edge"
        for (int jj = 0; jj < k; ++jj) {
            // Popuplate column buffer
            for (int s = 0; s < n; ++s) col_buf[s] = B[s * k + jj];

            int dst_coords[2] = {0, jj};
            int dst_rank;

            MPI_Cart_rank(cart, dst_coords, &dst_rank);

            // If we are the root process, we own the buffer vector and can assign
            // the memory directly
            if (dst_rank == my_rank) {
                b_col = col_buf;
            }

            // Otherwise we need to send it
            else {
                MPI_Send(col_buf.data(), n, MPI_INT, dst_rank, /*tag=*/1, cart);
            }
        }
    }

    // If we are not the root process, need to receive values
    else {
        // ...but only if we are "at the edge"
        if (j == 0) {
            MPI_Recv(a_row.data(), n, MPI_INT, /*source=*/0, /*tag=*/0, cart, MPI_STATUS_IGNORE);
        }

        if (i == 0) {
            MPI_Recv(b_col.data(), n, MPI_INT, /*source=*/0, /*tag=*/1, cart, MPI_STATUS_IGNORE);
        }
    }

    // Do multiplication + summing
    long long c = 0;
    const int total_steps = m + k + n - 2;

    for (int step = 0; step < total_steps; ++step) {
        // Different step for each process, because the computation
        // needs to be shifted
        const int s = step - i - j;

        if (s < 0 || s >= n) continue;

        int a_val, b_val;

        // If at the edge, we just iterate over the values
        // which we will use for the multiplication
        if (j == 0) {
            a_val = a_row[s];
        }
        // Otherwise look for the value from the process on the left
        else {
            MPI_Recv(&a_val, 1, MPI_INT, left, /*tag=*/10, cart, MPI_STATUS_IGNORE);
        }

        // If we are at the edge, we just iterate over the values
        // which we will use for the multiplication
        if (i == 0) {
            b_val = b_col[s];
        }
        // Otherwise we receive the value from the processor at the top
        else {
            MPI_Recv(&b_val, 1, MPI_INT, up, /*tag=*/11, cart, MPI_STATUS_IGNORE);
        }

        // Multiply and add to the result
        c += static_cast<long long>(a_val) * static_cast<long long>(b_val);

        // If not the last processor in the row, send the a_val to the right
        if (right != MPI_PROC_NULL) {
            MPI_Send(&a_val, 1, MPI_INT, right, /*tag=*/10, cart);
        }

        // If not the last processor in the row, send the b_val below
        if (down != MPI_PROC_NULL) {
            MPI_Send(&b_val, 1, MPI_INT, down, /*tag=*/11, cart);
        }
    }

    // Create a result array and prepare for result collection in the root process
    std::vector<long long> result;

    if (my_rank == kRoot) {
        result.resize(static_cast<size_t>(m) * k);
    }

    // Send the c result to the root process
    MPI_Gather(&c, 1, MPI_LONG_LONG, result.data(), 1, MPI_LONG_LONG, kRoot, cart);

    // Print result
    if (my_rank == kRoot) {
        std::ostringstream out;
        out << m << " " << k << "\n";

        for (int ii = 0; ii < m; ++ii) {
            for (int jj = 0; jj < k; ++jj) {
                if (jj) out << " ";
                out << result[ii * k + jj];
            }

            out << "\n";
        }

        std::cout << out.str();
    }

    // Cleanup
    MPI_Comm_free(&cart);
    MPI_Finalize();

    return 0;
}
