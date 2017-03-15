#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <fstream>
#include "cl.hpp"

using namespace std;

cl::Program::Sources load_sources(const string &file_name) {
    ifstream source_file(file_name);
    string source_string(istreambuf_iterator<char>(source_file), (istreambuf_iterator<char>()));
    return {{source_string.c_str(), source_string.length() + 1}};
}

void read_data(float *&a, size_t &n, float *&b, size_t &m) {
    FILE *const input_file = fopen("input.txt", "r");

    if (input_file == NULL) {
        printf("No input file!\n");
        return;
    }

    fscanf(input_file, "%d %d\n", &n, &m);

    a = new float[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fscanf(input_file, "%f", &a[i * n + j]);
        }
    }

    b = new float[m * m];
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            fscanf(input_file, "%f", &b[i * m + j]);
        }
    }

    fclose(input_file);
}

void write_data(const float *const data, const size_t n) {
    FILE *const output_file = fopen("output.txt", "w");

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fprintf(output_file, "%0.3f ", data[i * n + j]);
        }
        fprintf(output_file, "\n");
    }

    fflush(output_file);
    fclose(output_file);
}

void generate_data(const int n, const int m) {
    FILE *const output_file = fopen("input.txt", "w");

    fprintf(output_file, "%d %d\n", n, m);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fprintf(output_file, "%d ", 1);
        }
        fprintf(output_file, "\n");
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            fprintf(output_file, "%d ", 1);
        }
        fprintf(output_file, "\n");
    }

    fflush(output_file);
    fclose(output_file);
}

int main() {
    vector<cl::Platform> platforms;
    vector<cl::Device> devices;
    vector<cl::Kernel> kernels;

    try {
        cl::Platform::get(&platforms);

        if (platforms.empty()) {
            printf("Not found any platform!\n");
            return 1;
        }

        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        if (devices.empty()) {
            printf("Not found any device!");
            return 1;
        }

        cl::Context context(devices);
        cl::CommandQueue queue(context, devices.front(), CL_QUEUE_PROFILING_ENABLE);

        cl::Program program(context, load_sources("../kernels/convolution_local.cpp"));

        try {
            program.build(devices, "-D BLOCK_SIZE=16");
        } catch (cl::Error e) {
            size_t len;
            char buffer[2048];

            printf("Error: Failed to build program executable!\n");
            clGetProgramBuildInfo(program(), devices.front()(), CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
            printf("%s\n", buffer);
            return 1;
        }

        printf("Built program\n");

        float *a, *b;
        size_t n, m;
        read_data(a, n, b, m);
        float c[n * n];

        const size_t block_size = 16;
        const size_t matrix_size_a = n * n;
        const size_t matrix_size_b = m * m;
        const size_t matrix_size_c = n * n;

        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(float) * matrix_size_a);
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(float) * matrix_size_b);
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(float) * matrix_size_c);

        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(float) * matrix_size_a, a);
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(float) * matrix_size_b, b);

        const size_t N = block_size * ((n + block_size - 1) / block_size);
        cl::Kernel kernel(program, "matrix_convolution");
        cl::KernelFunctor matrix_convolution(kernel, queue, cl::NullRange, cl::NDRange(N, N),
                                             cl::NDRange(block_size, block_size));
        printf("Created kernel functor\n");

        matrix_convolution(dev_a, dev_b, dev_c, (int) n, (int) m);
        printf("matrix_convolution completed\n");

        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(float) * matrix_size_c, c);
        printf("Read buffer\n");

        write_data(c, n);
        printf("Finished\n");
    } catch (cl::Error e) {
        printf("\n%s: %d\n", e.what(), e.err());
    }
    return 0;
}