#define __CL_ENABLE_EXCEPTIONS
#define SIZE(sz) BLOCK_SIZE * ((sz + BLOCK_SIZE - 1) / BLOCK_SIZE)

#include <iostream>
#include <vector>
#include <fstream>
#include "cl.hpp"

using namespace std;

const size_t BLOCK_SIZE = 512;

cl::Program::Sources load_sources(const string &file_name) {
    ifstream source_file(file_name);
    string source_string(istreambuf_iterator<char>(source_file), (istreambuf_iterator<char>()));
    return {{source_string.c_str(), source_string.length() + 1}};
}

void read_data(vector<float> &data) {
    FILE *const input_file = fopen("input.txt", "r");

    if (input_file == NULL) {
        printf("No input file!\n");
        return;
    }

    int n;
    fscanf(input_file, "%d\n", &n);

    data.resize((size_t) n);
    for (int i = 0; i < n; ++i) {
        fscanf(input_file, "%f", &data[i]);
    }

    fclose(input_file);
}

void write_data(const vector<float> &data, const size_t limit) {
    FILE *const output_file = fopen("output.txt", "w");

    for (int i = 0; i < data.size() && i < limit; ++i) {
        fprintf(output_file, "%0.3f ", data[i]);
    }
    fprintf(output_file, "\n");

    fflush(output_file);
    fclose(output_file);
}

vector<float> prefix_sum(cl::Context &context,
                         cl::CommandQueue &queue,
                         cl::Program &program,
                         vector<float> &input) {
    vector<float> output(input.size(), 0);

    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * input.size());
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * output.size());

    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * input.size(), &input[0]);

    cl::Kernel kernel(program, "scan_hillis_steele");
    cl::KernelFunctor scan_hs(kernel, queue, cl::NullRange, cl::NDRange(input.size()), cl::NDRange(BLOCK_SIZE));
    cl::Event event = scan_hs(dev_input,
                              dev_output,
                              cl::__local(sizeof(float) * BLOCK_SIZE),
                              cl::__local(sizeof(float) * BLOCK_SIZE));

    event.wait();

    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * output.size(), &output[0]);

    if (output.size() == BLOCK_SIZE) {
        return output;
    } else {
        vector <float> tails(SIZE(output.size() / BLOCK_SIZE), 0);
        for (int i = 1; i * BLOCK_SIZE - 1 < output.size() && i < tails.size(); ++i) {
            tails[i] = output[i * BLOCK_SIZE - 1];
        }

        vector <float> tails_prefix = prefix_sum(context, queue, program, tails);

        cl::Buffer dev_input_inc(context, CL_MEM_READ_ONLY, sizeof(float) * tails_prefix.size());

        queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * output.size(), &output[0]);
        queue.enqueueWriteBuffer(dev_input_inc, CL_TRUE, 0, sizeof(float) * tails_prefix.size(), &tails_prefix[0]);

        cl::Kernel kernel_inc(program, "inc");
        cl::KernelFunctor inc(kernel_inc, queue, cl::NullRange, cl::NDRange(input.size()), cl::NDRange(BLOCK_SIZE));
        cl::Event event_inc = inc(dev_input, dev_input_inc, dev_output);

        event_inc.wait();

        vector<float> result(input.size(), 0);
        queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * result.size(), &result[0]);

        return result;
    }
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

        cl::Program program(context, load_sources("../scan.cl"));

        try {
            program.build(devices, "-D BLOCK_SIZE=512");
        } catch (cl::Error e) {
            size_t len;
            char buffer[2048];

            printf("Error: Failed to build program executable!\n");
            clGetProgramBuildInfo(program(), devices.front()(), CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
            printf("%s\n", buffer);
            return 1;
        }

        printf("Built program\n");

        vector<float> input;
        read_data(input);

        const size_t n = input.size();
        input.resize(SIZE(n));

        vector <float> output = prefix_sum(context, queue, program, input);
        write_data(output, n);

        printf("Finished\n");
    } catch (cl::Error e) {
        printf("\n%s: %d\n", e.what(), e.err());
    }
    return 0;
}