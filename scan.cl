#define SWAP(a, b) {__local float * tmp=a; a=b; b=tmp;}

__kernel void scan_hillis_steele(__global const float * input, __global float * output, __local float * a, __local float * b) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int block_size = get_local_size(0);

    a[lid] = b[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 1; s < block_size; s <<= 1) {
        if (lid > (s - 1)) {
            b[lid] = a[lid] + a[lid - s];
        } else {
            b[lid] = a[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a, b);
    }

    output[gid] = a[lid];
}

__kernel void inc(__global const float * input, __global const float * add, __global float * output) {
    int gid = get_global_id(0);
    int block_size = get_local_size(0);

    output[gid] = input[gid] + add[gid / block_size];
}