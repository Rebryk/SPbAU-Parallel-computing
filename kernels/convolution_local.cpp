__kernel void matrix_convolution(__global const float *a,
                                 __global const float *b,
                                 __global float *c,
                                 const int n,
                                 const int m) {
    __local float local_a[BLOCK_SIZE + 8][BLOCK_SIZE + 8];
    __local float local_b[9][9];

    int grow = get_global_id(0);
    int gcol = get_global_id(1);

    int lrow = get_local_id(0);
    int lcol = get_local_id(1);

    int xrow = grow - lrow - m / 2;
    int xcol = gcol - lcol - m / 2;

    int id = lrow * BLOCK_SIZE + lcol;

    int N = BLOCK_SIZE + m - 1;
    int BLOCK_SQ = BLOCK_SIZE * BLOCK_SIZE;
    int load_cnt = (N * N + BLOCK_SQ - 1) / BLOCK_SQ;

    // load b to local memory
    if (lrow < m && lcol < m) {
        local_b[lrow][lcol] = b[lrow * m + lcol];
    }

    // load a to local memory
    int loaded = id * load_cnt;
    for (int i = loaded; i < loaded + load_cnt && i < N * N; ++i) {
        int row = xrow + i / N;
        int col = xcol + i % N;

        if (row >= 0 && row < n && col >= 0 && col < n) {
            local_a[i / N][i % N] = a[row * n + col];
        } else {
            local_a[i / N][i % N] = 0;
        }
    }

    // wait all threads
    barrier(CLK_LOCAL_MEM_FENCE);

    if (grow < n && gcol < n) {
        float result = 0;

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                result += local_a[lrow + i][lcol + j] * local_b[i][j];
            }
        }

        c[grow * n + gcol] = result;
    }
}