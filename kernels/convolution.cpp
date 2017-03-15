__kernel void matrix_convolution(__global const float *a,
                                 __global const float *b,
                                 __global float *c,
                                 const int n,
                                 const int m) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row >= n || col >= n) {
        return;
    }

    float result = 0;

    int hm = m / 2;
    int new_row = 0;
    int new_col = 0;

    for (int i = -hm; i <= hm; ++i) {
        for (int j = -hm; j <= hm; ++j) {
            new_row = row + i;
            new_col = col + j;

            if (new_row < 0 || new_row >= n || new_col < 0 || new_col >= n) {
                continue;
            }

            result += a[new_row * n + new_col] * b[(hm + i) * m + hm + j];
        }
    }

    c[row * n + col] = result;
}