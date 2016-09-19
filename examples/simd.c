#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>

#define N 4
#define A(r, c) (A[(r)*N + (c)])


void set(float* A, float* b, float* x) {
    A(0, 0) = 1.0f;
    A(0, 1) = 1337.0f;
    A(0, 2) = 13.0f;
    A(0, 3) = 2.0f;
    A(1, 0) = 14.0f;
    A(1, 1) = 1.0f;
    A(1, 2) = 0.0f;
    A(1, 3) = 12.0f;
    A(2, 0) = 65.0f;
    A(2, 1) = 34.0f;
    A(2, 2) = 45.0f;
    A(2, 3) = -160.0f;
    A(3, 0) = 76.0f;
    A(3, 1) = 23.0f;
    A(3, 2) = -1.0f;
    A(3, 3) = 0.5f;

    b[0] = -1.0f;
    b[1] = 1.0f;
    b[2] = 21.5f;
    b[3] = 42.0f;

    x[0] = 0.0f;
    x[1] = 0.0f;
    x[2] = 0.0f;
    x[3] = 0.0f;
}


void mul(const float* A, const float* b, float* x) {
    for (size_t i = 0; i < N; ++i) {
        x[i] = A(i, 0) * b[0] + A(i, 1) * b[1] + A(i, 2) * b[2] + A(i, 3) * b[3];
    }
}

inline void dot(const float* row, const float* col, float* res) {
    const int mask = 0xff;
    __m128 res_vec = _mm_dp_ps(_mm_load_ps(row), _mm_load_ps(col), mask);
    _mm_store_ss(res, res_vec);
}
void mul_simd(float* A, float* b, float* x) {
    for (size_t i = 0; i < N; ++i) {
        const float* row = &A[N*i];
        const float* col = b;
        float* res = &x[i];
        dot(row, col, res);
    }
}


void check(float* x) {
    assert(x[0] == 1699.5f);
    assert(x[1] == 491.0f);
    assert(x[2] == -5783.5f);
    assert(x[3] == -53.5f);
}


int main(int argc, char const *argv[]) {
    clock_t begin, end;

    float* A = NULL;
    posix_memalign((void **)&A, 16, N * N * sizeof(float));

    float* b = NULL;
    posix_memalign((void **)&b, 16, N * 1 * sizeof(float));

    float* x = NULL;
    posix_memalign((void **)&x, 16, N * 1 * sizeof(float));


    begin = clock();
    for (size_t i = 0; i < 10000000; ++i) {
        set(A, b, x);
        mul(A, b, x);
        check(x);
    }
    end = clock();
    printf("mul:      %f\n", (double)(end - begin) / CLOCKS_PER_SEC);

    begin = clock();
    for (size_t i = 0; i < 10000000; ++i) {
        set(A, b, x);
        mul_simd(A, b, x);
        check(x);
    }
    end = clock();
    printf("mul_simd: %f\n", (double)(end - begin) / CLOCKS_PER_SEC);

    return EXIT_SUCCESS;
}
