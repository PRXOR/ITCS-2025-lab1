#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#ifdef SIMD
#include <arm_neon.h>
#endif

#include "common.h"
#include "compute.h"

void zero_z() {
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            Z[i][j] = 0;
        }
    }
}

void compute_row_major_mnk() {
    zero_z();
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            for (int l = 0; l != k; ++l) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
}

void compute_row_major_mkn() {
    // TODO: task 1
    zero_z();
    for (int i = 0; i != m; ++i) {
        for (int l = 0; l != k; ++l) {
            for (int j = 0; j != n; ++j) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
}

void compute_row_major_kmn() {
    // TODO: task 1
    zero_z();
    for (int l = 0; l != k; ++l) {
        for (int i = 0; i != m; ++i) {
            for (int j = 0; j != n; ++j) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
}

void compute_row_major_nmk() {
    // TODO: task 1
    zero_z();
    for (int j = 0; j != n; ++j) {
        for (int i = 0; i != m; ++i) {
            for (int l = 0; l != k; ++l) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
}

void compute_row_major_nkm() {
    // TODO: task 1
    zero_z();
    for (int j = 0; j != n; ++j) {
        for (int l = 0; l != k; ++l) {
            for (int i = 0; i != m; ++i) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
}

void compute_row_major_knm() {
    // TODO: task 1
    zero_z();
    for (int l = 0; l != k; ++l) {
        for (int j = 0; j != n; ++j) {
            for (int i = 0; i != m; ++i) {
                Z[i][j] += X[i][l] * Y[l][j];
            }
        }
    }
}

void compute_y_transpose_mnk() {
    // TODO: task 2
    zero_z();
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            for (int l = 0; l != k; ++l) {
                Z[i][j] += X[i][l] * YP[j][l];
            }
        }
    }
    
}

void compute_row_major_mnkkmn_b32() {
    // TODO: task 2
    zero_z();
    const int B = 4;
    for (int ii = 0; ii != m; ii += B)
    {
        for (int jj = 0; jj != n; jj += B)
        {
            for (int ll = 0; ll != k; ll += B)
            {
                for (int l = ll; l != ll + B; ++l)
                {
                    for (int j = jj; j != jj + B; ++j)
                    {
                        for (int i = ii; i != ii + B; ++i)
                        {
                            Z[i][j] += X[i][l] * Y[l][j];
                        }
                    }
                }
            }
        }
    }
}

void compute_row_major_mnk_lu2() {
    // TODO: task 2
    zero_z();
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            uint64_t Z1=0,Z2=0;
            for (int l = 0; l != k; l += 2) {
                Z1 += X[i][l] * Y[l][j];
                Z2 += X[i][l + 1] * Y[l + 1][j];
            }
            Z[i][j]=Z1+Z2;
        }
    }
}

void compute_thunder(){
    zero_z();
    //const uint64_t column = n;
    for (register int i = 0; i != m; ++i) {
        for (register int j = 0; j != n; ++j) {
            register uint64_t Z1=0,Z2=0,Z3=0,Z4=0;
            register uint64_t* X1=&X[i][0];
            register uint64_t* Y1=&YP[j][0];
            for (register int l = 0; l != k; l += 4) {
                Z1 += (*(X1)) * (*(Y1));
                Z2 += (*(X1+1)) * (*(Y1+1));
                Z3 += (*(X1+2)) * (*(Y1+2));
                Z4 += (*(X1+3)) * (*(Y1+3));
                X1+=4;
                Y1+=4;
            }
            Z[i][j]=Z1+Z2+Z3+Z4;
        }
    }
}

void compute_simd() {
#ifdef SIMD
    // TODO: task 3
    zero_z();
    register int kk = k / 4;
    for (register int i = 0; i != m; ++i) {
        for (register int j = 0; j != n; ++j) {
            register uint32x4_t zt = vmovq_n_u32(0);
            //register uint32_t zz = 0;
            for (register int l = 0; l != kk; ++l)
            {
                zt = vmull_u16(XC[i][l],YPC[j][l]);
                Z[i][j] += vaddlvq_u32(zt);
            }
            //Z[i][j] = zz;
        }
    }
#endif
}

uint64_t elapsed(const struct timespec start, const struct timespec end) {
    struct timespec result;
    result.tv_sec = end.tv_sec - start.tv_sec;
    result.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (result.tv_nsec < 0) {
        --result.tv_sec;
        result.tv_nsec += SEC;
    }
    uint64_t res = result.tv_sec * SEC + result.tv_nsec;
    return res;

}

uint64_t compute() {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    switch (COMPUTE_SELECT) {
        case COMPUTE_ROW_MAJOR_MNK:
            //printf("COMPUTE_ROW_MAJOR_MNK\n");
            compute_row_major_mnk();
            break;
        case COMPUTE_ROW_MAJOR_MKN:
            //printf("COMPUTE_ROW_MAJOR_MKN\n");
            compute_row_major_mkn();
            break;
        case COMPUTE_ROW_MAJOR_KMN:
            //printf("COMPUTE_ROW_MAJOR_KMN\n");
            compute_row_major_kmn();
            break;
        case COMPUTE_ROW_MAJOR_NMK:
            //printf("COMPUTE_ROW_MAJOR_NMK\n");
            compute_row_major_nmk();
            break;
        case COMPUTE_ROW_MAJOR_NKM:
            //printf("COMPUTE_ROW_MAJOR_NKM\n");
            compute_row_major_nkm();
            break;
        case COMPUTE_ROW_MAJOR_KNM:
            //printf("COMPUTE_ROW_MAJOR_KNM\n");
            compute_row_major_knm();
            break;
        case COMPUTE_Y_TRANSPOSE_MNK:
            //printf("COMPUTE_Y_TRANSPOSE_MNK\n");
            compute_y_transpose_mnk();
            break;
        case COMPUTE_ROW_MAJOR_MNKKMN_B32:
            //printf("COMPUTE_ROW_MAJOR_MNKKMN_B32\n");
            compute_row_major_mnkkmn_b32();
            break;
        case COMPUTE_ROW_MAJOR_MNK_LU2:
            //printf("COMPUTE_ROW_MAJOR_MNK_LU2\n");
            compute_row_major_mnk_lu2();
            break;
        case COMPUTE_THUNDER:
            compute_thunder();
            break;
        case COMPUTE_SIMD:
            //printf("COMPUTE_SIMD\n");
            compute_simd();
            break;
        default:
            printf("Unreachable!");
            return 0;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    return elapsed(start, end);
}

