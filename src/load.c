#include <stdio.h>

#include "common.h"
#include "load.h"

#ifdef SIMD
#include <arm_neon.h>
#endif

int load(const char* file) {
    FILE* fp = fopen(file, "r");
    if (!fp) return 1;

    fscanf(fp, "%d%d%d", &m, &k, &n);
    if (m <= 0 || m > M) return 1;
    if (k <= 0 || k > K) return 1;
    if (n <= 0 || n > N) return 1;

    //switch (LOAD_SELECT) {
    //    case LOAD_ROW_MAJOR:
    //        printf("LOAD_ROW_MAJOR\n");
    //        break;
    //    case LOAD_Y_TRANSPOSE:
    //        printf("LOAD_Y_TRANSPOSE\n");
    //        break;
    //    case LOAD_Y_TRANSPOSE_32:
    //        printf("LOAD_Y_TRANSPOSE_32\n");
    //        break;
    //    case LOAD_Y_TRANSPOSE_16:
    //        printf("LOAD_Y_TRANSPOSE_16\n");
    //        break;
    //    default:
    //        printf("Unreachable!");
    //        return 1;
    //}
    //* read to X
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != k; ++j) {
            switch (LOAD_SELECT) {
                case LOAD_ROW_MAJOR:
                case LOAD_Y_TRANSPOSE:
                    fscanf(fp, "%" SCNx64, &X[i][j]);
                    break;
                case LOAD_Y_TRANSPOSE_32:
                    fscanf(fp, "%" SCNx32, &X32[i][j]);
                    break;
                case LOAD_SIMD:
                case LOAD_Y_TRANSPOSE_16:
                    fscanf(fp, "%" SCNx16, &X16[i][j]);
                    break;
                default:
                    printf("Unreachable!");
                    return 1;
            }
        }
    }
    #ifdef SIMD
    if(LOAD_SELECT == LOAD_SIMD)
    {
        for (register int i = 0; i != m; ++i) {
            for (register int j = 0; j != k; j+=4) {
                uint16_t temp[4];
                temp[0] = X16[i][j];
                temp[1] = X16[i][j+1];
                temp[2] = X16[i][j+2];
                temp[3] = X16[i][j+3];
                XC[i][j/4] = vld1_u16(temp);
            }
        }
    }
    #endif
    //* read to Y
    for (int i = 0; i != k; ++i) {
        for (int j = 0; j != n; ++j) {
            switch (LOAD_SELECT) {
                case LOAD_ROW_MAJOR:
                    fscanf(fp, "%" SCNx64, &Y[i][j]);
                    break;
                case LOAD_Y_TRANSPOSE:
                    fscanf(fp, "%" SCNx64, &YP[j][i]);
                    break;
                case LOAD_Y_TRANSPOSE_32:
                    fscanf(fp, "%" SCNx32, &YP32[j][i]);
                    break;
                case LOAD_SIMD:
                case LOAD_Y_TRANSPOSE_16:
                    fscanf(fp, "%" SCNx16, &YP16[j][i]);
                    break;
                default:
                    printf("Unreachable!");
                    return 1;
            }
        }
    }

    #ifdef SIMD
    if(LOAD_SELECT == LOAD_SIMD)
    {
        for (register int i = 0; i != n; ++i) {
            for (register int j = 0; j != k; j+=4) {
                uint16_t temp[4];
                temp[0] = YP16[i][j];
                temp[1] = YP16[i][j+1];
                temp[2] = YP16[i][j+2];
                temp[3] = YP16[i][j+3];
                YPC[i][j/4] = vld1_u16(temp);
            }
        }
    }
    #endif

    for (int i = 0; i!= m; ++i) {
        for (int j = 0; j!= n; ++j) {
            fscanf(fp, "%" SCNx64, &R[i][j]);
        }
    }
    return 0;
}
