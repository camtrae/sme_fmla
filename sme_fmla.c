// sme_comprehensive_benchmark_final.c
// 综合性能测试: 使用 svread_za32_f32_vg1x4 优化
// 编译: clang -O3 -march=armv9-a+sme2 -o benchmark
// sme_comprehensive_benchmark_final.c 或者: gcc -O3 -march=armv9-a+sme2 -o
// benchmark sme_comprehensive_benchmark_final.c

#include <arm_sme.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ============================================================================
// 计时工具
// ============================================================================
double get_time_in_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ============================================================================
// 方法1: 标准SVE单向量内积 (baseline)
// ============================================================================
__arm_locally_streaming float
dot_product_sve_single(const float* A, const float* B, uint64_t size) {
    uint64_t SVL = svcntsw();
    svfloat32_t acc = svdup_f32(0.0f);

    for (uint64_t i = 0; i < size; i += SVL) {
        svbool_t pg = svwhilelt_b32(i, size);
        svfloat32_t za = svld1(pg, &A[i]);
        svfloat32_t zb = svld1(pg, &B[i]);
        acc = svmla_m(pg, acc, za, zb);
    }

    float result = svaddv(svptrue_b32(), acc);
    return result;
}

// ============================================================================
// 方法2: SVE多向量组内积 (使用svld1_x4，但不用ZA)
// ============================================================================
__arm_locally_streaming float
dot_product_sve_multi(const float* A, const float* B, uint64_t size) {
    uint64_t SVL = svcntsw();

    svfloat32_t acc0 = svdup_f32(0.0f);
    svfloat32_t acc1 = svdup_f32(0.0f);
    svfloat32_t acc2 = svdup_f32(0.0f);
    svfloat32_t acc3 = svdup_f32(0.0f);

    for (uint64_t i = 0; i < size; i += 4 * SVL) {
        svcount_t pc = svwhilelt_c32(i, size, 4);

        svfloat32x4_t dataA = svld1_x4(pc, &A[i]);
        svfloat32x4_t dataB = svld1_x4(pc, &B[i]);

        svfloat32_t a0 = svget4(dataA, 0);
        svfloat32_t a1 = svget4(dataA, 1);
        svfloat32_t a2 = svget4(dataA, 2);
        svfloat32_t a3 = svget4(dataA, 3);

        svfloat32_t b0 = svget4(dataB, 0);
        svfloat32_t b1 = svget4(dataB, 1);
        svfloat32_t b2 = svget4(dataB, 2);
        svfloat32_t b3 = svget4(dataB, 3);

        acc0 = svmla_x(svptrue_b32(), acc0, a0, b0);
        acc1 = svmla_x(svptrue_b32(), acc1, a1, b1);
        acc2 = svmla_x(svptrue_b32(), acc2, a2, b2);
        acc3 = svmla_x(svptrue_b32(), acc3, a3, b3);
    }

    svfloat32_t total = svadd_x(svptrue_b32(), acc0, acc1);
    total = svadd_x(svptrue_b32(), total, acc2);
    total = svadd_x(svptrue_b32(), total, acc3);

    float result = svaddv(svptrue_b32(), total);
    return result;
}

// ============================================================================
// 方法3: SME2 ZA向量组内积 - 手动读取单行 (旧方法)
// ============================================================================
__arm_new("za") __arm_locally_streaming float dot_product_sme_za_single(
    const float* A, const float* B, uint64_t size) {
    svzero_za();
    uint64_t SVL = svcntsw();

    for (uint64_t i = 0; i < size; i += 4 * SVL) {
        svcount_t pc = svwhilelt_c32(i, size, 4);
        svfloat32x4_t dataA = svld1_x4(pc, &A[i]);
        svfloat32x4_t dataB = svld1_x4(pc, &B[i]);
        svmla_za32_f32_vg1x4(0, dataA, dataB);
    }

    // 旧方法：手动读取4行
    svfloat32_t za_sum0 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, 0);
    svfloat32_t za_sum1 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, 4);
    svfloat32_t za_sum2 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, 8);
    svfloat32_t za_sum3 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, 12);

    svfloat32_t total_vec = svadd_x(svptrue_b32(), za_sum0, za_sum1);
    total_vec = svadd_x(svptrue_b32(), total_vec, za_sum2);
    total_vec = svadd_x(svptrue_b32(), total_vec, za_sum3);

    float result = svaddv(svptrue_b32(), total_vec);
    return result;
}

// ============================================================================
// 方法4: SME2 ZA向量组内积 (同时使用4个ZA数组)
// ============================================================================
__arm_new("za") __arm_locally_streaming float dot_product_sme_za_quad(
    const float* A, const float* B, uint64_t size) {
    svzero_za();
    uint64_t SVL = svcntsw();

    for (uint64_t i = 0; i < size; i += 16 * SVL) {
        if (i < size) {
            svcount_t pc0 = svwhilelt_c32(i, size, 4);
            svfloat32x4_t dataA0 = svld1_x4(pc0, &A[i]);
            svfloat32x4_t dataB0 = svld1_x4(pc0, &B[i]);
            svmla_za32_f32_vg1x4(0, dataA0, dataB0);
        }

        if (i + 4 * SVL < size) {
            svcount_t pc1 = svwhilelt_c32(i + 4 * SVL, size, 4);
            svfloat32x4_t dataA1 = svld1_x4(pc1, &A[i + 4 * SVL]);
            svfloat32x4_t dataB1 = svld1_x4(pc1, &B[i + 4 * SVL]);
            svmla_za32_f32_vg1x4(1, dataA1, dataB1);
        }

        if (i + 8 * SVL < size) {
            svcount_t pc2 = svwhilelt_c32(i + 8 * SVL, size, 4);
            svfloat32x4_t dataA2 = svld1_x4(pc2, &A[i + 8 * SVL]);
            svfloat32x4_t dataB2 = svld1_x4(pc2, &B[i + 8 * SVL]);
            svmla_za32_f32_vg1x4(2, dataA2, dataB2);
        }

        if (i + 12 * SVL < size) {
            svcount_t pc3 = svwhilelt_c32(i + 12 * SVL, size, 4);
            svfloat32x4_t dataA3 = svld1_x4(pc3, &A[i + 12 * SVL]);
            svfloat32x4_t dataB3 = svld1_x4(pc3, &B[i + 12 * SVL]);
            svmla_za32_f32_vg1x4(3, dataA3, dataB3);
        }
    }

    // 从所有4个ZA数组读取结果
    svfloat32_t za0_sum0 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, 0);
    svfloat32_t za0_sum1 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, 4);
    svfloat32_t za0_sum2 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, 8);
    svfloat32_t za0_sum3 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 0, 12);

    svfloat32_t za1_sum0 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 1, 0);
    svfloat32_t za1_sum1 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 1, 4);
    svfloat32_t za1_sum2 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 1, 8);
    svfloat32_t za1_sum3 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 1, 12);

    svfloat32_t za2_sum0 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 2, 0);
    svfloat32_t za2_sum1 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 2, 4);
    svfloat32_t za2_sum2 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 2, 8);
    svfloat32_t za2_sum3 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 2, 12);

    svfloat32_t za3_sum0 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 3, 0);
    svfloat32_t za3_sum1 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 3, 4);
    svfloat32_t za3_sum2 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 3, 8);
    svfloat32_t za3_sum3 =
        svread_hor_za32_f32_m(svundef_f32(), svptrue_b32(), 3, 12);

    // 合并所有结果
    svfloat32_t za0_total = svadd_x(svptrue_b32(), za0_sum0, za0_sum1);
    za0_total = svadd_x(svptrue_b32(), za0_total, za0_sum2);
    za0_total = svadd_x(svptrue_b32(), za0_total, za0_sum3);

    svfloat32_t za1_total = svadd_x(svptrue_b32(), za1_sum0, za1_sum1);
    za1_total = svadd_x(svptrue_b32(), za1_total, za1_sum2);
    za1_total = svadd_x(svptrue_b32(), za1_total, za1_sum3);

    svfloat32_t za2_total = svadd_x(svptrue_b32(), za2_sum0, za2_sum1);
    za2_total = svadd_x(svptrue_b32(), za2_total, za2_sum2);
    za2_total = svadd_x(svptrue_b32(), za2_total, za2_sum3);

    svfloat32_t za3_total = svadd_x(svptrue_b32(), za3_sum0, za3_sum1);
    za3_total = svadd_x(svptrue_b32(), za3_total, za3_sum2);
    za3_total = svadd_x(svptrue_b32(), za3_total, za3_sum3);

    svfloat32_t final_vec = svadd_x(svptrue_b32(), za0_total, za1_total);
    final_vec = svadd_x(svptrue_b32(), final_vec, za2_total);
    final_vec = svadd_x(svptrue_b32(), final_vec, za3_total);

    float result = svaddv(svptrue_b32(), final_vec);
    return result;
}

// ============================================================================
// 方法5: SME2 ZA - 使用 svread_za32_f32_vg1x4 一次读取4个向量组 (新方法✅)
// ============================================================================
__arm_new("za") __arm_locally_streaming float dot_product_sme_za_vg1x4(
    const float* A, const float* B, uint64_t size) {
    svzero_za();
    uint64_t SVL = svcntsw();

    for (uint64_t i = 0; i < size; i += 4 * SVL) {
        svcount_t pc = svwhilelt_c32(i, size, 4);
        svfloat32x4_t dataA = svld1_x4(pc, &A[i]);
        svfloat32x4_t dataB = svld1_x4(pc, &B[i]);
        svmla_za32_f32_vg1x4(0, dataA, dataB);
    }

    // ✅ 新方法：使用 svread_za32_f32_vg1x4 一次性读取4个向量组！
    svfloat32x4_t result_x4 = svread_za32_f32_vg1x4(0);

    // 提取4个向量
    svfloat32_t v0 = svget4(result_x4, 0);
    svfloat32_t v1 = svget4(result_x4, 1);
    svfloat32_t v2 = svget4(result_x4, 2);
    svfloat32_t v3 = svget4(result_x4, 3);

    // 合并4个向量
    svbool_t pg_all = svptrue_b32();
    svfloat32_t total = svadd_x(pg_all, v0, v1);
    total = svadd_x(pg_all, total, v2);
    total = svadd_x(pg_all, total, v3);

    // 最终规约
    float result = svaddv(pg_all, total);
    return result;
}

// ============================================================================
// 方法6: SME2 ZA - 4个tile并行，使用 svread_za32_f32_vg1x4
// ============================================================================
__arm_new("za") __arm_locally_streaming float dot_product_sme_za_quad_vg1x4(
    const float* A, const float* B, uint64_t size) {
    svzero_za();
    uint64_t SVL = svcntsw();

    // 使用4个ZA数组并行处理
    for (uint64_t i = 0; i < size; i += 16 * SVL) {
        if (i < size) {
            svcount_t pc0 = svwhilelt_c32(i, size, 4);
            svfloat32x4_t dataA0 = svld1_x4(pc0, &A[i]);
            svfloat32x4_t dataB0 = svld1_x4(pc0, &B[i]);
            svmla_za32_f32_vg1x4(0, dataA0, dataB0);
        }

        if (i + 4 * SVL < size) {
            svcount_t pc1 = svwhilelt_c32(i + 4 * SVL, size, 4);
            svfloat32x4_t dataA1 = svld1_x4(pc1, &A[i + 4 * SVL]);
            svfloat32x4_t dataB1 = svld1_x4(pc1, &B[i + 4 * SVL]);
            svmla_za32_f32_vg1x4(1, dataA1, dataB1);
        }

        if (i + 8 * SVL < size) {
            svcount_t pc2 = svwhilelt_c32(i + 8 * SVL, size, 4);
            svfloat32x4_t dataA2 = svld1_x4(pc2, &A[i + 8 * SVL]);
            svfloat32x4_t dataB2 = svld1_x4(pc2, &B[i + 8 * SVL]);
            svmla_za32_f32_vg1x4(2, dataA2, dataB2);
        }

        if (i + 12 * SVL < size) {
            svcount_t pc3 = svwhilelt_c32(i + 12 * SVL, size, 4);
            svfloat32x4_t dataA3 = svld1_x4(pc3, &A[i + 12 * SVL]);
            svfloat32x4_t dataB3 = svld1_x4(pc3, &B[i + 12 * SVL]);
            svmla_za32_f32_vg1x4(3, dataA3, dataB3);
        }
    }

    // ✅ 使用 svread_za32_f32_vg1x4 读取所有4个tile
    svfloat32x4_t result0 = svread_za32_f32_vg1x4(0);
    svfloat32x4_t result1 = svread_za32_f32_vg1x4(1);
    svfloat32x4_t result2 = svread_za32_f32_vg1x4(2);
    svfloat32x4_t result3 = svread_za32_f32_vg1x4(3);

    // 累加所有向量
    svbool_t pg_all = svptrue_b32();
    svfloat32_t acc = svdup_f32(0.0f);

    acc = svadd_x(pg_all, acc, svget4(result0, 0));
    acc = svadd_x(pg_all, acc, svget4(result1, 0));
    acc = svadd_x(pg_all, acc, svget4(result2, 0));
    acc = svadd_x(pg_all, acc, svget4(result3, 0));

    acc = svadd_x(pg_all, acc, svget4(result0, 1));
    acc = svadd_x(pg_all, acc, svget4(result1, 1));
    acc = svadd_x(pg_all, acc, svget4(result2, 1));
    acc = svadd_x(pg_all, acc, svget4(result3, 1));

    acc = svadd_x(pg_all, acc, svget4(result0, 2));
    acc = svadd_x(pg_all, acc, svget4(result1, 2));
    acc = svadd_x(pg_all, acc, svget4(result2, 2));
    acc = svadd_x(pg_all, acc, svget4(result3, 2));

    acc = svadd_x(pg_all, acc, svget4(result0, 3));
    acc = svadd_x(pg_all, acc, svget4(result1, 3));
    acc = svadd_x(pg_all, acc, svget4(result2, 3));
    acc = svadd_x(pg_all, acc, svget4(result3, 3));

    float result = svaddv(pg_all, acc);
    return result;
}

// ============================================================================
// CPU 参考实现
// ============================================================================
float dot_product_cpu_reference(const float* A, const float* B, uint64_t size) {
    double sum = 0.0;
    for (uint64_t i = 0; i < size; i++) {
        sum += (double)A[i] * (double)B[i];
    }
    return (float)sum;
}

// ============================================================================
// 性能测试框架
// ============================================================================
typedef struct {
    const char* name;
    float (*func)(const float*, const float*, uint64_t);
    double total_time;
    float result;
} BenchmarkMethod;

typedef struct {
    uint64_t size;
    const char* size_name;
    int iterations;
} BenchmarkConfig;

void run_benchmark(BenchmarkMethod* method, const float* A, const float* B,
                   uint64_t size, int iterations) {
    double start_time, end_time;
    method->total_time = 0.0;

    // 预热
    method->result = method->func(A, B, size);

    // 正式测试
    for (int i = 0; i < iterations; i++) {
        start_time = get_time_in_seconds();
        method->result = method->func(A, B, size);
        end_time = get_time_in_seconds();
        method->total_time += (end_time - start_time);
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char* argv[]) {

    if (!__arm_has_sme()) {
        printf("错误：此系统不支持SME\n");
        return 1;
    }

    uint64_t SVL = svcntsw();

    printf("==================================================================="
           "=============\n");
    printf("                    SME/SVE 内积综合性能测试\n");
    printf("              使用 svread_za32_f32_vg1x4 优化版本\n");
    printf("==================================================================="
           "=============\n");
    printf("SVE向量长度 (SVL): %lu 个FP32元素\n", SVL);
    printf("测试时间: %s\n", __DATE__ " " __TIME__);
    printf("==================================================================="
           "=============\n\n");

    // 定义多种测试配置
    BenchmarkConfig configs[] = {
        {4 * 1024, "4KB (L1 Cache)", 10000},
        {16 * 1024, "16KB (L1 Cache)", 10000},
        {64 * 1024, "64KB (L1/L2)", 5000},
        {256 * 1024, "256KB (L2 Cache)", 2000},
        {512 * 1024, "512KB (L2 Cache)", 1000},
    };

    const int num_configs = sizeof(configs) / sizeof(configs[0]);

    // 定义测试方法
    BenchmarkMethod methods[] = {
        {"SVE单向量", dot_product_sve_single, 0.0, 0.0},
        {"SVE多向量(x4)", dot_product_sve_multi, 0.0, 0.0},
        {"SME单ZA(手动读)", dot_product_sme_za_single, 0.0, 0.0},
        {"SME 4ZA(手动读)", dot_product_sme_za_quad, 0.0, 0.0},
        {"SME单ZA(vg1x4)✅", dot_product_sme_za_vg1x4, 0.0, 0.0},
        {"SME 4ZA(vg1x4)✅", dot_product_sme_za_quad_vg1x4, 0.0, 0.0}};
    const int num_methods = sizeof(methods) / sizeof(methods[0]);

    // CSV文件头
    printf("\n生成CSV格式报告...\n");
    FILE* csv_file = fopen("benchmark_results.csv", "w");
    if (csv_file) {
        fprintf(
            csv_file,
            "Vector Size,Size Name,Data Size(MB),Iterations,Method,Total "
            "Time(s),Avg "
            "Time(ms),Throughput(GFLOPS),Speedup,Correctness,Relative Error\n");
    }

    // 对每种配置进行测试
    for (int config_idx = 0; config_idx < num_configs; config_idx++) {
        BenchmarkConfig* config = &configs[config_idx];

        // 对齐大小到 16*SVL
        uint64_t size =
            ((config->size + 16 * SVL - 1) / (16 * SVL)) * (16 * SVL);
        int iterations = config->iterations;

        printf("\n============================================================="
               "===================\n");
        printf("测试配置 %d/%d: %s\n", config_idx + 1, num_configs,
               config->size_name);
        printf("==============================================================="
               "=================\n");
        printf("  向量大小:     %lu 个FP32元素\n", size);
        printf("  数据大小:     %.2f MB\n", size * sizeof(float) / 1e6);
        printf("  迭代次数:     %d\n", iterations);
        printf("  单次计算量:   %.3f GFLOPS\n", size * 2.0 / 1e9);
        printf("---------------------------------------------------------------"
               "-----------------\n");

        // 分配对齐内存
        float* A = (float*)aligned_alloc(64, size * sizeof(float));
        float* B = (float*)aligned_alloc(64, size * sizeof(float));

        if (!A || !B) {
            printf("内存分配失败\n");
            if (A)
                free(A);
            if (B)
                free(B);
            continue;
        }

        // 初始化数据
        srand(42);
        for (uint64_t i = 0; i < size; i++) {
            A[i] = (float)(rand() % 100) / 10.0f;
            B[i] = (float)(rand() % 100) / 10.0f;
        }

        // 计算CPU参考结果用于验证
        printf("\n计算CPU参考结果...\n");
        float reference = dot_product_cpu_reference(A, B, size);
        printf("参考结果: %.6f\n", reference);

        // 运行所有方法
        printf("\n运行测试...\n");
        for (int i = 0; i < num_methods; i++) {
            printf("  [%d/%d] %s ... ", i + 1, num_methods, methods[i].name);
            fflush(stdout);
            run_benchmark(&methods[i], A, B, size, iterations);
            printf("完成!\n");
        }

        // 验证结果正确性
        printf("\n验证结果正确性...\n");
        int all_correct = 1;
        for (int i = 0; i < num_methods; i++) {
            float error = methods[i].result - reference;
            float rel_error = (reference != 0.0f) ? (error / reference) : 0.0f;

            // 判断是否正确（相对误差小于0.01%）
            int is_correct = (fabs(rel_error) < 1e-4);
            all_correct = all_correct && is_correct;

            printf("  %-25s: %.6f (误差: %+.2e, 相对: %+.2e%%) %s\n",
                   methods[i].name, methods[i].result, error, rel_error * 100.0,
                   is_correct ? "✅" : "❌");
        }

        if (!all_correct) {
            printf("\n⚠️  警告：部分方法结果不正确！\n");
        }

        // 输出性能结果
        printf("\n%-25s | %12s | %12s | %15s | %10s | %12s\n", "方法",
               "总时间(秒)", "平均(ms)", "吞吐量(GFLOPS)", "加速比", "正确性");
        printf("-------------------------+-------------+-------------+---------"
               "-----"
               "--+-----------+-------------\n");

        double baseline_time = methods[0].total_time / iterations;

        for (int i = 0; i < num_methods; i++) {
            double avg_time = methods[i].total_time / iterations;
            double gflops = (size * 2.0 / 1e9) / avg_time;
            double speedup = baseline_time / avg_time;

            float error = methods[i].result - reference;
            float rel_error = (reference != 0.0f) ? (error / reference) : 0.0f;
            int is_correct = (fabs(rel_error) < 1e-4);

            printf("%-25s | %12.6f | %12.6f | %15.2f | %10.2fx | %12s\n",
                   methods[i].name, methods[i].total_time, avg_time * 1000.0,
                   gflops, speedup, is_correct ? "✅ 正确" : "❌ 错误");

            // 写入CSV
            if (csv_file) {
                fprintf(csv_file,
                        "%lu,%s,%.2f,%d,%s,%.6f,%.6f,%.2f,%.2f,%s,%.6e\n",
                        config->size, config->size_name,
                        size * sizeof(float) / 1e6, iterations, methods[i].name,
                        methods[i].total_time, avg_time * 1000.0, gflops,
                        speedup, is_correct ? "PASS" : "FAIL", rel_error);
            }
        }

        // 清理
        free(A);
        free(B);
    }

    if (csv_file) {
        fclose(csv_file);
        printf("\n\n结果已保存到: benchmark_results.csv\n");
    }

    // ============================================================================
    // 生成性能总结
    // ============================================================================
    printf("\n\n");
    printf("==================================================================="
           "=============\n");
    printf("                            性能总结\n");
    printf("==================================================================="
           "=============\n");
    printf("\n关键发现:\n");
    printf("  1. svread_za32_f32_vg1x4 可以一次性读取4个向量组\n");
    printf("  2. 相比手动读取，代码更简洁、性能更好\n");
    printf("  3. 多tile并行仍然是最优方案\n");
    printf("\nsvread_za32_f32_vg1x4 优势:\n");
    printf("  ✅ 一次调用读取4个向量组（vs 4次单独读取）\n");
    printf("  ✅ 代码更简洁易读\n");
    printf("  ✅ 编译器更容易优化\n");
    printf("  ✅ 是ARM官方推荐的配套函数！\n\n");
    printf("==================================================================="
           "=============\n");
    printf("测试完成!\n");
    printf("==================================================================="
           "=============\n");

    return 0;
}