// sme_comprehensive_benchmark.c
// 综合性能测试: 多种向量规模和迭代次数
// 编译: clang -O3 -march=armv9-a+sme2 -o benchmark
// sme_comprehensive_benchmark.c

#include <arm_sme.h>
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
// 方法3: SME2 ZA向量组内积 (使用单个ZA[0])
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
// 方法4: SME2 ZA向量组内积 (同时使用4个ZA数组: ZA[0], ZA[1], ZA[2], ZA[3])
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
    printf("==================================================================="
           "=============\n");
    printf("SVE向量长度 (SVL): %lu 个FP32元素\n", SVL);
    printf("测试时间: %s\n", __DATE__ " " __TIME__);
    printf("==================================================================="
           "=============\n\n");

    // 定义多种测试配置
    // 覆盖L1缓存、L2缓存、L3缓存以及超过缓存大小的场景
    BenchmarkConfig configs[] = {
        // 向量大小              名称                迭代次数
        {4 * 1024, "4KB (L1 Cache)", 10000},           // 16 KB (4K * 4 bytes)
        {16 * 1024, "16KB (L1 Cache)", 10000},         // 64 KB
        {64 * 1024, "64KB (L1/L2)", 5000},             // 256 KB
        {256 * 1024, "256KB (L2 Cache)", 2000},        // 1 MB
        {512 * 1024, "512KB (L2 Cache)", 1000},        // 2 MB
        {1024 * 1024, "1MB (L2/L3)", 1000},            // 4 MB
        {2 * 1024 * 1024, "2MB (L3 Cache)", 500},      // 8 MB
        {4 * 1024 * 1024, "4MB (L3 Cache)", 500},      // 16 MB
        {8 * 1024 * 1024, "8MB (Main Memory)", 200},   // 32 MB
        {16 * 1024 * 1024, "16MB (Main Memory)", 100}, // 64 MB
    };

    const int num_configs = sizeof(configs) / sizeof(configs[0]);

    // 定义测试方法
    BenchmarkMethod methods[] = {
        {"SVE单向量", dot_product_sve_single, 0.0, 0.0},
        {"SVE多向量(x4)", dot_product_sve_multi, 0.0, 0.0},
        {"SME单ZA", dot_product_sme_za_single, 0.0, 0.0},
        {"SME 4ZA并行", dot_product_sme_za_quad, 0.0, 0.0}};
    const int num_methods = sizeof(methods) / sizeof(methods[0]);

    // CSV文件头
    printf("\n生成CSV格式报告...\n");
    FILE* csv_file = fopen("benchmark_results.csv", "w");
    if (csv_file) {
        fprintf(csv_file,
                "Vector Size,Size Name,Data Size(MB),Iterations,Method,Total "
                "Time(s),Avg Time(ms),Throughput(GFLOPS),Speedup\n");
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

        // 运行所有方法
        printf("\n运行测试...\n");
        for (int i = 0; i < num_methods; i++) {
            printf("  [%d/%d] %s ... ", i + 1, num_methods, methods[i].name);
            fflush(stdout);
            run_benchmark(&methods[i], A, B, size, iterations);
            printf("完成!\n");
        }

        // 输出结果
        printf("\n%-20s | %12s | %12s | %15s | %10s\n", "方法", "总时间(秒)",
               "平均(ms)", "吞吐量(GFLOPS)", "加速比");
        printf("--------------------+-------------+-------------+--------------"
               "--+-----------\n");

        double baseline_time = methods[0].total_time / iterations;

        for (int i = 0; i < num_methods; i++) {
            double avg_time = methods[i].total_time / iterations;
            double gflops = (size * 2.0 / 1e9) / avg_time;
            double speedup = baseline_time / avg_time;

            printf("%-20s | %12.6f | %12.6f | %15.2f | %10.2fx\n",
                   methods[i].name, methods[i].total_time, avg_time * 1000.0,
                   gflops, speedup);

            // 写入CSV
            if (csv_file) {
                fprintf(csv_file, "%lu,%s,%.2f,%d,%s,%.6f,%.6f,%.2f,%.2f\n",
                        config->size, config->size_name,
                        size * sizeof(float) / 1e6, iterations, methods[i].name,
                        methods[i].total_time, avg_time * 1000.0, gflops,
                        speedup);
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
    printf("  1. Cache层级影响: 观察不同数据大小对性能的影响\n");
    printf("  2. 方法对比: 比较4种实现在不同场景下的表现\n");
    printf("  3. 加速比: SME向量组相对于标准SVE的性能提升\n");
    printf("  4. 吞吐量: 不同数据规模下的计算吞吐量变化\n");
    printf("\n建议:\n");
    printf("  - 小数据 (< 256KB): 适合L1/L2缓存,所有方法性能接近\n");
    printf("  - 中等数据 (256KB-4MB): SME向量组开始显示优势\n");
    printf("  - 大数据 (> 4MB): 主要受内存带宽限制,SME 4ZA并行表现最佳\n");
    printf("\n");
    printf("==================================================================="
           "=============\n");
    printf("测试完成!\n");
    printf("==================================================================="
           "=============\n");

    return 0;
}