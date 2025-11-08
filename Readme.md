# ARM SME vs SVE Dot Product Performance Analysis

Comprehensive performance comparison between ARM Scalable Vector Extension (SVE) and Scalable Matrix Extension 2 (SME2) for vector dot product operations on Apple Silicon M4, demonstrating **up to 14× speedup** with SME2 parallel tile optimization.

![SME vs SVE Performance Analysis](sme_sve_performance_hires.png)

## 🎯 Overview

This project provides an in-depth performance analysis comparing ARM's SVE and SME2 instruction sets for dot product operations on **Apple Mac mini M4 chip**. It implements and benchmarks four optimization strategies across different data sizes and cache hierarchies:

- **SVE Single Vector**: Traditional SVE-based single vector accumulation (baseline)
- **SVE Multi-Vector (×4)**: SVE with 4-way vector groups using `svld1_x4`
- **SME Single ZA Tile**: SME2 acceleration using one ZA accumulator tile
- **SME 4-ZA Tiles Parallel**: Advanced optimization leveraging all 4 ZA tiles simultaneously

## 🖥️ Test Environment

- **Hardware**: Apple Mac mini (2024)
- **Chip**: Apple M4 (ARMv9-A with SME2)
- **Architecture**: arm64
- **Instruction Sets**: ARM SVE2 + SME2
- **Vector Length**: 512-bit (16 × 32-bit floats)
- **Cache Hierarchy**: L1 (192KB), L2 (16MB), L3 (24MB shared)

## 📊 Performance Results

Performance comparison across different data sizes and cache levels:

### Key Performance Metrics (1MB L2/L3 Cache Level)

| Implementation | Time (μs) | Speedup | GFLOPS | ZA Utilization |
|----------------|-----------|---------|--------|----------------|
| SVE Single Vector | 135.4 | 1.00× | 15.5 | N/A |
| SVE Multi-Vector (×4) | 67.8 | 2.00× | 30.9 | N/A |
| SME Single ZA Tile | 17.1 | 7.93× | 122.7 | 25% (1/4) |
| SME 4-ZA Tiles Parallel | 9.7 | **14.00×** | **216.7** | **100% (4/4)** |

### Scaling Performance Across Data Sizes

| Data Size | Cache Level | SVE Single (μs) | SME 4-ZA (μs) | Speedup | Peak GFLOPS |
|-----------|-------------|-----------------|---------------|---------|-------------|
| 4KB | L1 | 1.055 | 0.217 | 4.9× | 37.7 |
| 16KB | L1 | 2.534 | 0.282 | 9.0× | 116.0 |
| 64KB | L1/L2 | 8.520 | 0.726 | 11.7× | 180.5 |
| 256KB | L2 | 33.726 | 2.535 | 13.3× | 206.9 |
| 512KB | L2 | 67.836 | 4.972 | 13.6× | 210.9 |
| 1MB | L2/L3 | 135.445 | 9.677 | 14.0× | 216.7 |

### Key Findings

- 🚀 **Maximum Speedup**: 14.0× over SVE single vector baseline (1MB data size)
- ⚡ **Peak Throughput**: 216.7 GFLOPS on M4 (SME 4-ZA tiles)
- 📈 **SME vs SVE Multi**: 7.0× improvement with full tile utilization
- 🔥 **4-Tiles Parallel Impact**: 1.77× improvement over single ZA tile (at 1MB)
- ✅ **Full Tile Utilization**: 100% (4/4 ZA tiles active in parallel)
- 📊 **Cache Scaling**: Performance scales excellently from L1 to L2/L3

## 🚀 Quick Start

### Prerequisites

- **Apple Silicon Mac** with M4 chip (or ARM processor with SME2 support)
- **macOS Sonoma 14.0+** or newer
- **Xcode Command Line Tools** or **Homebrew GCC 13+**
- **Python 3.8+** with matplotlib and numpy for visualization

### Check SME2 Support

```bash
# Check if your Mac supports SME2
sysctl hw.optional.arm.FEAT_SME2

# Should return: hw.optional.arm.FEAT_SME2: 1
```

### Compilation on macOS

```bash
# Using Apple Clang (recommended for M4)
clang -arch arm64 -march=native+sme2 -O3 -o benchmark sme_comprehensive_benchmark.c -lm

# Or using Homebrew GCC
gcc-13 -march=native+sme2 -O3 -o benchmark sme_comprehensive_benchmark.c -lm

# Run the comprehensive benchmark
./benchmark
```

### Compilation Flags Explained

- `-arch arm64`: Target ARM64 architecture (Apple Silicon)
- `-march=native+sme2`: Enable ARMv9-A architecture with SME2 extensions
- `-O3`: Maximum optimization level
- `-lm`: Link math library for timing functions

## 📈 Visualization

Generate performance comparison charts:

```bash
# Install required Python packages
pip3 install matplotlib numpy

# Generate visualizations
python3 sme_matmul_visualization.py
```

This will create:
- `sme_sve_performance_comparison.png` - Standard resolution (300 DPI)
- `sme_sve_performance_comparison.pdf` - Vector format for papers
- `sme_sve_performance_comparison.svg` - Editable vector graphics
- `sme_sve_performance_hires.png` - High resolution (600 DPI)

## 📁 Project Structure

```
.
├── sme_comprehensive_benchmark.c       # Complete SME vs SVE implementation
├── sme_matmul_visualization.py         # Performance visualization script
├── benchmark_results.csv               # Generated CSV with all results
├── sme_sve_performance_comparison.png  # Performance comparison chart
└── README.md                           # This file
```

## 🔬 Technical Details

### Implementation Strategies

#### 1. **SVE Single Vector** (Baseline)
```c
__arm_locally_streaming float
dot_product_sve_single(const float* A, const float* B, uint64_t size)
```
- Traditional SVE-based implementation
- Single vector accumulator
- Uses `svld1`, `svmla_m`, `svaddv` intrinsics
- Baseline for performance comparison

#### 2. **SVE Multi-Vector (×4)**
```c
__arm_locally_streaming float
dot_product_sve_multi(const float* A, const float* B, uint64_t size)
```
- Utilizes SVE vector groups with `svld1_x4`
- 4-way parallel accumulation without ZA tiles
- Demonstrates SVE's multi-vector capabilities
- 2× speedup over single vector

#### 3. **SME Single ZA Tile**
```c
__arm_new("za") __arm_locally_streaming float
dot_product_sme_za_single(const float* A, const float* B, uint64_t size)
```
- Uses SME2's ZA accumulator array (single tile)
- `svmla_za32_f32_vg1x4` for fused multiply-accumulate
- 7.93× speedup with 25% tile utilization
- Reads from ZA rows 0, 4, 8, 12

#### 4. **SME 4-ZA Tiles Parallel** (Best Performance)
```c
__arm_new("za") __arm_locally_streaming float
dot_product_sme_za_quad(const float* A, const float* B, uint64_t size)
```
- Processes 4 data blocks simultaneously
- Full utilization of all 4 available ZA tiles (ZA[0-3])
- Maximum parallelism and computational efficiency
- 14× speedup with 100% tile utilization

### Key SME2 Intrinsics Used

- `svmla_za32_f32_vg1x4`: Matrix multiply-accumulate with vector groups
- `svld1_x4`: Load 4 vectors simultaneously with count predication
- `svzero_za`: Zero all ZA tiles
- `svread_hor_za32_f32_m`: Read horizontal slices from ZA tiles
- `svwhilelt_c32`: Generate count predicates for vector groups
- `svget4`, `svcreate4`: Manipulate 4-element vector tuples
- `svaddv`: Horizontal vector reduction

## 🔗 Resources

### ARM Documentation
- [ARM SME2 Specification](https://developer.arm.com/documentation/ddi0602/latest/)
- [ARM SVE2 Programming Guide](https://developer.arm.com/documentation/102476/latest/)
- [ARM C Language Extensions (ACLE)](https://github.com/ARM-software/acle)
- [ARM Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)

### Apple Silicon Resources
- [Apple Silicon Developer Guide](https://developer.apple.com/documentation/apple-silicon)
- [Optimizing for Apple Silicon](https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms)

### Performance Analysis
- [ARM Architecture Reference Manual](https://developer.arm.com/documentation/ddi0487/latest/)
- [Optimizing Software for ARM](https://developer.arm.com/documentation/102107/latest/)

## 📝 License

This project is provided as-is for educational and research purposes. Feel free to use, modify, and distribute with attribution.

## 🙏 Acknowledgments

- ARM for the excellent SME2 and SVE2 instruction set architecture
- Apple for powerful M4 chip with full SME2 support
- Open source community for tools and documentation

**Note**: This implementation is specifically optimized for Apple Silicon M4 with SME2 support. Performance results may vary on different ARM processors with SME/SME2 capabilities. Always benchmark on your target hardware for production deployments.