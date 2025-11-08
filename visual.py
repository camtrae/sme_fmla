#!/usr/bin/env python3
"""
SME vs SVE Performance Visualization Script
ARM Scalable Matrix Extension Performance Analysis
Author: Performance Analysis Team
Date: 2024
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

def setup_plot_style():
    """Set up the plot style and color scheme"""
    # Set scientific paper style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 学术配色方案
    colors = {
        'sve_single': '#045DB7',      # SVE单向量 - 学术蓝
        'sve_multi': '#7FA8FF',       # SVE多向量 - 浅蓝
        'sme_single': '#6A178B',      # SME单ZA - 深紫
        'sme_multi': '#B580C5',       # SME 4ZA - 浅紫
        'grid': '#E8E8E8',            # 浅灰网格
        'text': '#333333',            # 深灰文字
        'baseline': '#7F7F7F',        # 基准线
    }
    
    # Configure fonts
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 14,              
        'axes.labelsize': 15,          
        'axes.titlesize': 16,          
        'xtick.labelsize': 12,         
        'ytick.labelsize': 12,         
        'legend.fontsize': 12,         
        'figure.titlesize': 18,        
        'axes.linewidth': 1.5,         
        'axes.edgecolor': colors['text'],
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    return colors

def load_data():
    """Load SME vs SVE performance data"""
    # 数据大小（用于X轴）
    data_sizes = ['4KB\n(L1)', '16KB\n(L1)', '64KB\n(L1/L2)', 
                  '256KB\n(L2)', '512KB\n(L2)', '1MB\n(L2/L3)']
    
    # 加速比数据（相对于SVE单向量）
    speedup_data = {
        'sve_single': [1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        'sve_multi': [1.98, 2.15, 1.98, 2.00, 2.01, 2.00],
        'sme_single': [4.53, 7.05, 7.34, 7.75, 8.00, 7.93],
        'sme_multi': [4.86, 8.97, 11.73, 13.31, 13.64, 14.00],
    }
    
    # 吞吐量数据（GFLOPS）
    throughput_data = {
        'sve_single': [7.76, 12.93, 15.38, 15.55, 15.46, 15.48],
        'sve_multi': [15.35, 27.76, 30.42, 31.10, 31.00, 30.94],
        'sme_single': [35.19, 91.12, 112.88, 120.54, 123.71, 122.73],
        'sme_multi': [37.70, 116.03, 180.49, 206.86, 210.90, 216.72],
    }
    
    # 执行时间数据（微秒）
    time_data = {
        'sve_single': [1.055, 2.534, 8.520, 33.726, 67.836, 135.445],
        'sve_multi': [0.534, 1.181, 4.309, 16.860, 33.823, 67.777],
        'sme_single': [0.233, 0.360, 1.161, 4.349, 8.476, 17.088],
        'sme_multi': [0.217, 0.282, 0.726, 2.535, 4.972, 9.677],
    }
    
    return data_sizes, speedup_data, throughput_data, time_data

def create_speedup_plot(ax, data_sizes, speedup_data, colors):
    """Create the speedup comparison line chart"""
    x = np.arange(len(data_sizes))
    
    # Plot lines with markers
    methods = {
        'sve_single': ('SVE Single Vector', 'o', colors['sve_single']),
        'sve_multi': ('SVE Multi-Vector (×4)', 's', colors['sve_multi']),
        'sme_single': ('SME Single ZA Tile', '^', colors['sme_single']),
        'sme_multi': ('SME 4-ZA Tiles Parallel', 'D', colors['sme_multi']),
    }
    
    for key, (label, marker, color) in methods.items():
        ax.plot(x, speedup_data[key], 
               marker=marker, color=color, 
               linewidth=3.0, markersize=10, markeredgewidth=2,
               markeredgecolor='white', label=label, 
               alpha=0.9, zorder=3)
    
    # Add value labels for SME 4-ZA (最优性能)
    for i, val in enumerate(speedup_data['sme_multi']):
        if i % 2 == 0:  # 只在偶数位置添加标签，避免拥挤
            ax.text(x[i], val * 1.15, f'{val:.1f}×', 
                   ha='center', va='bottom',
                   fontsize=11, color=colors['sme_multi'], 
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.8, edgecolor=colors['sme_multi'], linewidth=0.5))
    
    # Baseline reference line
    ax.axhline(y=1.0, color=colors['baseline'], 
              linestyle='--', linewidth=2.0, alpha=0.5, 
              label='Baseline (SVE Single)', zorder=1)
    
    # Configure axes
    ax.set_ylabel('Speedup Factor (×)', fontweight='bold', color=colors['text'])
    ax.set_ylim([0.5, 18])
    ax.set_xlim([-0.3, len(data_sizes) - 0.7])
    ax.set_xticks(x)
    ax.set_xticklabels(data_sizes, fontsize=12)
    ax.set_xlabel('Data Size (Cache Level)', fontweight='bold', color=colors['text'])
    ax.set_title('Speedup Factor Analysis', fontweight='bold', pad=15, color=colors['text'])
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=11,
             fancybox=False, edgecolor=colors['grid'], ncol=1)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0, color=colors['grid'])

def create_throughput_plot(ax, data_sizes, throughput_data, colors):
    """Create the throughput comparison bar chart"""
    x = np.arange(len(data_sizes))
    bar_width = 0.2
    
    # Create grouped bars
    methods = [
        ('sve_single', 'SVE Single', colors['sve_single'], -1.5),
        ('sve_multi', 'SVE Multi (×4)', colors['sve_multi'], -0.5),
        ('sme_single', 'SME Single ZA', colors['sme_single'], 0.5),
        ('sme_multi', 'SME 4-ZA', colors['sme_multi'], 1.5),
    ]
    
    for key, label, color, offset in methods:
        values = throughput_data[key]
        bars = ax.bar(x + offset * bar_width, values, bar_width,
                     label=label, color=color, alpha=0.9, 
                     edgecolor='white', linewidth=1.5)
        
        # Add value labels on top of bars for SME methods
        if 'sme' in key:
            for i, (bar, val) in enumerate(zip(bars, values)):
                if i % 2 == 1:  # 只在奇数位置添加标签，避免拥挤
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                           f'{val:.0f}',
                           ha='center', va='bottom', fontsize=10,
                           color=color, fontweight='bold')
    
    # Configure axes
    ax.set_ylabel('Throughput (GFLOPS)', fontweight='bold', color=colors['text'])
    ax.set_ylim([0, 240])
    ax.set_xticks(x)
    ax.set_xticklabels(data_sizes, fontsize=12)
    ax.set_xlabel('Data Size (Cache Level)', fontweight='bold', color=colors['text'])
    ax.set_title('Throughput Performance Comparison', fontweight='bold', pad=15, color=colors['text'])
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=11,
             fancybox=False, edgecolor=colors['grid'], ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y', color=colors['grid'])

def create_execution_time_plot(ax, data_sizes, time_data, colors):
    """Create the execution time comparison plot"""
    x = np.arange(len(data_sizes))
    
    # Plot lines with markers
    methods = {
        'sve_single': ('SVE Single Vector', 'o', colors['sve_single']),
        'sve_multi': ('SVE Multi-Vector (×4)', 's', colors['sve_multi']),
        'sme_single': ('SME Single ZA Tile', '^', colors['sme_single']),
        'sme_multi': ('SME 4-ZA Tiles Parallel', 'D', colors['sme_multi']),
    }
    
    for key, (label, marker, color) in methods.items():
        ax.plot(x, time_data[key], 
               marker=marker, color=color, 
               linewidth=3.0, markersize=10, markeredgewidth=2,
               markeredgecolor='white', label=label, 
               alpha=0.9, zorder=3)
    
    # Add value labels for best and worst cases
    # SME 4-ZA (best)
    best_vals = time_data['sme_multi']
    for i in [0, -1]:  # 首尾两点
        ax.text(x[i], best_vals[i] * 0.7, f'{best_vals[i]:.2f}μs', 
               ha='center', va='top',
               fontsize=10, color=colors['sme_multi'], 
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        alpha=0.8, edgecolor=colors['sme_multi'], linewidth=0.5))
    
    # Configure axes
    ax.set_ylabel('Execution Time (μs)', fontweight='bold', color=colors['text'])
    ax.set_yscale('log')
    ax.set_ylim([0.15, 200])
    ax.set_xlim([-0.3, len(data_sizes) - 0.7])
    ax.set_xticks(x)
    ax.set_xticklabels(data_sizes, fontsize=12)
    ax.set_xlabel('Data Size (Cache Level)', fontweight='bold', color=colors['text'])
    ax.set_title('Execution Time Comparison (Log Scale)', fontweight='bold', pad=15, color=colors['text'])
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=11,
             fancybox=False, edgecolor=colors['grid'])
    ax.grid(True, alpha=0.3, linestyle='--', which='both', color=colors['grid'])

def main():
    """Main function to create and save the visualization"""
    print("Starting SME vs SVE performance visualization generation...")
    print("-" * 60)
    
    # Setup
    colors = setup_plot_style()
    data_sizes, speedup_data, throughput_data, time_data = load_data()
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Create subplots
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    
    # Generate plots
    create_speedup_plot(ax1, data_sizes, speedup_data, colors)
    create_throughput_plot(ax2, data_sizes, throughput_data, colors)
    create_execution_time_plot(ax3, data_sizes, time_data, colors)
    
    # Overall title
    fig.suptitle('ARM SME vs SVE Performance Analysis', 
                fontsize=20, fontweight='bold', y=1.00, color=colors['text'])
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.12)
    
    # Save in multiple formats
    output_formats = {
        'png': {'dpi': 300, 'desc': 'GitHub/Web display'},
        'pdf': {'dpi': None, 'desc': 'LaTeX/Papers'},
        'svg': {'dpi': None, 'desc': 'Vector graphics'},
    }
    
    print("\nSaving files...")
    print("-" * 60)
    
    for fmt, settings in output_formats.items():
        filename = f'sme_sve_performance_comparison.{fmt}'
        plt.savefig(filename, dpi=settings['dpi'], 
                   bbox_inches='tight', facecolor='white')
        print(f'✓ Saved {filename} - {settings["desc"]}')
    
    # High-resolution version
    plt.savefig('sme_sve_performance_hires.png', dpi=600, 
               bbox_inches='tight', facecolor='white')
    print('✓ Saved sme_sve_performance_hires.png - Publication quality (600 DPI)')
    
    # Display summary
    print("\n" + "=" * 60)
    print("📊 SME vs SVE Performance Visualization Complete!")
    print("=" * 60)
    print("Key Findings:")
    print(f"  • Maximum Speedup: {max(speedup_data['sme_multi']):.1f}× (SME 4-ZA vs SVE Single)")
    print(f"  • Peak Throughput: {max(throughput_data['sme_multi']):.1f} GFLOPS (SME 4-ZA)")
    print(f"  • Best Execution Time: {min(time_data['sme_multi']):.3f} μs (SME 4-ZA)")
    print(f"  • SME Advantage over SVE Multi: {speedup_data['sme_multi'][-1] / speedup_data['sve_multi'][-1]:.1f}×")
    print("=" * 60)
    
    # Show plot
    # plt.show()

if __name__ == "__main__":
    main()