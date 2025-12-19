# CPU vs GPU ML Inference Benchmark

A comprehensive Python tool for comparing machine learning inference performance between CPU and GPU (specifically optimized for NVIDIA GTX 750 Ti). This benchmark visualizes the speed advantage of GPU acceleration for neural network inference.

![Performance Comparison](https://img.shields.io/badge/GPU_Speedup-4.9x-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![CUDA](https://img.shields.io/badge/CUDA-11.8-purple)

## 📋 Overview

This project benchmarks and compares the performance of CPU vs GPU for machine learning inference tasks. It's specifically tested and optimized for the NVIDIA GTX 750 Ti (2GB VRAM) but works with any CUDA-compatible GPU.

### Key Features
- ✅ **CPU vs GPU performance comparison** with detailed metrics
- ✅ **Interactive visualization** with matplotlib
- ✅ **Batch size analysis** to find optimal configurations
- ✅ **GTX 750 Ti optimized** for 2GB VRAM constraints
- ✅ **Real-time progress tracking** with tqdm
- ✅ **Professional reporting** with multiple visualization formats
- ✅ **Automatic GPU detection** and configuration


## 🚀 Quick Start

### Prerequisites
- NVIDIA GPU with CUDA support (GTX 750 Ti in my case from my OLD work pc)
- Windows 10/11 or Linux with Python 3.8+
- NVIDIA drivers installed
- At least 8GB system RAM


📊 Understanding the Output
The program generates three visualizations:

inference_comparison.png - Main dashboard with 4 subplots:

Inference time distribution (box plot)

Throughput comparison (bar chart)

Batch size performance (line chart)

Speedup factor (line chart)

batch_size_comparison.png - Detailed analysis of different batch sizes:

Shows how performance scales with batch size

Identifies optimal batch size for your GPU

performance_summary.png - Executive summary report:

Comparison table with key metrics

Device specifications

Conclusions and recommendations

🖥️ Interactive Viewer Controls
When the benchmark completes, an interactive viewer opens with:

text
Image Viewer Controls:
----------------------------------------
• Use LEFT/RIGHT arrow keys or buttons to navigate
• Press 'ESC' or click 'Exit Viewer' to close
• Current image will be shown in full screen
----------------------------------------
Keyboard Shortcuts
Left Arrow / 'p' - Previous image

Right Arrow / 'n' - Next image

ESC / 'q' - Exit viewer

🔍 Technical Details
Neural Network Architecture
The benchmark uses a custom CNN with:

3 convolutional layers (64, 128, 256 filters)

Max pooling after each layer

2 fully connected layers

ReLU activations

Dropout regularization

Total parameters: 8,765,066

Test Configuration
Input size: 64×64 RGB images

Batch sizes tested: [1, 2, 4, 8, 16]

Warmup iterations: 10

Measurement iterations: 50

Precision: FP32 (single precision)

🎯 Optimization Tips for GTX 750 Ti
Memory Management
Maximum batch size: 16 (for 2GB VRAM)

Optimal batch size: 8 (best speed/VRAM balance)

VRAM usage: ~80MB per batch

Performance Tuning
Close other applications to free VRAM

Use smaller models for real-time applications

Batch processing improves GPU utilization

Monitor temperatures during extended runs

Expected Performance
Model Size	CPU FPS	GPU FPS	Speedup	VRAM Usage
Small (8M params)	20-30	100-150	4-6x	~80MB
Medium (50M params)	3-5	20-30	5-8x	~500MB
Large (200M params)	1-2	8-12	6-10x	~1.5GB
