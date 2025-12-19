import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
from matplotlib.widgets import Button
import warnings
warnings.filterwarnings('ignore')

# Fix the deprecation warning
plt.rcParams['figure.max_open_warning'] = 50

class SimpleCNN(nn.Module):
    """A simple CNN for demonstration"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 256 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ImageViewer:
    """Interactive image viewer class"""
    def __init__(self, image_files):
        self.image_files = [f for f in image_files if os.path.exists(f)]
        self.current_index = 0
        self.fig = None
        self.ax = None
        self.img_display = None
        
        if not self.image_files:
            print("No image files found to display.")
            return
            
        self.create_viewer()
    
    def create_viewer(self):
        """Create the interactive viewer window"""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.2)
        
        # Navigation buttons
        ax_prev = plt.axes([0.2, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.35, 0.05, 0.1, 0.075])
        ax_exit = plt.axes([0.55, 0.05, 0.2, 0.075])
        
        btn_prev = Button(ax_prev, 'Previous')
        btn_next = Button(ax_next, 'Next')
        btn_exit = Button(ax_exit, 'Exit Viewer')
        
        btn_prev.on_clicked(self.previous_image)
        btn_next.on_clicked(self.next_image)
        btn_exit.on_clicked(self.exit_viewer)
        
        # Display first image
        self.show_current_image()
        
        # Add keyboard shortcuts
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        print(f"\nImage Viewer Controls:")
        print("-" * 40)
        print("• Use LEFT/RIGHT arrow keys or buttons to navigate")
        print("• Press 'ESC' or click 'Exit Viewer' to close")
        print("• Current image will be shown in full screen")
        print("-" * 40)
        
        plt.show()
    
    def show_current_image(self):
        """Display the current image"""
        if self.current_index >= len(self.image_files):
            return
            
        img_file = self.image_files[self.current_index]
        
        try:
            # Clear previous image
            self.ax.clear()
            
            # Load and display image
            img = plt.imread(img_file)
            self.img_display = self.ax.imshow(img)
            self.ax.set_title(f"{img_file} ({self.current_index + 1}/{len(self.image_files)})", 
                            fontsize=14, fontweight='bold', pad=20)
            self.ax.axis('off')
            
            # Add image info
            info_text = f"Size: {img.shape[1]}x{img.shape[0]} pixels\n"
            info_text += f"Created: {time.ctime(os.path.getctime(img_file))}"
            
            self.fig.text(0.02, 0.02, info_text, fontsize=10, 
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            
            plt.draw()
            
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
    
    def next_image(self, event=None):
        """Show next image"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_current_image()
    
    def previous_image(self, event=None):
        """Show previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
    
    def exit_viewer(self, event=None):
        """Exit the viewer"""
        plt.close('all')
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'right' or event.key == 'n':
            self.next_image()
        elif event.key == 'left' or event.key == 'p':
            self.previous_image()
        elif event.key == 'escape' or event.key == 'q':
            self.exit_viewer()

def check_device_capabilities():
    """Check GPU capabilities"""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Check compute capability
        major = torch.cuda.get_device_capability(0)[0]
        minor = torch.cuda.get_device_capability(0)[1]
        print(f"Compute Capability: {major}.{minor}")
    else:
        print("Warning: CUDA not available. Running on CPU only.")
    
    print("=" * 60)
    return torch.cuda.is_available()

def run_inference(device, model, batch_size=16, num_batches=50, warmup=10):
    """Run inference on specified device"""
    print(f"\nRunning inference on: {device.upper()}")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    input_shape = (batch_size, 3, 64, 64)
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup runs
    print(f"Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Measure performance
    print(f"Measuring performance ({num_batches} iterations)...")
    times = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc=f"Inference on {device}", 
                     bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            torch.cuda.synchronize() if device == 'cuda' else None
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1000 / avg_time if avg_time > 0 else 0
    
    print(f"Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    
    if device == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(0) / 1e6
        memory_cached = torch.cuda.memory_reserved(0) / 1e6
        print(f"GPU Memory allocated: {memory_allocated:.2f} MB")
        print(f"GPU Memory cached: {memory_cached:.2f} MB")
    
    return times, avg_time, fps

def create_plots(cpu_times, gpu_times, avg_cpu, avg_gpu, fps_cpu, fps_gpu, batch_results):
    """Create and save all plots"""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Plot 1: Main comparison plot
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: Inference time distribution
    plt.subplot(2, 2, 1)
    box_data = [cpu_times, gpu_times]
    bp = plt.boxplot(box_data, patch_artist=True, 
                     tick_labels=['CPU', 'GPU'], widths=0.6)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
    plt.title('Inference Time Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add mean values
    for i, (avg, label) in enumerate([(avg_cpu, 'CPU'), (avg_gpu, 'GPU')]):
        plt.text(i + 1, avg + 0.5, f'{avg:.1f} ms', 
                ha='center', va='bottom', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Subplot 2: Throughput comparison
    plt.subplot(2, 2, 2)
    devices = ['CPU', 'GPU']
    throughput = [fps_cpu, fps_gpu]
    bars = plt.bar(devices, throughput, color=['#1f77b4', '#2ca02c'], 
                   alpha=0.8, edgecolor='black', linewidth=2)
    
    plt.ylabel('Throughput (FPS)', fontsize=12, fontweight='bold')
    plt.title('Throughput Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add values on bars
    for bar, value in zip(bars, throughput):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f} FPS', ha='center', va='bottom', 
                fontweight='bold', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Subplot 3: Batch size performance
    if batch_results:
        plt.subplot(2, 2, 3)
        batch_sizes = list(batch_results.keys())
        cpu_batch_times = [batch_results[bs]['cpu'] for bs in batch_sizes]
        gpu_batch_times = [batch_results[bs]['gpu'] for bs in batch_sizes]
        
        plt.plot(batch_sizes, cpu_batch_times, 'o-', label='CPU', 
                linewidth=3, markersize=10, color='#1f77b4')
        plt.plot(batch_sizes, gpu_batch_times, 's-', label='GPU', 
                linewidth=3, markersize=10, color='#2ca02c')
        
        plt.xlabel('Batch Size', fontsize=12, fontweight='bold')
        plt.ylabel('Average Time (ms)', fontsize=12, fontweight='bold')
        plt.title('Batch Size Performance', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
    
    # Subplot 4: Speedup factor
    plt.subplot(2, 2, 4)
    if batch_results:
        speedups = [batch_results[bs]['speedup'] for bs in batch_sizes]
        plt.plot(batch_sizes, speedups, 'D-', linewidth=3, markersize=10, 
                color='#ff7f0e', markerfacecolor='yellow')
        
        plt.xlabel('Batch Size', fontsize=12, fontweight='bold')
        plt.ylabel('Speedup (GPU/CPU)', fontsize=12, fontweight='bold')
        plt.title('GPU Speedup Factor', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add speedup values
        for bs, speedup in zip(batch_sizes, speedups):
            plt.text(bs, speedup + 0.1, f'{speedup:.1f}x', 
                    ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(f'ML Inference Performance: CPU vs GPU (GTX 750 Ti)\n', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('inference_comparison.png', dpi=120, bbox_inches='tight')
    print("✓ Saved: inference_comparison.png")
    
    # Plot 2: Detailed batch size comparison
    if batch_results:
        plt.figure(figsize=(12, 8))
        
        batch_sizes = list(batch_results.keys())
        x = np.arange(len(batch_sizes))
        width = 0.35
        
        # Prepare data
        cpu_vals = [batch_results[bs]['cpu'] for bs in batch_sizes]
        gpu_vals = [batch_results[bs]['gpu'] for bs in batch_sizes]
        speedups = [batch_results[bs]['speedup'] for bs in batch_sizes]
        
        # Create subplots
        plt.subplot(2, 1, 1)
        bars1 = plt.bar(x - width/2, cpu_vals, width, label='CPU', 
                       color='#1f77b4', alpha=0.8, edgecolor='black')
        bars2 = plt.bar(x + width/2, gpu_vals, width, label='GPU', 
                       color='#2ca02c', alpha=0.8, edgecolor='black')
        
        plt.xlabel('Batch Size', fontsize=12, fontweight='bold')
        plt.ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
        plt.title('Inference Time by Batch Size', fontsize=14, fontweight='bold')
        plt.xticks(x, batch_sizes)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add values on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                        f'{height:.1f}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=9)
        
        plt.subplot(2, 1, 2)
        bars3 = plt.bar(batch_sizes, speedups, color='#ff7f0e', 
                       alpha=0.8, edgecolor='black')
        
        plt.xlabel('Batch Size', fontsize=12, fontweight='bold')
        plt.ylabel('Speedup Factor', fontsize=12, fontweight='bold')
        plt.title('GPU Speedup by Batch Size', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Add speedup values
        for bar, speedup in zip(bars3, speedups):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{speedup:.1f}x', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        
        plt.suptitle('Detailed Batch Size Analysis\n', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('batch_size_comparison.png', dpi=120, bbox_inches='tight')
        print("✓ Saved: batch_size_comparison.png")
    
    # Plot 3: Performance summary
    plt.figure(figsize=(10, 8))
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        device_name = "CPU Only"
        memory_gb = 0
    
    # Create a summary table
    summary_data = [
        ["Metric", "CPU", "GPU", "Speedup"],
        ["Avg Time (ms)", f"{avg_cpu:.2f}", f"{avg_gpu:.2f}", f"{avg_cpu/avg_gpu:.2f}x"],
        ["Throughput (FPS)", f"{fps_cpu:.2f}", f"{fps_gpu:.2f}", f"{fps_gpu/fps_cpu:.2f}x"],
        ["Device", "Intel/AMD CPU", device_name, ""],
        ["Memory", "System RAM", f"{memory_gb:.1f} GB VRAM", ""],
        ["Batch Size", "8", "8", "Same"]
    ]
    
    # Create table
    ax = plt.gca()
    ax.axis('off')
    
    # Create table
    table = plt.table(cellText=summary_data,
                     cellLoc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25],
                     loc='center',
                     bbox=[0.1, 0.3, 0.8, 0.6])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Color the header row
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternating rows
    for i in range(1, len(summary_data)):
        for j in range(len(summary_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")
    
    plt.title('Performance Summary Report\n', fontsize=16, fontweight='bold', pad=20)
    
    # Add conclusion text
    conclusion = "CONCLUSION: GPU provides significant speedup for ML inference\n"
    conclusion += f"GTX 750 Ti achieves {avg_cpu/avg_gpu:.1f}x faster inference than CPU"
    
    plt.figtext(0.5, 0.1, conclusion, ha='center', fontsize=12, 
               fontweight='bold', bbox=dict(boxstyle="round,pad=1", 
                                           facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('performance_summary.png', dpi=120, bbox_inches='tight')
    print("✓ Saved: performance_summary.png")

def run_batch_comparison():
    """Compare performance with different batch sizes"""
    print("\n" + "=" * 60)
    print("BATCH SIZE COMPARISON")
    print("=" * 60)
    
    batch_sizes = [1, 2, 4, 8, 16]
    batch_results = {}
    
    for batch_size in batch_sizes:
        print(f"\n{'='*40}")
        print(f"Testing Batch Size: {batch_size}")
        print('='*40)
        
        # CPU
        model_cpu = SimpleCNN()
        _, avg_cpu, fps_cpu = run_inference('cpu', model_cpu, batch_size, 
                                           num_batches=20, warmup=5)
        
        # GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            model_gpu = SimpleCNN()
            _, avg_gpu, fps_gpu = run_inference('cuda', model_gpu, batch_size, 
                                               num_batches=20, warmup=5)
            
            speedup = avg_cpu / avg_gpu
            print(f"\n🚀 Speedup (GPU vs CPU): {speedup:.2f}x")
            
            batch_results[batch_size] = {
                'cpu': avg_cpu,
                'gpu': avg_gpu,
                'speedup': speedup
            }
            
            del model_gpu
        
        del model_cpu
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return batch_results

def main():
    """Main function to run CPU vs GPU comparison"""
    print("\n" + "=" * 60)
    print("ML INFERENCE BENCHMARK - CPU vs GPU")
    print("=" * 60)
    
    # Check system
    has_gpu = check_device_capabilities()
    
    # Create model
    print("\nCreating neural network model...")
    model = SimpleCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params:,} parameters")
    
    # Test batch size (conservative for GTX 750 Ti)
    batch_size = 8
    
    # Run CPU inference
    print("\n" + "=" * 60)
    print("PHASE 1: CPU INFERENCE")
    print("=" * 60)
    cpu_times, avg_cpu, fps_cpu = run_inference('cpu', model, batch_size)
    
    # Run GPU inference if available
    if has_gpu:
        print("\n" + "=" * 60)
        print("PHASE 2: GPU INFERENCE")
        print("=" * 60)
        torch.cuda.empty_cache()
        gpu_times, avg_gpu, fps_gpu = run_inference('cuda', model, batch_size)
        
        # Calculate speedup
        speedup = avg_cpu / avg_gpu
        
        # Display results
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<20} {'CPU':<15} {'GPU':<15} {'Speedup':<10}")
        print("-" * 60)
        print(f"{'Avg Time (ms)':<20} {avg_cpu:<15.2f} {avg_gpu:<15.2f} {speedup:<10.2f}x")
        print(f"{'Throughput (FPS)':<20} {fps_cpu:<15.2f} {fps_gpu:<15.2f} {fps_gpu/fps_cpu:<10.2f}x")
        print("=" * 60)
        
        # Run batch comparison
        batch_results = run_batch_comparison()
        
        # Create visualizations
        create_plots(cpu_times, gpu_times, avg_cpu, avg_gpu, fps_cpu, fps_gpu, batch_results)
        
        # Display images
        image_files = [
            'inference_comparison.png',
            'batch_size_comparison.png',
            'performance_summary.png'
        ]
        
        print("\n" + "=" * 60)
        print("LAUNCHING IMAGE VIEWER")
        print("=" * 60)
        
        # Open interactive viewer
        viewer = ImageViewer(image_files)
        
    else:
        print("\n⚠️  Skipping GPU tests (CUDA not available)")
        print("\nTo enable GPU acceleration:")
        print("1. Install NVIDIA drivers")
        print("2. Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    for f in ['inference_comparison.png', 'batch_size_comparison.png', 'performance_summary.png']:
        if os.path.exists(f):
            size_kb = os.path.getsize(f) / 1024
            print(f"  • {f} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)