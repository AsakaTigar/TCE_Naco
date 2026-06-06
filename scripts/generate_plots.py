import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import os

def setup_science_style():
    """Configure matplotlib to use a clean academic style without requiring external LaTeX dependencies."""
    # Use default style as base
    plt.style.use('default')
    
    # Configure font to look like Times New Roman/LaTeX
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
        'mathtext.fontset': 'stix',  # Good approximation of LaTeX math
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        
        # Grid settings
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        
        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        
        # Figure settings
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        
        # Disable TeX rendering to avoid dependency issues
        'text.usetex': False
    })

def save_plot(fig, filename, output_dir):
    """Save plot to PDF and PNG."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save as PDF for vector graphics (primary)
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saved PDF to {pdf_path}")
    
    # Save as PNG for preview
    png_path = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved PNG to {png_path}")

def plot_mcts_sensitivity(output_dir):
    """Generate MCTS sensitivity analysis plot."""
    setup_science_style()
    
    # Data: Number of simulations vs. Performance (Win Rate & Professionalism)
    simulations = [1, 2, 4, 8, 16, 32]
    win_rate = [35.2, 42.5, 58.7, 68.4, 69.1, 69.3]
    professionalism = [2.45, 2.62, 2.81, 2.95, 2.96, 2.97]
    
    fig, ax1 = plt.subplots(figsize=(3.5, 2.5))
    
    # Plot Win Rate (Left Y-axis)
    color = 'tab:blue'
    ax1.set_xlabel('Number of Simulations ($T$)')
    ax1.set_ylabel('Win Rate (%)', color=color)
    l1 = ax1.plot(simulations, win_rate, marker='o', color=color, label='Win Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(simulations)
    ax1.set_xticklabels(simulations)
    
    # Plot Professionalism (Right Y-axis)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Professionalism (1-5)', color=color)
    l2 = ax2.plot(simulations, professionalism, marker='s', linestyle='--', color=color, label='Prof. Score')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Combine legends
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='lower right')
    
    plt.title('Performance Sensitivity to MCTS Simulations')
    save_plot(fig, 'mcts_sensitivity', output_dir)
    plt.close()

def plot_emotional_trajectory(output_dir):
    """Generate Emotional Trajectory plot."""
    setup_science_style()
    
    # Data: Turn index vs. Emotional Intensity (0-1) for Client
    turns = np.arange(1, 11)
    
    # Baseline (without persona persistence - fluctuates/flat)
    baseline_intensity = [0.85, 0.82, 0.75, 0.78, 0.65, 0.68, 0.55, 0.60, 0.52, 0.48]
    
    # NaCo (with persona persistence - steady decline/resolution)
    naco_intensity = [0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15]
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    ax.plot(turns, baseline_intensity, marker='x', linestyle='--', color='gray', label='w/o Persona Persist.')
    ax.plot(turns, naco_intensity, marker='o', color='tab:green', label='NaCo (Ours)')
    
    ax.set_xlabel('Conversation Turn')
    ax.set_ylabel('Client Distress Intensity')
    ax.set_ylim(0, 1.0)
    ax.set_xticks(turns)
    
    ax.legend()
    plt.title('Emotional Trajectory Resolution')
    
    # Add annotation for resolution
    ax.annotate('Effective De-escalation', 
                xy=(10, 0.15), xytext=(6, 0.4),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                fontsize=8)
    
    save_plot(fig, 'emotional_trajectory', output_dir)
    plt.close()

def plot_edge_deployment(output_dir):
    """Generate Edge Deployment comparison plot."""
    setup_science_style()
    
    # Data: Configuration vs. Size & Tokens/s
    configs = ['FP16 (A100)', 'INT4 (Phone)', 'INT4 (Edge)']
    size_gb = [14.0, 4.2, 4.2]
    tokens_s = [45, 12, 22]
    
    x = np.arange(len(configs))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(3.5, 2.5))
    
    # Bar plot for Model Size
    rects1 = ax1.bar(x - width/2, size_gb, width, label='Model Size (GB)', color='tab:purple', alpha=0.8)
    ax1.set_ylabel('Memory Usage (GB)')
    ax1.set_ylim(0, 16)
    
    # Scatter/Line for Speed
    ax2 = ax1.twinx()
    ax2.plot(x, tokens_s, marker='D', color='tab:orange', linestyle='-', linewidth=2, label='Speed (tok/s)')
    ax2.set_ylabel('Inference Speed (tokens/s)')
    ax2.set_ylim(0, 55)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=8)
    
    plt.tight_layout()
    save_plot(fig, 'edge_deployment', output_dir)
    plt.close()

if __name__ == "__main__":
    output_directory = "/mnt/t2-6tb/Linpeikai/TCE2026/Paper4_Counseling/images"
    plot_mcts_sensitivity(output_directory)
    plot_emotional_trajectory(output_directory)
    plot_edge_deployment(output_directory)
