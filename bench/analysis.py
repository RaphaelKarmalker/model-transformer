# bench/analysis.py
"""
Benchmark Analysis & Visualization Script
Erstellt professionelle Plots für Uni-Präsentationen
"""

import json
import os
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Styling für Präsentationen
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})

# Farben für verschiedene Modelle (schöne Farbpalette)
MODEL_COLORS = {
    'openai/gpt-oss-20b': '#2E86AB',           # Blau
    'Qwen/Qwen2.5-7B-Instruct': '#A23B72',     # Magenta
    'mistralai/Mistral-7B-Instruct-v0.3': '#F18F01',  # Orange
    'microsoft/Phi-3-mini-4k-instruct': '#C73E1D',    # Rot
}

# Kurze Namen für die Plots
MODEL_SHORT_NAMES = {
    'openai/gpt-oss-20b': 'GPT-OSS-20B',
    'Qwen/Qwen2.5-7B-Instruct': 'Qwen2.5-7B',
    'mistralai/Mistral-7B-Instruct-v0.3': 'Mistral-7B',
    'microsoft/Phi-3-mini-4k-instruct': 'Phi-3-Mini',
}

# Paths
BENCH_DIR = Path(__file__).parent
RESULTS_DIR = BENCH_DIR / "results"
PLOTS_DIR = BENCH_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_latest_results() -> dict:
    """Lädt die neuesten Ergebnisse für jedes Modell."""
    results = {}
    
    # Gruppiere Dateien nach Modell
    model_files = defaultdict(list)
    for f in RESULTS_DIR.glob("*.json"):
        if f.name == ".gitkeep":
            continue
        model_files[f.name.rsplit('_', 2)[0]].append(f)
    
    # Wähle jeweils die neueste Datei
    for model_prefix, files in model_files.items():
        latest = max(files, key=lambda x: x.stat().st_mtime)
        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results[data['model_id']] = data
    
    return results


def plot_accuracy_comparison(results: dict):
    """Bar Chart: Accuracy Vergleich zwischen Modellen."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    short_names = [MODEL_SHORT_NAMES.get(m, m.split('/')[-1]) for m in models]
    accuracies = [results[m]['accuracy'] * 100 for m in models]
    colors = [MODEL_COLORS.get(m, '#888888') for m in models]
    
    bars = ax.bar(short_names, accuracies, color=colors, edgecolor='white', linewidth=2)
    
    # Werte über den Balken
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy Comparison')
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect Score')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'accuracy_comparison.png')
    plt.savefig(PLOTS_DIR / 'accuracy_comparison.pdf')
    plt.close()
    print(f"[PLOT] Saved: accuracy_comparison.png/pdf")


def plot_inference_time_comparison(results: dict):
    """Bar Chart: Durchschnittliche Inference Zeit."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    short_names = [MODEL_SHORT_NAMES.get(m, m.split('/')[-1]) for m in models]
    times = [results[m]['average_inference_time'] for m in models]
    colors = [MODEL_COLORS.get(m, '#888888') for m in models]
    
    bars = ax.bar(short_names, times, color=colors, edgecolor='white', linewidth=2)
    
    # Werte über den Balken
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.annotate(f'{t:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Average Inference Time (seconds)')
    ax.set_title('Average Inference Time per Test Case')
    ax.set_ylim(0, max(times) * 1.2)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'inference_time_comparison.png')
    plt.savefig(PLOTS_DIR / 'inference_time_comparison.pdf')
    plt.close()
    print(f"[PLOT] Saved: inference_time_comparison.png/pdf")


def plot_accuracy_vs_speed(results: dict):
    """Scatter Plot: Accuracy vs Speed Trade-off."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for model_id, data in results.items():
        accuracy = data['accuracy'] * 100
        avg_time = data['average_inference_time']
        color = MODEL_COLORS.get(model_id, '#888888')
        short_name = MODEL_SHORT_NAMES.get(model_id, model_id.split('/')[-1])
        
        ax.scatter(avg_time, accuracy, s=300, c=color, edgecolors='white', 
                   linewidth=2, label=short_name, zorder=5)
        
        # Label neben dem Punkt
        ax.annotate(short_name,
                    xy=(avg_time, accuracy),
                    xytext=(10, 0),
                    textcoords="offset points",
                    fontsize=10, fontweight='bold',
                    va='center')
    
    ax.set_xlabel('Average Inference Time (seconds)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs. Speed Trade-off')
    ax.set_ylim(70, 105)
    
    # Idealer Bereich markieren
    ax.axhspan(95, 105, alpha=0.1, color='green', label='High Accuracy Zone')
    ax.axvspan(0, 2, alpha=0.1, color='blue', label='Fast Inference Zone')
    
    ax.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'accuracy_vs_speed.png')
    plt.savefig(PLOTS_DIR / 'accuracy_vs_speed.pdf')
    plt.close()
    print(f"[PLOT] Saved: accuracy_vs_speed.png/pdf")


def plot_per_test_results(results: dict):
    """Heatmap: Ergebnisse pro Test Case."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(results.keys())
    short_names = [MODEL_SHORT_NAMES.get(m, m.split('/')[-1]) for m in models]
    
    # Sammle Test-Ergebnisse
    test_ids = [r['test_id'] for r in results[models[0]]['results']]
    test_descriptions = [r['description'][:25] + '...' if len(r['description']) > 25 
                         else r['description'] for r in results[models[0]]['results']]
    
    # Erstelle Matrix: 1 = passed, 0 = failed
    matrix = []
    for model_id in models:
        row = [1 if r['passed'] else 0 for r in results[model_id]['results']]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Custom colormap: Rot für Fehler, Grün für Erfolg
    cmap = plt.cm.colors.ListedColormap(['#E74C3C', '#2ECC71'])
    
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    
    # Achsen
    ax.set_xticks(range(len(test_ids)))
    ax.set_xticklabels([f'T{i}' for i in test_ids], rotation=45, ha='right')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(short_names)
    
    # Grid
    ax.set_xticks(np.arange(-.5, len(test_ids), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    
    ax.set_xlabel('Test Case')
    ax.set_ylabel('Model')
    ax.set_title('Test Results per Model (Green=Pass, Red=Fail)')
    
    # Legende
    legend_elements = [
        mpatches.Patch(facecolor='#2ECC71', edgecolor='white', label='Passed'),
        mpatches.Patch(facecolor='#E74C3C', edgecolor='white', label='Failed')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'per_test_results.png')
    plt.savefig(PLOTS_DIR / 'per_test_results.pdf')
    plt.close()
    print(f"[PLOT] Saved: per_test_results.png/pdf")


def plot_inference_time_boxplot(results: dict):
    """Box Plot: Verteilung der Inference Zeiten."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    short_names = [MODEL_SHORT_NAMES.get(m, m.split('/')[-1]) for m in models]
    
    # Sammle alle Inference-Zeiten pro Modell
    all_times = []
    for model_id in models:
        times = [r['total_inference_time'] for r in results[model_id]['results']]
        all_times.append(times)
    
    colors = [MODEL_COLORS.get(m, '#888888') for m in models]
    
    bp = ax.boxplot(all_times, labels=short_names, patch_artist=True)
    
    # Farben für die Boxen
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    ax.set_ylabel('Inference Time (seconds)')
    ax.set_title('Inference Time Distribution per Model')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'inference_time_boxplot.png')
    plt.savefig(PLOTS_DIR / 'inference_time_boxplot.pdf')
    plt.close()
    print(f"[PLOT] Saved: inference_time_boxplot.png/pdf")


def plot_model_load_time(results: dict):
    """Bar Chart: Model Load Time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    short_names = [MODEL_SHORT_NAMES.get(m, m.split('/')[-1]) for m in models]
    load_times = [results[m]['model_load_time'] for m in models]
    colors = [MODEL_COLORS.get(m, '#888888') for m in models]
    
    bars = ax.bar(short_names, load_times, color=colors, edgecolor='white', linewidth=2)
    
    for bar, t in zip(bars, load_times):
        height = bar.get_height()
        ax.annotate(f'{t:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Load Time (seconds)')
    ax.set_title('Model Load Time Comparison')
    ax.set_ylim(0, max(load_times) * 1.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'model_load_time.png')
    plt.savefig(PLOTS_DIR / 'model_load_time.pdf')
    plt.close()
    print(f"[PLOT] Saved: model_load_time.png/pdf")


def plot_combined_metrics(results: dict):
    """Combined Dashboard: Alle wichtigen Metriken."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    models = list(results.keys())
    short_names = [MODEL_SHORT_NAMES.get(m, m.split('/')[-1]) for m in models]
    colors = [MODEL_COLORS.get(m, '#888888') for m in models]
    
    # 1. Accuracy
    ax1 = axes[0, 0]
    accuracies = [results[m]['accuracy'] * 100 for m in models]
    bars = ax1.bar(short_names, accuracies, color=colors, edgecolor='white', linewidth=1.5)
    for bar, acc in zip(bars, accuracies):
        ax1.annotate(f'{acc:.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy')
    ax1.set_ylim(0, 110)
    ax1.tick_params(axis='x', rotation=15)
    
    # 2. Avg Inference Time
    ax2 = axes[0, 1]
    times = [results[m]['average_inference_time'] for m in models]
    bars = ax2.bar(short_names, times, color=colors, edgecolor='white', linewidth=1.5)
    for bar, t in zip(bars, times):
        ax2.annotate(f'{t:.2f}s', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Avg. Inference Time')
    ax2.tick_params(axis='x', rotation=15)
    
    # 3. Passed/Failed Tests
    ax3 = axes[1, 0]
    passed = [results[m]['passed'] for m in models]
    failed = [results[m]['failed'] for m in models]
    x = np.arange(len(models))
    width = 0.35
    bars1 = ax3.bar(x - width/2, passed, width, label='Passed', color='#2ECC71', edgecolor='white')
    bars2 = ax3.bar(x + width/2, failed, width, label='Failed', color='#E74C3C', edgecolor='white')
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_names, rotation=15)
    ax3.set_ylabel('Number of Tests')
    ax3.set_title('Passed vs Failed Tests')
    ax3.legend()
    
    # 4. Load Time
    ax4 = axes[1, 1]
    load_times = [results[m]['model_load_time'] for m in models]
    bars = ax4.bar(short_names, load_times, color=colors, edgecolor='white', linewidth=1.5)
    for bar, t in zip(bars, load_times):
        ax4.annotate(f'{t:.1f}s', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Model Load Time')
    ax4.tick_params(axis='x', rotation=15)
    
    fig.suptitle('Model Benchmark Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'dashboard.png')
    plt.savefig(PLOTS_DIR / 'dashboard.pdf')
    plt.close()
    print(f"[PLOT] Saved: dashboard.png/pdf")


def plot_test_category_performance(results: dict):
    """Grouped Bar Chart: Performance nach Test-Kategorie."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Kategorien basierend auf Test-Descriptions
    categories = {
        'Valid Transforms': [1, 2, 3, 4],
        'Invalid Actions': [5, 6, 7],
        'Unknown Objects': [8, 9],
        'Ambiguous': [10]
    }
    
    models = list(results.keys())
    short_names = [MODEL_SHORT_NAMES.get(m, m.split('/')[-1]) for m in models]
    x = np.arange(len(categories))
    width = 0.2
    
    for i, (model_id, short_name) in enumerate(zip(models, short_names)):
        category_scores = []
        for cat_name, test_ids in categories.items():
            passed = sum(1 for r in results[model_id]['results'] 
                        if r['test_id'] in test_ids and r['passed'])
            total = len(test_ids)
            category_scores.append(passed / total * 100)
        
        color = MODEL_COLORS.get(model_id, '#888888')
        bars = ax.bar(x + i * width, category_scores, width, label=short_name, 
                      color=color, edgecolor='white', linewidth=1)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Performance by Test Category')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(categories.keys())
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'category_performance.png')
    plt.savefig(PLOTS_DIR / 'category_performance.pdf')
    plt.close()
    print(f"[PLOT] Saved: category_performance.png/pdf")


def generate_summary_table(results: dict):
    """Erstellt eine Zusammenfassungs-Tabelle als Text und LaTeX."""
    
    # Text-Tabelle
    lines = []
    lines.append("=" * 80)
    lines.append("BENCHMARK SUMMARY")
    lines.append("=" * 80)
    lines.append(f"{'Model':<30} {'Accuracy':>10} {'Avg Time':>12} {'Load Time':>12}")
    lines.append("-" * 80)
    
    for model_id, data in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
        short_name = MODEL_SHORT_NAMES.get(model_id, model_id.split('/')[-1])
        acc = f"{data['accuracy']*100:.1f}%"
        avg_time = f"{data['average_inference_time']:.3f}s"
        load_time = f"{data['model_load_time']:.2f}s"
        lines.append(f"{short_name:<30} {acc:>10} {avg_time:>12} {load_time:>12}")
    
    lines.append("=" * 80)
    
    summary_text = "\n".join(lines)
    print(summary_text)
    
    # Speichern
    with open(PLOTS_DIR / 'summary.txt', 'w') as f:
        f.write(summary_text)
    
    # LaTeX Tabelle
    latex_lines = []
    latex_lines.append(r"\begin{table}[h]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Model Benchmark Results}")
    latex_lines.append(r"\begin{tabular}{lccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"\textbf{Model} & \textbf{Accuracy} & \textbf{Avg. Inference} & \textbf{Load Time} \\")
    latex_lines.append(r"\midrule")
    
    for model_id, data in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
        short_name = MODEL_SHORT_NAMES.get(model_id, model_id.split('/')[-1])
        acc = f"{data['accuracy']*100:.1f}\\%"
        avg_time = f"{data['average_inference_time']:.3f}s"
        load_time = f"{data['model_load_time']:.2f}s"
        latex_lines.append(f"{short_name} & {acc} & {avg_time} & {load_time} \\\\")
    
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\label{tab:benchmark}")
    latex_lines.append(r"\end{table}")
    
    latex_text = "\n".join(latex_lines)
    with open(PLOTS_DIR / 'summary_table.tex', 'w') as f:
        f.write(latex_text)
    
    print(f"[PLOT] Saved: summary.txt, summary_table.tex")


def main():
    """Führt die komplette Analyse durch."""
    print("\n" + "=" * 60)
    print("BENCHMARK ANALYSIS")
    print("=" * 60 + "\n")
    
    # Lade Ergebnisse
    results = load_latest_results()
    print(f"[INFO] Loaded results for {len(results)} models:")
    for m in results:
        print(f"  - {m}")
    
    print("\n[INFO] Generating plots...\n")
    
    # Alle Plots generieren
    plot_accuracy_comparison(results)
    plot_inference_time_comparison(results)
    plot_accuracy_vs_speed(results)
    plot_per_test_results(results)
    plot_inference_time_boxplot(results)
    plot_model_load_time(results)
    plot_combined_metrics(results)
    plot_test_category_performance(results)
    
    # Summary
    print()
    generate_summary_table(results)
    
    print(f"\n[DONE] All plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
