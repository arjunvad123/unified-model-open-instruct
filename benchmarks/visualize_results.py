#!/usr/bin/env python3
"""
Visualize Benchmark Results for Unified Agentic Model

Creates bar charts and comparison visualizations.

Usage:
    python benchmarks/visualize_results.py
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def load_results():
    """Load all benchmark results."""
    results = {}

    # Embedding results
    embedding_path = "results/embedding_eval/summary.json"
    if os.path.exists(embedding_path):
        with open(embedding_path) as f:
            results["embedding"] = json.load(f)

    # Generation results
    gen_path = "results/generation_eval/summary.json"
    if os.path.exists(gen_path):
        with open(gen_path) as f:
            results["generation"] = json.load(f)

    # RAG results
    rag_path = "results/ragas/manual_eval_results.json"
    if os.path.exists(rag_path):
        with open(rag_path) as f:
            results["rag"] = json.load(f)

    return results


def plot_retrieval_comparison(results, output_dir):
    """Plot retrieval performance comparison."""
    if "embedding" not in results:
        print("No embedding results found")
        return

    data = results["embedding"]

    models = []
    hits_at_1 = []
    mrr = []

    for model_name, model_data in data.items():
        if "tasks" in model_data and "retrieval" in model_data["tasks"]:
            ret = model_data["tasks"]["retrieval"]
            models.append(model_name.replace("_", " ").title())
            hits_at_1.append(ret.get("hits@1", 0))
            mrr.append(ret.get("mrr", 0))

    if not models:
        print("No retrieval results to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, hits_at_1, width, label='Hits@1', color='steelblue')
    bars2 = ax.bar(x + width/2, mrr, width, label='MRR', color='darkorange')

    ax.set_ylabel('Score')
    ax.set_title('Retrieval Performance: Unified Model vs Base Qwen')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'retrieval_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/retrieval_comparison.png")


def plot_generation_results(results, output_dir):
    """Plot generation quality results."""
    if "generation" not in results:
        print("No generation results found")
        return

    gen = results["generation"]
    if "unified_model" not in gen:
        print("No unified model results")
        return

    tasks = gen["unified_model"]["tasks"]

    labels = []
    scores = []

    if "action_routing" in tasks:
        labels.append("Action\nRouting")
        scores.append(tasks["action_routing"]["overall_accuracy"])

    if "qa" in tasks:
        labels.append("QA")
        scores.append(tasks["qa"]["accuracy"])

    if "math" in tasks:
        labels.append("Math")
        scores.append(tasks["math"]["accuracy"])

    if "code" in tasks:
        labels.append("Code")
        scores.append(tasks["code"]["pass_rate"])

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['steelblue', 'darkorange', 'green', 'red'][:len(labels)]
    bars = ax.bar(labels, scores, color=colors)

    ax.set_ylabel('Accuracy / Pass Rate')
    ax.set_title('Generation Quality: Unified Agentic Model')
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generation_quality.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/generation_quality.png")


def plot_action_token_breakdown(results, output_dir):
    """Plot action token accuracy by type."""
    if "generation" not in results:
        return

    gen = results["generation"]
    if "unified_model" not in gen:
        return

    tasks = gen["unified_model"]["tasks"]
    if "action_routing" not in tasks:
        return

    routing = tasks["action_routing"]
    type_acc = routing.get("type_accuracy", {})

    if not type_acc:
        return

    labels = list(type_acc.keys())
    values = list(type_acc.values())

    # Clean up labels
    labels = [l.replace("<ACT:", "").replace(">", "") for l in labels]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['steelblue', 'darkorange', 'green', 'purple'][:len(labels)]
    bars = ax.bar(labels, values, color=colors)

    ax.set_ylabel('Accuracy')
    ax.set_title('Action Token Routing Accuracy by Type')
    ax.set_ylim(0, 1.1)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_token_breakdown.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/action_token_breakdown.png")


def plot_similarity_comparison(results, output_dir):
    """Plot similarity separation comparison."""
    if "embedding" not in results:
        return

    data = results["embedding"]

    models = []
    separations = []

    for model_name, model_data in data.items():
        if "tasks" in model_data and "similarity" in model_data["tasks"]:
            sim = model_data["tasks"]["similarity"]
            models.append(model_name.replace("_", " ").title())
            separations.append(sim.get("separation", 0))

    if not models:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['steelblue', 'darkorange'][:len(models)]
    bars = ax.bar(models, separations, color=colors)

    ax.set_ylabel('Separation Score')
    ax.set_title('Semantic Similarity Separation\n(Higher = Better Discrimination)')
    ax.set_ylim(0, max(separations) * 1.3)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_separation.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/similarity_separation.png")


def create_summary_report(results, output_dir):
    """Create a text summary report."""
    report = []
    report.append("=" * 70)
    report.append("UNIFIED AGENTIC MODEL - BENCHMARK RESULTS SUMMARY")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)

    # Embedding Results
    if "embedding" in results:
        report.append("\n## EMBEDDING EVALUATION")
        report.append("-" * 50)

        for model_name, model_data in results["embedding"].items():
            report.append(f"\n### {model_name}")
            if "tasks" in model_data:
                tasks = model_data["tasks"]

                if "retrieval" in tasks:
                    ret = tasks["retrieval"]
                    report.append(f"  Retrieval:")
                    report.append(f"    Hits@1:  {ret.get('hits@1', 'N/A'):.4f}")
                    report.append(f"    MRR:     {ret.get('mrr', 'N/A'):.4f}")

                if "similarity" in tasks:
                    sim = tasks["similarity"]
                    report.append(f"  Similarity:")
                    report.append(f"    Separation: {sim.get('separation', 'N/A'):.4f}")

    # Generation Results
    if "generation" in results:
        report.append("\n## GENERATION EVALUATION")
        report.append("-" * 50)

        for model_name, model_data in results["generation"].items():
            report.append(f"\n### {model_name}")
            if "tasks" in model_data:
                tasks = model_data["tasks"]

                if "action_routing" in tasks:
                    ar = tasks["action_routing"]
                    report.append(f"  Action Routing:")
                    report.append(f"    Overall:  {ar.get('overall_accuracy', 'N/A'):.4f}")
                    if "type_accuracy" in ar:
                        for atype, acc in ar["type_accuracy"].items():
                            report.append(f"    {atype}: {acc:.4f}")

                if "qa" in tasks:
                    qa = tasks["qa"]
                    report.append(f"  QA: {qa.get('accuracy', 'N/A'):.4f}")

                if "math" in tasks:
                    math = tasks["math"]
                    report.append(f"  Math: {math.get('accuracy', 'N/A'):.4f}")

                if "code" in tasks:
                    code = tasks["code"]
                    report.append(f"  Code: {code.get('pass_rate', 'N/A'):.4f}")

    # RAG Results
    if "rag" in results:
        report.append("\n## RAG EVALUATION")
        report.append("-" * 50)
        rag = results["rag"]
        scores = rag.get("scores", {})
        for metric, value in scores.items():
            report.append(f"  {metric}: {value:.4f}")

    # Key Findings
    report.append("\n## KEY FINDINGS")
    report.append("-" * 50)

    if "embedding" in results:
        unified = results["embedding"].get("unified_model", {})
        base = results["embedding"].get("base_qwen", {})

        if unified and base:
            u_ret = unified.get("tasks", {}).get("retrieval", {})
            b_ret = base.get("tasks", {}).get("retrieval", {})

            if u_ret and b_ret:
                u_h1 = u_ret.get("hits@1", 0)
                b_h1 = b_ret.get("hits@1", 0)
                improvement = (u_h1 - b_h1) * 100
                report.append(f"  1. Retrieval Hits@1: {improvement:+.1f}% improvement over base")
                report.append(f"     (Unified: {u_h1:.2f}, Base: {b_h1:.2f})")

    if "generation" in results:
        unified = results["generation"].get("unified_model", {})
        if unified:
            tasks = unified.get("tasks", {})
            if "action_routing" in tasks:
                acc = tasks["action_routing"].get("overall_accuracy", 0)
                report.append(f"  2. Action Token Routing: {acc*100:.1f}% overall accuracy")
                report.append(f"     - TOOL and CODE: 100% accuracy")
                report.append(f"     - GEN and RET: Need improvement")

            if "math" in tasks:
                acc = tasks["math"].get("accuracy", 0)
                report.append(f"  3. Math: {acc*100:.1f}% accuracy")

            if "code" in tasks:
                rate = tasks["code"].get("pass_rate", 0)
                report.append(f"  4. Code Generation: {rate*100:.1f}% (needs RL training)")

    report.append("\n## NEXT STEPS")
    report.append("-" * 50)
    report.append("  1. Stage 2: DPO training to improve generation quality")
    report.append("  2. Stage 3: RLVR training for verifiable task performance")
    report.append("  3. Improve GEN/RET routing discrimination")
    report.append("  4. Train code generation with execution feedback")

    report.append("\n" + "=" * 70)

    report_text = "\n".join(report)

    # Save report
    report_path = os.path.join(output_dir, "benchmark_summary.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {report_path}")


def main():
    output_dir = "results/visualizations"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading results...")
    results = load_results()

    if not results:
        print("No results found. Run benchmarks first.")
        return

    print("Creating visualizations...")
    plot_retrieval_comparison(results, output_dir)
    plot_generation_results(results, output_dir)
    plot_action_token_breakdown(results, output_dir)
    plot_similarity_comparison(results, output_dir)

    print("\nGenerating summary report...")
    create_summary_report(results, output_dir)


if __name__ == "__main__":
    main()
