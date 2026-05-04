#!/usr/bin/env python3
"""
Test script to verify data loading for Unified Agentic Model training.
This can be run without a GPU to ensure datasets load correctly.
"""

import sys
sys.path.insert(0, '.')

from datasets import load_dataset
from transformers import AutoTokenizer
from rich.console import Console
from rich.table import Table
import time

from open_instruct.action_tokens import ACTION_TOKENS

console = Console()


def test_dataset_loading():
    """Test loading all datasets used in training."""

    console.print("\n[bold blue]=" * 60)
    console.print("[bold blue]Testing Unified Agentic Model Data Loading")
    console.print("[bold blue]=" * 60)

    results = []

    # Test 1: TOUCAN (Embedding) - Agent-Ark/Toucan-1.5M
    console.print("\n[yellow]Testing TOUCAN dataset (Agent-Ark/Toucan-1.5M, config=SFT)...[/yellow]")
    start = time.time()
    try:
        toucan = load_dataset("Agent-Ark/Toucan-1.5M", "SFT", split="train", streaming=True)
        sample = next(iter(toucan.take(5)))
        elapsed = time.time() - start
        results.append(("TOUCAN", "✓", f"{elapsed:.2f}s", str(list(sample.keys()))))
        console.print(f"  [green]✓ TOUCAN loaded successfully[/green]")
        console.print(f"  Sample keys: {list(sample.keys())}")
    except Exception as e:
        results.append(("TOUCAN", "✗", "-", str(e)[:50]))
        console.print(f"  [red]✗ Failed: {e}[/red]")

    # Test 2: MS MARCO (Embedding)
    console.print("\n[yellow]Testing MS MARCO dataset...[/yellow]")
    start = time.time()
    try:
        msmarco = load_dataset("ms_marco", "v2.1", split="train", streaming=True)
        sample = next(iter(msmarco.take(5)))
        elapsed = time.time() - start
        results.append(("MS MARCO", "✓", f"{elapsed:.2f}s", str(list(sample.keys()))))
        console.print(f"  [green]✓ MS MARCO loaded successfully[/green]")
        console.print(f"  Sample keys: {list(sample.keys())}")
    except Exception as e:
        results.append(("MS MARCO", "✗", "-", str(e)[:50]))
        console.print(f"  [red]✗ Failed: {e}[/red]")

    # Test 3: Tulu 3 (Generation)
    console.print("\n[yellow]Testing Tulu 3 dataset...[/yellow]")
    start = time.time()
    try:
        tulu = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
        sample = next(iter(tulu.take(5)))
        elapsed = time.time() - start
        results.append(("Tulu 3", "✓", f"{elapsed:.2f}s", str(list(sample.keys()))))
        console.print(f"  [green]✓ Tulu 3 loaded successfully[/green]")
        console.print(f"  Sample keys: {list(sample.keys())}")
    except Exception as e:
        results.append(("Tulu 3", "✗", "-", str(e)[:50]))
        console.print(f"  [red]✗ Failed: {e}[/red]")

    # Test 4: RAGBench (Generation)
    console.print("\n[yellow]Testing RAGBench dataset...[/yellow]")
    start = time.time()
    try:
        ragbench = load_dataset("rungalileo/ragbench", "hotpotqa", split="test", streaming=True)
        sample = next(iter(ragbench.take(5)))
        elapsed = time.time() - start
        results.append(("RAGBench", "✓", f"{elapsed:.2f}s", str(list(sample.keys()))))
        console.print(f"  [green]✓ RAGBench loaded successfully[/green]")
        console.print(f"  Sample keys: {list(sample.keys())}")
    except Exception as e:
        results.append(("RAGBench", "✗", "-", str(e)[:50]))
        console.print(f"  [red]✗ Failed: {e}[/red]")

    # Test 5: HotpotQA (Generation)
    console.print("\n[yellow]Testing HotpotQA dataset...[/yellow]")
    start = time.time()
    try:
        hotpot = load_dataset("hotpot_qa", "fullwiki", split="train", streaming=True)
        sample = next(iter(hotpot.take(5)))
        elapsed = time.time() - start
        results.append(("HotpotQA", "✓", f"{elapsed:.2f}s", str(list(sample.keys()))))
        console.print(f"  [green]✓ HotpotQA loaded successfully[/green]")
        console.print(f"  Sample keys: {list(sample.keys())}")
    except Exception as e:
        results.append(("HotpotQA", "✗", "-", str(e)[:50]))
        console.print(f"  [red]✗ Failed: {e}[/red]")

    # Summary table
    console.print("\n[bold blue]=" * 60)
    console.print("[bold blue]Summary")
    console.print("[bold blue]=" * 60)

    table = Table(title="Dataset Loading Results")
    table.add_column("Dataset", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Time", style="yellow")
    table.add_column("Keys/Error", style="white")

    for name, status, time_str, info in results:
        style = "green" if status == "✓" else "red"
        table.add_row(name, f"[{style}]{status}[/{style}]", time_str, info[:40])

    console.print(table)

    passed = sum(1 for r in results if r[1] == "✓")
    console.print(f"\n[bold]Passed: {passed}/{len(results)} datasets[/bold]")

    return passed == len(results)


def test_tokenizer():
    """Test tokenizer loading and action token addition."""
    console.print("\n[bold blue]=" * 60)
    console.print("[bold blue]Testing Tokenizer")
    console.print("[bold blue]=" * 60)

    console.print("\n[yellow]Loading Qwen2.5-0.5B tokenizer...[/yellow]")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
        console.print(f"  [green]✓ Tokenizer loaded[/green]")
        console.print(f"  Vocab size: {len(tokenizer)}")

        # Add action tokens
        console.print("\n[yellow]Adding action tokens...[/yellow]")
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": ACTION_TOKENS})
        console.print(f"  [green]✓ Added {num_added} tokens[/green]")
        console.print(f"  New vocab size: {len(tokenizer)}")

        # Test tokenization
        console.print("\n[yellow]Testing action token encoding...[/yellow]")
        for token in ACTION_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            decoded = tokenizer.decode([token_id])
            console.print(f"  {token}: ID={token_id}, decoded='{decoded}'")

        # Test full sequence
        test_text = "User: What is AI?\n\n<ACT:THINK> Let me think about this.\n\n<ACT:GEN> AI is...\n\n<ACT:STOP>"
        encoded = tokenizer(test_text, return_tensors="pt")
        decoded = tokenizer.decode(encoded["input_ids"][0])
        console.print(f"\n[yellow]Test sequence encoding:[/yellow]")
        console.print(f"  Input tokens: {encoded['input_ids'].shape[1]}")
        console.print(f"  Decoded matches: {decoded == test_text}")

        return True
    except Exception as e:
        console.print(f"  [red]✗ Failed: {e}[/red]")
        return False


def main():
    console.print("[bold magenta]Unified Agentic Model - Data Loading Test[/bold magenta]\n")

    # Test datasets
    datasets_ok = test_dataset_loading()

    # Test tokenizer
    tokenizer_ok = test_tokenizer()

    # Final summary
    console.print("\n[bold blue]=" * 60)
    console.print("[bold blue]Final Results")
    console.print("[bold blue]=" * 60)

    if datasets_ok and tokenizer_ok:
        console.print("\n[bold green]✓ All tests passed! Ready for training.[/bold green]")
        return 0
    else:
        console.print("\n[bold red]✗ Some tests failed. Please check errors above.[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
