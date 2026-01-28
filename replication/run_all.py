"""Run all MOSAIC replication notebooks.

Usage:
    python run_all.py           # Run all layers
    python run_all.py --layer 1 # Run specific layer
    python run_all.py --quick   # Quick import test only
"""

import sys
from pathlib import Path


def run_notebook(notebook_path: Path) -> bool:
    """Execute a notebook and return success status."""
    import nbformat
    from nbclient import NotebookClient

    print(f"\n{'='*60}")
    print(f"Running: {notebook_path.name}")
    print('='*60)

    try:
        nb = nbformat.read(notebook_path, as_version=4)
        client = NotebookClient(nb, timeout=600, kernel_name='python3')
        client.execute()
        nbformat.write(nb, notebook_path)
        print(f"[PASS] {notebook_path.name}")
        return True

    except Exception as e:
        print(f"[FAIL] {notebook_path.name}")
        print(f"  Error: {str(e)[:500]}")
        return False


def quick_test() -> bool:
    """Quick import test without running notebooks."""
    print("Quick import test...")
    try:
        from config import DATA
        from functions import get, store
        print(f"  [PASS] config.py: {len(DATA)} DATA entries")
        print(f"  [PASS] functions.py: get/store imported")
        return True
    except Exception as e:
        print(f"  [FAIL] Import failed: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run MOSAIC replication notebooks')
    parser.add_argument('--layer', type=int, choices=[1, 2, 3, 4], help='Run specific layer only')
    parser.add_argument('--quick', action='store_true', help='Quick import test only')
    args = parser.parse_args()

    replication_dir = Path(__file__).parent

    if args.quick:
        success = quick_test()
        sys.exit(0 if success else 1)

    notebooks = {
        1: replication_dir / 'layer1_inputs.ipynb',
        2: replication_dir / 'layer2_calculations.ipynb',
        3: replication_dir / 'layer3_simulation.ipynb',
        4: replication_dir / 'layer4_channel3.ipynb',
    }

    if args.layer:
        notebooks = {args.layer: notebooks[args.layer]}

    # Run quick test first
    if not quick_test():
        print("\nQuick test failed. Fix imports before running notebooks.")
        sys.exit(1)

    # Run selected notebooks
    results = {}
    for layer, path in sorted(notebooks.items()):
        if path.exists():
            results[layer] = run_notebook(path)
        else:
            print(f"[FAIL] NOT FOUND: {path}")
            results[layer] = False

    # Summary
    print(f"\n{'='*60}")
    print("REPLICATION SUMMARY")
    print('='*60)

    all_passed = True
    for layer, passed in sorted(results.items()):
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  Layer {layer}: {status}")
        if not passed:
            all_passed = False

    print('='*60)
    if all_passed:
        print("All replications PASSED")
        sys.exit(0)
    else:
        print("Some replications FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
