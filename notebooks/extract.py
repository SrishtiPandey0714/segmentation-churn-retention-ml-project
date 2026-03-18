import json
import sys
try:
    with open('c:/ds projects/churn-segmentation-ml/notebooks/01_data_understanding_eda.ipynb', 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open('c:/ds projects/churn-segmentation-ml/notebooks/extract.txt', 'w', encoding='utf-8') as out:
        out.write(f"Total cells: {len(data['cells'])}\n")
        # Print the last 20 cells (source only)
        for i, cell in enumerate(data['cells'][-20:]):
            out.write(f"\n--- CELL {len(data['cells']) - 20 + i} ({cell['cell_type']}) ---\n")
            out.write("".join(cell.get('source', [])))
            out.write("\n")
except Exception as e:
    with open('c:/ds projects/churn-segmentation-ml/notebooks/extract.txt', 'w', encoding='utf-8') as out:
        out.write(f"Error: {e}\n")
