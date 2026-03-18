import json
import sys

try:
    with open('c:/ds projects/churn-segmentation-ml/notebooks/01_data_understanding_eda.ipynb', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    with open('c:/ds projects/churn-segmentation-ml/notebooks/extracted_code.py', 'w', encoding='utf-8') as out:
        out.write("# Extracted notebook code\n\n")
        for i, cell in enumerate(data.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source = "".join(cell.get('source', []))
                if source.strip():
                    out.write(f"\n# --- CELL {i} ---\n")
                    out.write(source)
                    out.write("\n")
                    
except Exception as e:
    print(f"Error: {e}")
