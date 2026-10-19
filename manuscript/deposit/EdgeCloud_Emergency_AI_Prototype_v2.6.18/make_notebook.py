from pathlib import Path
import json

root = Path(__file__).resolve().parent
cells = [
    {"cell_type":"markdown","metadata":{},"source":["# EdgeCloud Emergency AI Controlled Prototype\n","This notebook runs the core synthetic benchmark: five seeds, every method, every condition, the ablations and the sweeps. The outputs test software and method behaviour; they are not field-validation results.\n","\n","**It is a walkthrough of the core benchmark, not a reproduction of the paper.** The adversarial sweeps, the link-realism and decentralisation experiments, the compute-cost measurement, the real-data comparison on CrisisMMD and its split-robustness rotations each have their own runner, listed in README.md, and the manuscript's tables and figures are built from all of them by the scripts at the package root. Running this notebook alone reproduces Sections 6.1 to 6.7 and no more."]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["from pathlib import Path\n","import json\n","import pandas as pd\n","from core import CONFIG, run_all\n"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["summary, confidence_intervals, ablation, sweep = run_all(Path('results'))\n","print('Benchmark complete')\n"]},
    {"cell_type":"markdown","metadata":{},"source":["## Main results"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["main = summary.groupby(['method','scenario'])[['macro_f1','errors','mean_bytes','p95_latency_s','review_rate','conformal_coverage']].mean().reset_index()\n","main.round(4)"]},
    {"cell_type":"markdown","metadata":{},"source":["## Low-bandwidth comparison"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["main[main.scenario.isin(['low_bandwidth','delayed_imagery','misaligned_evidence'])].sort_values(['scenario','errors']).round(4)"]},
    {"cell_type":"markdown","metadata":{},"source":["## Ablation results"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["ablation.round(4)"]},
    {"cell_type":"markdown","metadata":{},"source":["## Delay-penalty sweep"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["sweep.round(4)"]},
    {"cell_type":"markdown","metadata":{},"source":["## Paired differences and decision states"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["paired = pd.read_csv(Path('results') / 'paired_differences.csv')\n","records = pd.read_csv(Path('results') / 'record_level_results.csv.gz')\n","display(paired[paired.scenario.eq('low_bandwidth')].round(4))\n","records[records.method.eq('proposed')].groupby(['scenario','decision_status']).size().unstack(fill_value=0)"]},
    {"cell_type":"markdown","metadata":{},"source":["## Fixed configuration"]},
    {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["CONFIG"]},
]
nb = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
(root/"EdgeCloud_Emergency_AI_Prototype.ipynb").write_text(json.dumps(nb,indent=2),encoding="utf-8")
print(root/"EdgeCloud_Emergency_AI_Prototype.ipynb")
