# %%
import os
import json
from pprint import pprint

def open_json_file(file_path)->dict:
    """Open a JSON file and return its content.
    Format:
    all_models_progress = {
        "last_processed_index": -1,
        "total_captions_processed": 0,
        "clip": {
            "recall@": {"1": 0,"5": 0,"10": 0},
            "precision@": {"1": 0.0,"5": 0.0,"10": 0.0}
            },
            ...,
    }}
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            all_models_progress = json.load(f)
        
    return all_models_progress

def calc_metrics(results:dict)->tuple[dict, int]:
    captions = results["total_captions_processed"]
    images = results["last_processed_index"]
    
    for model in ["clip", "clip+paligemma"]:
        for metric_name, metric_dict in results[model].items():
            for k, value in metric_dict.items():
                results[model][metric_name][k] = value / captions
    pprint("Metrics calculated:")
    pprint(results)
    return results, captions, images

#%%
JSON_RESULTS_FILE = "benchmarks/eval_progress_pali_10000.json"
normalized_results, samples, images = calc_metrics(open_json_file(JSON_RESULTS_FILE))                    
# %%
print(f"Metrics for {samples} captions")
print("Normalized Clip Results:")
pprint(normalized_results["clip"])
print("\n")
print("Normalized Clip+Pali Results:")
pprint(normalized_results["clip+paligemma"])
# %%
