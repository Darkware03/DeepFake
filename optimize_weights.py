import json
import math

with open('baseline_audit.json', 'r') as f:
    data = json.load(f)

# Load raw variables
# variables['IA']['var_name'] -> list of values
def get_vals(t, var):
    # we need row by row. But baseline_audit.json only has aggregated values!
    # Ah, it doesn't store row by row. Let me modify baseline_audit to output raw arrays.
    pass

