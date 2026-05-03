# Router for FeDeLiS

This repository contains the data preparation script and training data files for the query-strategy router used in the FeDeLiS project.

## Files

The two source data files are:

- `RoG-webqsp_train_router_labels.jsonl`
- `RoG-cwq_new.jsonl`

These are the only files that need to be updated manually when new router-label data or query data is added.

## Updating the Data

To update the router training data:

1. Edit or replace the two source files:

   ```text
   RoG-webqsp_train_router_labels.jsonl
   RoG-cwq_new.jsonl

2. Run the data collection script:

   ```text
   python3 collect_router_training_data.py

3. The script will automatically generate two output files:

   ```text
   clean_router_training_data.jsonl
   correct_router_lines.jsonl