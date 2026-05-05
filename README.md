# Router for FiDeLiS

This directory contains the offline router-training assets for the FiDeLiS project.

## Layout

```text
Router_for_FeDeLiS/
├── BERT/
│   ├── webqsp_cwq/
│   │   ├── 01_distilbert_frozen_strategy_classifier.ipynb
│   │   ├── 02_two_head_frozen_distilbert_no_weights.ipynb
│   │   ├── 03_two_head_unfreeze_last1_no_oversampling.ipynb
│   │   ├── 04_two_head_unfreeze_last1_mild_weights_no_oversampling.ipynb
│   │   ├── 05_two_head_finetune_all_mild_weights_oversampling.ipynb
│   │   └── 06_action_ranker_pointwise_pairwise.ipynb
│   └── crlt/
│       └── 01_crlt_support_action_ranker.ipynb
├── preprocessed/
│   ├── webqsp_cwq/
│   │   ├── router_query_table.jsonl
│   │   ├── router_action_table.jsonl
│   │   └── router_pairwise_table.jsonl
│   └── crlt/
│       ├── router_query_table.jsonl
│       ├── router_action_table.jsonl
│       └── router_pairwise_table.jsonl
├── raw_data/
│   ├── webqsp_cwq/
│   │   ├── RoG-webqsp_train_router_labels.jsonl
│   │   ├── RoG-cwq_new.jsonl
│   │   ├── clean_router_training_data.jsonl
│   │   └── correct_router_lines.jsonl
│   └── crlt/
│       ├── CL-LT-KGQA_train_router_labels.jsonl
│       ├── CR-LT-QA.json
│       └── CR-LT-ClaimVerification.json
└── scripts/
    ├── webqsp_cwq/
    │   ├── collect_router_training_data.py
    │   ├── preprocess_router_training_data.py
    │   └── train_bert_action_ranker.py
    └── crlt/
        ├── preprocess_crlt_support_training_data.py
        └── train_bert_crlt_support_ranker.py
```

## Dataset groups

- `webqsp_cwq/`: shared pipeline for WebQSP and CWQ.
- `crlt/`: CR-LT support-alignment pipeline.

The separation is intentional because CR-LT uses a different supervision target.

## Main entry points

### WebQSP + CWQ

- Preprocess:
  - `python Router_for_FeDeLiS/scripts/webqsp_cwq/preprocess_router_training_data.py`
- Train:
  - `python Router_for_FeDeLiS/scripts/webqsp_cwq/train_bert_action_ranker.py`
- Notebook:
  - `Router_for_FeDeLiS/BERT/webqsp_cwq/06_action_ranker_pointwise_pairwise.ipynb`

### CR-LT

- Preprocess:
  - `python Router_for_FeDeLiS/scripts/crlt/preprocess_crlt_support_training_data.py`
- Train:
  - `python Router_for_FeDeLiS/scripts/crlt/train_bert_crlt_support_ranker.py`
- Notebook:
  - `Router_for_FeDeLiS/BERT/crlt/01_crlt_support_action_ranker.ipynb`

## Notes

- The preprocessing scripts write only to `preprocessed/...`. They do not modify the online FiDeLiS generation pipeline.
- The notebooks are configured to work with this directory layout both locally and in Colab Drive mode.
- Output artifacts are written under `Router_for_FeDeLiS/outputs/...` when training runs are executed.
