#!/bin/bash
# A bash script to run python evaluate_mt.py with different lang pairs

# Define an array of lang pairs
lang_pairs=("afr_Latn-nya_Latn" "eng_Latn-nya_Latn" "fra_Latn-nya_Latn" "eng_Latn-kas_Arab" "eng_Latn-kas_Deva" "hin_Deva-kas_Arab" "hin_Deva-kas_Deva" "fra_Latn-run_Latn" "eng_Latn-run_Latn")

# Loop through the array and run the python script with each lang pair
for lang_pair in "${lang_pairs[@]}"
do
  echo "Running python evaluate_mt.py with lang pair: $lang_pair"
  python evaluate_mt.py --lang_pair "$lang_pair" --do_finetune 1 --nickname "$lang_pair" --short_run 1 --base_model_name google/mt5-base --batch_size 8
done
