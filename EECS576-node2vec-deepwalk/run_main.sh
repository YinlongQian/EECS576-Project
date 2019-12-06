#!/bin/bash

for num in $(seq 0 50)
do
    echo "${num}"
    python src/main.py --input RM_sparse/Weekly/vc-week-${num}.npz --output emb_node2vec/RM/week/output_week_${num}.emb --p 4 --q 1 
done
