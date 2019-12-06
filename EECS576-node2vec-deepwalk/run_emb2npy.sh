#!/bin/bash
for num in $(seq 0 73)
do
    echo "${num}"
    python emb2npy.py ${num}
done
