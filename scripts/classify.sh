#!/bin/bash

mkdir -p anis2_results anis4_results anis6_results anis8_results

for file in slurm-*.out; do
    last_line=$(tail -n 1 "$file")
    if [[ "$last_line" == *anis2* ]]; then
        cp "$file" anis2_results/
    elif [[ "$last_line" == *anis4* ]]; then
        cp "$file" anis4_results/
    elif [[ "$last_line" == *anis6* ]]; then
        cp "$file" anis6_results/
    elif [[ "$last_line" == *anis8* ]]; then
        cp "$file" anis8_results/
    fi
done
