#!/bin/bash

# Define the arrays of pi-multipliers
theta_vals=(-1 0.5 0 0.5 1)
curv_x_vals=(-0.5 -0.25 0 0.25 0.5)
curv_y_vals=(-0.5 -0.25 0 0.25 0.5)
grad_x_vals=(-1 -0.5 0 0.5 1)
grad_y_vals=(-1 -0.5 0 0.5 1)

echo "Starting parameter sweep..."

for t in "${theta_vals[@]}"; do
    for cx in "${curv_x_vals[@]}"; do
        for cy in "${curv_y_vals[@]}"; do
            for gx in "${grad_x_vals[@]}"; do
                for gy in "${grad_x_vals[@]}"; do
                    echo "Running with mean_base_theta=${t}pi, mean_curv_x=${cx}pi, mean_curv_y=${cy}pi, mean_grad_x=${gx}, mean_grad_y=${gy}"
        
                    python3 offline.py \
                        --mean_base_theta_pi "$t" \
                        --mean_curv_x_pi "$cx" \
                        --mean_curv_y_pi "$cy" \
                        --mean_mag_grad_x "$gx" \
                        --mean_mag_grad_y "$gx" \
            
                done
            done
        done
    done
done

echo "Sweep complete."
