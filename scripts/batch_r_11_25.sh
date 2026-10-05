for val in 11 13 17 19 21 23 25; do
    sed "s/^num_loc_basis=.*/num_loc_basis=$val/" online_anis2.sh | sbatch
    sed "s/^num_loc_basis=.*/num_loc_basis=$val/" online_anis4.sh | sbatch
done
