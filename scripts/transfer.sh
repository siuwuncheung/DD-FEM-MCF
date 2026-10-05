timestamp=$(date "+%Y%m%d-%H%M%S")
destination_dir="/p/lustre2/${USER}/scp_local/DD-FEM-results-${timestamp}"
echo "${destination_dir}"
mkdir ${destination_dir}

rsync -av --exclude='*.pkl' --exclude='*.npz' single_null_* ${destination_dir}
rsync -av --exclude='*.pkl' --exclude='*.npz' double_null_* ${destination_dir}
rsync -av --exclude='*.pkl' --exclude='*.npz' magnetic_island_* ${destination_dir}

mv slurm-*.out ${destination_dir}
