#!/bin/bash
#SBATCH -J online_anis8
#SBATCH -p pbatch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH --exclusive
#SBATCH --mem=128G

order=8
n_FOM_x=2048

num_mode=20

Nx=256

num_loc_basis=25

#medium_type="single_null"
#medium_type="double_null"
medium_type="magnetic_island"

verbosity=2

VENV_DIR=/p/lustre2/${USER}/DD-FEM-MCF-venv/
module load python/3.13.2
eval "$(conda shell.bash hook)"
conda activate ${VENV_DIR}

python3 online.py --root_dir fourier_mode_harmonic_extension --multiply_POU 1 --train_anisotropy_order ${order} --num_loc_bc_mode ${num_mode} --medium_type ${medium_type} --test_anisotropy_order ${order} --n_FOM_x ${n_FOM_x} --Nx ${Nx} --verbosity ${verbosity} --num_loc_basis ${num_loc_basis} --solve_FOM -1
