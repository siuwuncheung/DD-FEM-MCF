import time
import pickle
import numpy as np
import scipy.sparse as sp
import scipy.io as sio 
import pyamg
from scipy.sparse.linalg import splu, cg, spsolve_triangular
from sksparse.cholmod import ldl_factor

def AMG_PCG(A, b, run_dir, log_filename):
    CG_file = open(f'{run_dir}/{log_filename}', "w")
    CG_file.write("Iteration, Residual_Norm\n")
    iteration = 0

    def callback_handler(x):
        nonlocal iteration
        iteration += 1
        residual = np.linalg.norm(b - A @ x)
        CG_file.write(f"{iteration}, {residual}\n")

    ml = pyamg.smoothed_aggregation_solver(A)
    M = ml.aspreconditioner()

    start_time = time.process_time()
    x, info = cg(A, b, M=M, callback=callback_handler, rtol=1e-8)
    end_time = time.process_time()
    time_lapse = end_time - start_time

    if info == 0:
        CG_file.write(f"Converged in {iteration} iterations.")
    elif info > 0:
        CG_file.write(f"Convergence failed: Maximum iterations {iteration} reached.")
    else:
        CG_file.write(f"Error: Solver encountered an issue (info={info}).")

    CG_file.close()

    return x, time_lapse

def get_factor(option): 
    factor_file, factor_func = {
        1: ('ROM_Cholesky.pkl', ldl_factor),
        2: ('ROM_SuperLU.pkl', splu),
    }.get(option, (None, None))
    if factor_file is None:
        raise ValueError(f"Unknown option: {option}")
    return factor_file, factor_func

def factorize(A, factor_func):
    start_time = time.process_time()
    f = factor_func(A.tocsc())
    end_time = time.process_time()
    time_lapse = end_time - start_time
    return f, time_lapse

def solve_factor(f, b):
    start_time = time.process_time()
    x = f.solve(b)
    end_time = time.process_time()
    time_lapse = end_time - start_time
    return x, time_lapse

def save_SPLU(A, filename):
    """Saves a SciPy SuperLU factorization object to disk."""
    start_time = time.process_time()
    factor_obj = splu(A.tocsc())
    end_time = time.process_time()
    time_factor = end_time - start_time
    np.savez_compressed(
        filename,
        time_factor=time_factor,
        L_data=factor_obj.L.data,
        L_indices=factor_obj.L.indices,
        L_indptr=factor_obj.L.indptr,
        L_shape=factor_obj.L.shape,
        U_data=factor_obj.U.data,
        U_indices=factor_obj.U.indices,
        U_indptr=factor_obj.U.indptr,
        U_shape=factor_obj.U.shape,
        perm_r=factor_obj.perm_r,
        perm_c=factor_obj.perm_c,
        factor_type='SPLU'
    )

def solve_SPLU(b, filename):
    """Loads SuperLU factorization components from disk."""
    """Performs a linear solve Ax = b using loaded SuperLU components."""
    data = np.load(filename)

    time_factor = data['time_factor']
    L = sp.csc_matrix((data['L_data'], data['L_indices'], data['L_indptr']), shape=data['L_shape'])
    U = sp.csc_matrix((data['U_data'], data['U_indices'], data['U_indptr']), shape=data['U_shape'])
    perm_r = data['perm_r']
    perm_c = data['perm_c']
    
    start_time = time.process_time()

    # Apply row permutation to the right-hand side
    b_perm = b[perm_r]
    
    # Forward substitution: L * w = b_perm (SuperLU L has a unit diagonal)
    w = spsolve_triangular(L, b_perm, lower=True, unit_diagonal=True)
    
    # Back substitution: U * y = w
    y = spsolve_triangular(U, w, lower=False, unit_diagonal=False)
    
    # Apply column permutation to obtain the final solution x
    x = np.empty_like(y)
    x[perm_c] = y

    end_time = time.process_time()
    time_lapse = end_time - start_time
    return x, time_factor, time_lapse

def save_LDL(A, filename):
    """Saves an sksparse LDL factorization object to disk."""
    start_time = time.process_time()
    factor_obj = ldl_factor(A.tocsc())
    end_time = time.process_time()
    time_factor = end_time - start_time

    L = factor_obj.L
    D = np.asarray(factor_obj.D.diagonal())
    perm = factor_obj.perm
    
    np.savez_compressed(
        filename,
        time_factor=time_factor,
        L_data=L.data,
        L_indices=L.indices,
        L_indptr=L.indptr,
        L_shape=L.shape,
        D=D,
        perm=perm,
        factor_type='LDL'
    )

def solve_LDL(b, filename):
    """Loads sksparse LDL factorization components from disk."""
    """Performs a linear solve Ax = b using loaded sksparse LDL components."""
    data = np.load(filename)

    time_factor = data['time_factor']
    L = sp.csc_matrix((data['L_data'], data['L_indices'], data['L_indptr']), shape=data['L_shape'])
    D = data['D']
    perm = data['perm']
    
    start_time = time.process_time()

    # Permute the right-hand side: b_perm
    b_perm = b[perm]
    
    # 1. Forward substitution: L * w = b_perm (sksparse L has a unit diagonal)
    w = spsolve_triangular(L, b_perm, lower=True, unit_diagonal=True)
    
    # 2. Scale by the diagonal matrix D: z = w / D
    z = w / D
    
    # 3. Back substitution: L^T * y = z
    y = spsolve_triangular(L.T, z, lower=False, unit_diagonal=True)
    
    # 4. Un-permute to map the solution back to original coordinates
    x = np.empty_like(y)
    x[perm] = y

    end_time = time.process_time()
    time_lapse = end_time - start_time
    return x, time_factor, time_lapse

def FOM_solution(FOM_DA, FOM_RHS, run_dir):
    u_FOM, time_FOM = AMG_PCG(FOM_DA, FOM_RHS, run_dir, "FOM_log.txt")
    return u_FOM, time_FOM

def patch_solution(i_center, j_center, patch_size, ny, Ny, nx, Nx, is_DOF, FOM_DA, FOM_RHS, run_dir):
    i_min = max(0, i_center - patch_size)
    i_max = min(Nx - 1, i_center + patch_size - 1)
    j_min = max(0, j_center - patch_size)
    j_max = min(Ny - 1, j_center + patch_size - 1)
    patch_mask = np.zeros(
        ((ny * Ny + 1), (nx * Nx + 1)),
        dtype=bool
    )
    patch_mask[
        j_min * ny : (j_max + 1) * ny + 1,
        i_min * nx : (i_max + 1) * nx + 1
    ] = True
    patch_mask = patch_mask.flatten(order='F')
    patch_mask_DOF = patch_mask[is_DOF]
    patch_dofs = np.where(patch_mask_DOF)[0]
    patch_DA = FOM_DA[np.ix_(patch_dofs, patch_dofs)]
    patch_RHS = FOM_RHS[patch_dofs]
    u_patch, time_patch = AMG_PCG(patch_DA, patch_RHS, run_dir, "patch_log.txt")
    return patch_dofs, u_patch, time_patch

def ROM_solution(ROM_DA, ROM_RHS, basis_dir, option):
    if option == 1:
        u_ROM, time_factor, time_ROM = solve_LDL(ROM_RHS, f"{basis_dir}/ROM_DA_LDL.npz") 
    elif option == 2:
        u_ROM, time_factor, time_ROM = solve_SPLU(ROM_RHS, f"{basis_dir}/ROM_DA_SPLU.npz") 
    else:
        print("Invalid input option.")
    return u_ROM, time_factor, time_ROM
