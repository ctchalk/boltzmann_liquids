import numpy as np
import numba
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
import math
from tqdm import tqdm

# ===================
# Globals
# ===================

# Do not change these unless you understand the parallel GPU algorithms 
# as changing these can introduce race conditions and break detailed balance.
# If you need to change them, please e-mail me and I can discuss them with you.
# For now, the allowed values are:
# Canonical: Any subset of the Moore neighborhood
# Grand canonical: Must use default neighbor displacements (von Neumann)
DEFAULT_NEIGHBOR_DISPLACEMENTS = ((-1, 0, 0), ( 1, 0, 0), ( 0,-1, 0), 
                                  ( 0, 1, 0), ( 0, 0,-1), ( 0, 0, 1))

DEFAULT_SWAP_DIRECTIONS = (
    (-1, -1, -1), (-1, -1,  0), (-1, -1,  1),
    (-1,  0, -1), (-1,  0,  0), (-1,  0,  1),
    (-1,  1, -1), (-1,  1,  0), (-1,  1,  1),

    ( 0, -1, -1), ( 0, -1,  0), ( 0, -1,  1),
    ( 0,  0, -1),               ( 0,  0,  1),
    ( 0,  1, -1), ( 0,  1,  0), ( 0,  1,  1),

    ( 1, -1, -1), ( 1, -1,  0), ( 1, -1,  1),
    ( 1,  0, -1), ( 1,  0,  0), ( 1,  0,  1),
    ( 1,  1, -1), ( 1,  1,  0), ( 1,  1,  1))

# GPU config; you can adjust these based on your GPU
# Knowing how to set these optimally requires a deep knowledge
# of GPU architecture, which I don't have. I THINK that
# higher THREADS_PER_BLOCK_3D = faster and I'm pretty sure
# that higher THREADS_PER_BLOCK_1D is not necessarily faster
# and can be slower. Your GPU won't let you set them too high.
# These are worth playing with/reading about if you want performance.
THREADS_PER_BLOCK_3D = (8, 8, 8)
THREADS_PER_BLOCK_1D = 128

# ========================================
# n_ij and n_i counting functions/kernels
# ========================================
@cuda.jit
def n_ij_cuda_kernel(lattice, 
                     result, 
                     periodic,
                     neighbor_displacements):
    H, W, L = lattice.shape
    x, y, z = cuda.grid(3)

    if 0 <= x < H and 0 <= y < W and 0 <= z < L:
        current_species = lattice[x,y,z]

        for dx, dy, dz in neighbor_displacements:
            if periodic:
                nx, ny, nz = (x + dx) % H, (y + dy) % W, (z + dz) % L
                neighbor_species = lattice[nx, ny, nz]
                cuda.atomic.add(result, (current_species, neighbor_species), 1)
            else:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < H and 0 <= ny < W and 0 <= nz < L:
                    neighbor_species = lattice[nx, ny, nz]
                    cuda.atomic.add(result, (current_species, neighbor_species), 1)
    

@cuda.jit
def half_diagonal_cuda_kernel(result):
    i = cuda.grid(1)
    if i < result.shape[0]:
        result[i, i] = result[i, i] // 2


def get_n_ij(lattice, 
             n_sp, 
             periodic=True, 
             neighbor_displacements=DEFAULT_NEIGHBOR_DISPLACEMENTS):
    H, W, L = lattice.shape
    lattice_d = cuda.to_device(lattice)
    result = np.zeros((n_sp, n_sp), dtype=np.int32)
    result_d = cuda.to_device(result)

    blocks_per_grid = (
        (H + THREADS_PER_BLOCK_3D[0] - 1) // THREADS_PER_BLOCK_3D[0],
        (W + THREADS_PER_BLOCK_3D[1] - 1) // THREADS_PER_BLOCK_3D[1],
        (L + THREADS_PER_BLOCK_3D[2] - 1) // THREADS_PER_BLOCK_3D[2]
    )

    n_ij_cuda_kernel[blocks_per_grid, THREADS_PER_BLOCK_3D](lattice_d, 
                                                            result_d, 
                                                            periodic, 
                                                            neighbor_displacements)
    half_diagonal_cuda_kernel[n_sp, 1](result_d)
    result = result_d.copy_to_host()
    return result


@cuda.jit
def n_i_cuda_kernel(lattice, result):
    H, W, L = lattice.shape
    x, y, z = cuda.grid(3)

    if x >= H or y >= W or z >= L:
        return
    
    current_species = lattice[x,y,z]
    cuda.atomic.add(result, current_species, 1)


def get_n_i(lattice, n_sp):
    H, W, L = lattice.shape
    lattice_d = cuda.to_device(lattice)
    result = np.zeros(n_sp, dtype=np.int32)
    result_d = cuda.to_device(result)

    blocks_per_grid = (
        (H + THREADS_PER_BLOCK_3D[0] - 1) // THREADS_PER_BLOCK_3D[0],
        (W + THREADS_PER_BLOCK_3D[1] - 1) // THREADS_PER_BLOCK_3D[1],
        (L + THREADS_PER_BLOCK_3D[2] - 1) // THREADS_PER_BLOCK_3D[2]
    )

    n_i_cuda_kernel[blocks_per_grid, THREADS_PER_BLOCK_3D](lattice_d,
                                       result_d)
    result = result_d.copy_to_host()
    return result


# ==================================
# energy calculation helpers/kernels
# ==================================
@cuda.jit(device=True)
def local_energy(lattice,
                 position,
                 g_ij,
                 periodic,
                 neighbor_displacements,
                 lattice_dims):
    energy = 0.0

    H, W, L = lattice_dims

    x, y, z = position

    if periodic:
        for dx, dy, dz in neighbor_displacements:
            nx, ny, nz = (x + dx) % H, (y + dy) % W, (z + dz) % L
            energy += g_ij[int(lattice[nx, ny, nz])]

    else:
        for dx, dy, dz in neighbor_displacements:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < H and 0 <= ny < W and 0 <= nz < L:
                energy += g_ij[int(lattice[nx, ny, nz])]
    return energy


def config_energy(lattice, 
                  n_sp, 
                  g_ij, 
                  g_i, 
                  periodic, 
                  neighbor_displacements):
    result = get_n_ij(lattice, n_sp, periodic, neighbor_displacements)
    energy = 0.0
    for i in range(result.shape[0]):
        for j in range(i, result.shape[1]):
            energy += g_ij[i, j] * result[i, j]
    energy += (g_i * get_n_i(lattice, n_sp)).sum()
    return energy


# =====================================
# assorted lattice Markov chain kernels
# =====================================
@cuda.jit(device=True)
def smaller_or_equal_position(p1, p2):
    if p1[0] < p2[0]:
        return True
    elif p1[0] == p2[0]:
        if p1[1] < p2[1]:
            return True
        elif p1[1] == p2[1]:
            if p1[2] <= p2[2]:
                return True
    return False


@cuda.jit(device=True)
def get_random_free_species(rng_states,
                            thread_idx,
                            n_sp,
                            free_species):
    r = xoroshiro128p_uniform_float64(rng_states, thread_idx)
    r = int(math.floor(r * n_sp))

    while r == n_sp or not free_species[r]:
        r = xoroshiro128p_uniform_float64(rng_states, thread_idx)
        r = int(math.floor(r * n_sp))
    return r


@cuda.jit
def steps_kernel_gc(num_parallel_proposals,
                    lattice_batch,
                    n_sp,
                    kT_high_batch,
                    kT_low_batch,
                    free_pos_batch,
                    free_species_batch,
                    g_ij_batch,
                    g_i_batch,
                    rng_states,
                    periodic,
                    neighbor_displacements,
                    swap_directions,
                    even_positions,
                    odd_positions,
                    positions):
    idx = cuda.blockIdx.x
    lattice = lattice_batch[idx]
    free_pos = free_pos_batch[idx]
    free_species = free_species_batch[idx]
    g_ij = g_ij_batch[idx]
    g_i = g_i_batch[idx]
    H, W, L = lattice.shape

    kT_high = kT_high_batch[idx]
    kT_low = kT_low_batch[idx]

    inv_kT_shared = cuda.shared.array(1, dtype=numba.float64)
    inv_mult_shared = cuda.shared.array(1, dtype=numba.float64)

    checkerboard_parity_shared = cuda.shared.array(1, dtype=numba.int32)

    checkerboard_parity_shared[0] = 0

    init_thread_idx = cuda.threadIdx.x

    block_dim_x = cuda.blockDim.x

    global_thread_idx = init_thread_idx + idx * block_dim_x

    if init_thread_idx == 0:
        inv_kT_shared[0] = 1.0 / kT_high

        ratio = kT_high / kT_low
        inv_mult_shared[0] = math.pow(ratio, 1.0 / num_parallel_proposals)
    
    cuda.syncthreads()

    for _ in range(num_parallel_proposals):
        thread_idx = init_thread_idx

        inv_kT = inv_kT_shared[0]

        while thread_idx < even_positions.shape[0]:
            pos1 = even_positions[thread_idx] if checkerboard_parity_shared[0] == 0 else odd_positions[thread_idx]
            x = int(pos1[0])
            y = int(pos1[1])
            z = int(pos1[2])

            species1 = lattice[x, y, z]

            if free_pos[x, y, z] != 0 and free_species[species1] != 0:
                acceptance_rand = xoroshiro128p_uniform_float64(rng_states, global_thread_idx)
                species2 = get_random_free_species(rng_states,
                                                   global_thread_idx,
                                                   n_sp,
                                                   free_species)
                
                g_ij_1 = g_ij[species1]
                g_ij_2 = g_ij[species2]

                delta_e = 0

                delta_e -= local_energy(lattice, (x, y, z), g_ij_1, periodic, neighbor_displacements, (H, W, L))
                delta_e -= g_i[species1]

                lattice[x, y, z] = species2

                delta_e += local_energy(lattice, (x, y, z), g_ij_2, periodic, neighbor_displacements, (H, W, L))
                delta_e += g_i[species2]

                if (delta_e >= 0.0 and math.exp(-delta_e * inv_kT) <= acceptance_rand):
                    lattice[x, y, z] = species1
                
            thread_idx = thread_idx + block_dim_x

        if init_thread_idx == 0:
            checkerboard_parity_shared[0] = 1 - checkerboard_parity_shared[0]
        cuda.syncthreads()

        if init_thread_idx == 0:
            inv_kT_shared[0] *= inv_mult_shared[0]
        cuda.syncthreads()


@cuda.jit(device=True)
def get_int_less_than_k(rng_states, threadIdx, k):
    r = int(math.floor(xoroshiro128p_uniform_float64(rng_states, threadIdx) * k))
    while r == k:
        r = int(math.floor(xoroshiro128p_uniform_float64(rng_states, threadIdx) * k))
    return r


@cuda.jit
def steps_kernel_crange(num_parallel_proposals, 
                        lattice_batch,
                        n_sp,
                        kT_high_batch,
                        kT_low_batch,
                        free_pos_batch,
                        free_species_batch,
                        g_ij_batch,
                        g_i_batch,
                        rng_states,
                        periodic,
                        neighbor_displacements,
                        swap_directions,
                        even_positions,
                        odd_positions,
                        positions):
    idx = cuda.blockIdx.x
    lattice = lattice_batch[idx]
    free_positions = free_pos_batch[idx]
    g_ij = g_ij_batch[idx]
    H, W, L = lattice.shape

    kT_high = kT_high_batch[idx]
    kT_low = kT_low_batch[idx]

    inv_kT_shared = cuda.shared.array(1, dtype=numba.float64)
    inv_mult_shared = cuda.shared.array(1, dtype=numba.float64)

    local_offset_shared = cuda.shared.array(3, dtype=numba.int32)

    region_offset_shared = cuda.shared.array(3, dtype=numba.int32)

    init_thread_idx = cuda.threadIdx.x

    block_dim_x = cuda.blockDim.x

    global_thread_idx = init_thread_idx + idx * block_dim_x

    if init_thread_idx == 0:
        inv_kT_shared[0] = 1.0 / kT_high

        ratio = kT_high / kT_low
        inv_mult_shared[0] = math.pow(ratio, 1.0 / num_parallel_proposals)
    
    cuda.syncthreads()

    for _ in range(num_parallel_proposals):
        thread_idx = init_thread_idx

        inv_kT = inv_kT_shared[0]

        if thread_idx == 0:
            local_offset_shared[0] = get_int_less_than_k(rng_states, global_thread_idx, 4)
            local_offset_shared[1] = get_int_less_than_k(rng_states, global_thread_idx, 4)
            local_offset_shared[2] = get_int_less_than_k(rng_states, global_thread_idx, 4)
            region_offset_shared[0] = get_int_less_than_k(rng_states, global_thread_idx, H//4)
            region_offset_shared[1] = get_int_less_than_k(rng_states, global_thread_idx, W//4)
            region_offset_shared[2] = get_int_less_than_k(rng_states, global_thread_idx, L//4)
        cuda.syncthreads()

        while thread_idx < positions.shape[0]:
            x, y, z = positions[thread_idx]
            pos1 = ((x)+local_offset_shared[0], 
                    (y)+local_offset_shared[1],
                    (z)+local_offset_shared[2])
            
            distal_pos = ((pos1[0] + region_offset_shared[0] * 4) % H,
                          (pos1[1] + region_offset_shared[1] * 4) % W,
                          (pos1[2] + region_offset_shared[2] * 4) % L)
            
            if smaller_or_equal_position(pos1, distal_pos):
                s = len(swap_directions)
                r = get_int_less_than_k(rng_states, global_thread_idx, s)

                pos2 = ((distal_pos[0] + swap_directions[r][0]) % H,
                        (distal_pos[1] + swap_directions[r][1]) % W,
                        (distal_pos[2] + swap_directions[r][2]) % L)
                
                i1 = int(pos1[0])
                j1 = int(pos1[1])
                k1 = int(pos1[2])
                i2 = int(pos2[0])
                j2 = int(pos2[1])
                k2 = int(pos2[2])
                if (free_positions[i1, j1, k1] != 0) and (free_positions[i2, j2, k2] != 0):
                    acceptance_rand = xoroshiro128p_uniform_float64(rng_states, global_thread_idx)

                    species1 = lattice[i1, j1, k1]
                    species2 = lattice[i2, j2, k2]

                    g_ij_1 = g_ij[species1]
                    g_ij_2 = g_ij[species2]

                    delta_e = 0

                    delta_e -= local_energy(lattice, 
                                            pos1, 
                                            g_ij_1, 
                                            periodic,
                                            neighbor_displacements,
                                            (H, W, L))
                    delta_e -= local_energy(lattice,
                                            pos2,
                                            g_ij_2,
                                            periodic,
                                            neighbor_displacements,
                                            (H, W, L))
                    
                    lattice[i1, j1, k1] = species2
                    lattice[i2, j2, k2] = species1

                    delta_e += local_energy(lattice, 
                                            pos1, 
                                            g_ij_2, 
                                            periodic,
                                            neighbor_displacements, 
                                            (H, W, L))
                    delta_e += local_energy(lattice,
                                            pos2,
                                            g_ij_1,
                                            periodic,
                                            neighbor_displacements,
                                            (H, W, L))
                    
                    if (delta_e >= 0.0 and math.exp(-delta_e * inv_kT) <= acceptance_rand):
                        lattice[pos1] = species1
                        lattice[pos2] = species2

            thread_idx = thread_idx + block_dim_x
        cuda.syncthreads()

        if init_thread_idx == 0:
            inv_kT_shared[0] *= inv_mult_shared[0]
        cuda.syncthreads()

    cuda.syncthreads()


@cuda.jit
def steps_kernel_c(num_parallel_proposals,
                   lattice_batch,
                   n_sp,
                   kT_high_batch,
                   kT_low_batch,
                   free_pos_batch,
                   free_species_batch,
                   g_ij_batch,
                   g_i_batch,
                   rng_states,
                   periodic,
                   neighbor_displacements,
                   swap_directions,
                   even_positions,
                   odd_positions,
                   positions):
    idx = cuda.blockIdx.x
    lattice = lattice_batch[idx]
    free_pos = free_pos_batch[idx]
    g_ij = g_ij_batch[idx]
    H, W, L = lattice.shape

    kT_high = kT_high_batch[idx]
    kT_low = kT_low_batch[idx]

    inv_kT_shared = cuda.shared.array(1, dtype=numba.float64)
    inv_mult_shared = cuda.shared.array(1, dtype=numba.float64)

    local_offset_shared = cuda.shared.array(3, dtype=numba.int32)

    init_thread_idx = cuda.threadIdx.x

    block_dim_x = cuda.blockDim.x

    global_thread_idx = init_thread_idx + idx * block_dim_x

    if init_thread_idx == 0:
        inv_kT_shared[0] = 1.0 / kT_high

        ratio = kT_high / kT_low
        inv_mult_shared[0] = math.pow(ratio, 1.0 / num_parallel_proposals)
    
    cuda.syncthreads()

    for _ in range(num_parallel_proposals):
        thread_idx = init_thread_idx

        inv_kT = inv_kT_shared[0]

        if thread_idx == 0:
            local_offset_shared[0] = get_int_less_than_k(rng_states, global_thread_idx, 4)
            local_offset_shared[1] = get_int_less_than_k(rng_states, global_thread_idx, 4)
            local_offset_shared[2] = get_int_less_than_k(rng_states, global_thread_idx, 4)
        cuda.syncthreads()

        while thread_idx < positions.shape[0]:
            x, y, z = positions[thread_idx]
            i1 = (x)+local_offset_shared[0] 
            j1 = (y)+local_offset_shared[1]
            k1 = (z)+local_offset_shared[2]
            
            r = get_int_less_than_k(rng_states, global_thread_idx, len(swap_directions))

            dx = swap_directions[r][0]
            dy = swap_directions[r][1]
            dz = swap_directions[r][2]

            if periodic:
                i2 = (i1 + dx) % H
                j2 = (j1 + dy) % W
                k2 = (k1 + dz) % L
            else:
                i2 = i1 + dx
                j2 = j1 + dy
                k2 = k1 + dz
                # out-of-bounds neighbor => no proposal
                if i2 < 0 or i2 >= H or j2 < 0 or j2 >= W or k2 < 0 or k2 >= L:
                    thread_idx = thread_idx + block_dim_x
                    continue

            if (free_pos[i1, j1, k1] != 0) and (free_pos[i2, j2, k2] != 0):
                acceptance_rand = xoroshiro128p_uniform_float64(rng_states, global_thread_idx)

                species1 = lattice[i1, j1, k1]
                species2 = lattice[i2, j2, k2]

                g_ij_1 = g_ij[species1]
                g_ij_2 = g_ij[species2]

                delta_e = 0

                delta_e -= local_energy(lattice, 
                                        (i1, j1, k1), 
                                        g_ij_1, 
                                        periodic,
                                        neighbor_displacements,
                                        (H, W, L))
                delta_e -= local_energy(lattice,
                                        (i2, j2, k2),
                                        g_ij_2,
                                        periodic,
                                        neighbor_displacements,
                                        (H, W, L))
                
                lattice[i1, j1, k1] = species2
                lattice[i2, j2, k2] = species1

                delta_e += local_energy(lattice, 
                                        (i1, j1, k1), 
                                        g_ij_2, 
                                        periodic,
                                        neighbor_displacements, 
                                        (H, W, L))
                delta_e += local_energy(lattice,
                                        (i2, j2, k2),
                                        g_ij_1,
                                        periodic,
                                        neighbor_displacements,
                                        (H, W, L))
                
                if (delta_e >= 0.0 and math.exp(-delta_e * inv_kT) <= acceptance_rand):
                    lattice[(i1, j1, k1)] = species1
                    lattice[(i2, j2, k2)] = species2

            thread_idx = thread_idx + block_dim_x
        cuda.syncthreads()

        if init_thread_idx == 0:
            inv_kT_shared[0] *= inv_mult_shared[0]
        cuda.syncthreads()

    cuda.syncthreads()


@cuda.jit
def steps_kernel_hybrid_gc_crange(num_parallel_proposals,
                                  lattice_batch,
                                  n_sp,
                                  kT_high_batch,
                                  kT_low_batch,
                                  free_pos_batch,
                                  free_species_batch,
                                  g_ij_batch,
                                  g_i_batch,
                                  rng_states,
                                  periodic,
                                  neighbor_displacements,
                                  swap_directions,
                                  even_positions,
                                  odd_positions,
                                  positions):
    idx = cuda.blockIdx.x
    lattice = lattice_batch[idx]
    free_positions = free_pos_batch[idx]
    free_species = free_species_batch[idx]
    g_ij = g_ij_batch[idx]
    g_i = g_i_batch[idx]
    H, W, L = lattice.shape

    kT_high = kT_high_batch[idx]
    kT_low = kT_low_batch[idx]

    inv_kT_shared = cuda.shared.array(1, dtype=numba.float64)
    inv_mult_shared = cuda.shared.array(1, dtype=numba.float64)

    # --- shared "which protocol?" flag (0 = gc, 1 = crange) ---
    mode_shared = cuda.shared.array(1, dtype=numba.int32)

    checkerboard_parity_shared = cuda.shared.array(1, dtype=numba.int32)

    local_offset_shared = cuda.shared.array(3, dtype=numba.int32)
    region_offset_shared = cuda.shared.array(3, dtype=numba.int32)

    init_thread_idx = cuda.threadIdx.x
    block_dim_x = cuda.blockDim.x
    global_thread_idx = init_thread_idx + idx * block_dim_x

    if init_thread_idx == 0:
        inv_kT_shared[0] = 1.0 / kT_high
        ratio = kT_high / kT_low
        inv_mult_shared[0] = math.pow(ratio, 1.0 / num_parallel_proposals)

        checkerboard_parity_shared[0] = 0

    cuda.syncthreads()

    for _ in range(num_parallel_proposals):
        inv_kT = inv_kT_shared[0]

        # leader thread decides which protocol this step
        if init_thread_idx == 0:
            mode_shared[0] = get_int_less_than_k(rng_states, global_thread_idx, 2)

            if mode_shared[0] == 1:
                local_offset_shared[0] = get_int_less_than_k(rng_states, global_thread_idx, 4)
                local_offset_shared[1] = get_int_less_than_k(rng_states, global_thread_idx, 4)
                local_offset_shared[2] = get_int_less_than_k(rng_states, global_thread_idx, 4)

                region_offset_shared[0] = get_int_less_than_k(rng_states, global_thread_idx, H // 4)
                region_offset_shared[1] = get_int_less_than_k(rng_states, global_thread_idx, W // 4)
                region_offset_shared[2] = get_int_less_than_k(rng_states, global_thread_idx, L // 4)

        cuda.syncthreads()

        thread_idx = init_thread_idx

        # =========================
        # GC protocol (checkerboard)
        # =========================
        if mode_shared[0] == 0:
            while thread_idx < even_positions.shape[0]:
                pos1 = even_positions[thread_idx] if checkerboard_parity_shared[0] == 0 else odd_positions[thread_idx]

                i1 = int(pos1[0])
                j1 = int(pos1[1])
                k1 = int(pos1[2])

                species1 = lattice[i1, j1, k1]

                if free_positions[i1, j1, k1] != 0 and free_species[species1] != 0:
                    acceptance_rand = xoroshiro128p_uniform_float64(rng_states, global_thread_idx)

                    species2 = get_random_free_species(rng_states,
                                                       global_thread_idx,
                                                       n_sp,
                                                       free_species)

                    g_ij_1 = g_ij[species1]
                    g_ij_2 = g_ij[species2]

                    delta_e = 0.0
                    delta_e -= local_energy(lattice, (i1, j1, k1), g_ij_1, periodic, neighbor_displacements, (H, W, L))
                    delta_e -= g_i[species1]

                    lattice[i1, j1, k1] = species2

                    delta_e += local_energy(lattice, (i1, j1, k1), g_ij_2, periodic, neighbor_displacements, (H, W, L))
                    delta_e += g_i[species2]

                    # reject if needed
                    if (delta_e >= 0.0) and (math.exp(-delta_e * inv_kT) <= acceptance_rand):
                        lattice[i1, j1, k1] = species1

                thread_idx += block_dim_x

        # =========================
        # CRANGE protocol (ranged swap)
        # =========================
        else:
            while thread_idx < positions.shape[0]:
                x, y, z = positions[thread_idx]
                pos1 = (x + local_offset_shared[0],
                        y + local_offset_shared[1],
                        z + local_offset_shared[2])

                distal_pos = ((pos1[0] + region_offset_shared[0] * 4) % H,
                              (pos1[1] + region_offset_shared[1] * 4) % W,
                              (pos1[2] + region_offset_shared[2] * 4) % L)

                if smaller_or_equal_position(pos1, distal_pos):
                    s = swap_directions.shape[0]
                    r = get_int_less_than_k(rng_states, global_thread_idx, s)

                    pos2 = ((distal_pos[0] + swap_directions[r][0]) % H,
                            (distal_pos[1] + swap_directions[r][1]) % W,
                            (distal_pos[2] + swap_directions[r][2]) % L)

                    i1 = int(pos1[0]); j1 = int(pos1[1]); k1 = int(pos1[2])
                    i2 = int(pos2[0]); j2 = int(pos2[1]); k2 = int(pos2[2])

                    if (free_positions[i1, j1, k1] != 0) and (free_positions[i2, j2, k2] != 0):
                        acceptance_rand = xoroshiro128p_uniform_float64(rng_states, global_thread_idx)

                        species1 = lattice[i1, j1, k1]
                        species2 = lattice[i2, j2, k2]

                        g_ij_1 = g_ij[species1]
                        g_ij_2 = g_ij[species2]

                        delta_e = 0.0
                        delta_e -= local_energy(lattice, pos1, g_ij_1, periodic, neighbor_displacements, (H, W, L))
                        delta_e -= local_energy(lattice, pos2, g_ij_2, periodic, neighbor_displacements, (H, W, L))

                        lattice[i1, j1, k1] = species2
                        lattice[i2, j2, k2] = species1

                        delta_e += local_energy(lattice, pos1, g_ij_2, periodic, neighbor_displacements, (H, W, L))
                        delta_e += local_energy(lattice, pos2, g_ij_1, periodic, neighbor_displacements, (H, W, L))

                        # reject if needed
                        if (delta_e >= 0.0) and (math.exp(-delta_e * inv_kT) <= acceptance_rand):
                            lattice[i1, j1, k1] = species1
                            lattice[i2, j2, k2] = species2

                thread_idx += block_dim_x

        cuda.syncthreads()

        if init_thread_idx == 0 and mode_shared[0] == 0:
            checkerboard_parity_shared[0] = 1 - checkerboard_parity_shared[0]
        cuda.syncthreads()

        if init_thread_idx == 0:
            inv_kT_shared[0] *= inv_mult_shared[0]
        cuda.syncthreads()


# =====================================
# wrapper for Monte Carlo sampling using kernels
# =====================================
def mcmc(lattice_batch,
         num_parallel_proposals,
         num_samples,
         burn_in,
         num_species,
         g_ij_batch,
         g_i_batch = None,
         kT_batch = None,
         kT_high_batch = None,
         kT_low_batch = None,
         free_pos_batch = None,
         free_species_batch = None,
         periodic = False,
         canonical = False,
         ranged = False,
         hybrid = False,
         neighbor_displacements = DEFAULT_NEIGHBOR_DISPLACEMENTS,
         swap_directions = DEFAULT_SWAP_DIRECTIONS,
         seed = None,
         save_lattices = False,
         rng_states = None):
    """
        GPU-accelerated Metropolis-Hastings MCMC for the 3D lattice Boltzmann liquid model.

        This is a wrapper around several Numba-CUDA kernels. Each lattice in `lattice_batch`
        is simulated independently, with one CUDA block per lattice.
        Arguments called "*_batch", if passed as a list of arguments of length equal
        to the number of lattices given by lattice_batch, 
        will be used in each lattice (i.e., if a list of g_ij matrices are given, 
        then the nth lattice in the batch will use the nth g_ij matrix).
        All arguments called "*_batch" can also be passed in as singletons
        (i.e., a single lattice and a single g_ij; or, a list of lattices
        can be passed in along with a single g_ij matrix and each lattice
        will use the single g_ij matrix.)
        Within a kernel launch,
        the block runs many proposal steps without returning to the CPU. After burn-in, the
        wrapper advances the chain between samples, copies the lattice(s) back to host, and
        accumulates summary statistics (n_ij, n_i, energies).
        A save_lattices option allows the return of all sampled lattice configurations,
        instead of (by default) the single endpoint lattice configuration(s).

        Model and energy
        ----------------
        - Lattice sites store integer species IDs in [0, num_species-1].
        - Pairwise interaction energies are given by `g_ij` (symmetric).
        - Optional per-species energies are given by `g_i`.
        - The Metropolis accept probability uses:  min(1, exp(-ΔE / kT)).

        Update modes (kernel choice)
        ----------------------------
        Exactly one mode is used per call:
        - Grand-canonical (default): single-site resampling (site chooses a random new species).
        Uses a **checkerboard** update to avoid simultaneously updating interacting neighbors.
        - Canonical local swaps (canonical=True): swap species between two sites.
        Uses a **mod-4 sublattice** scheme to keep simultaneous swaps disjoint.
        - Canonical ranged swaps (canonical=True, ranged=True): long-range swaps between
        mod-4 sublattices plus a small displacement from `swap_directions`.
        - Hybrid (hybrid=True): randomly mixes GC steps and ranged-swap steps *inside the kernel*.

        Important: if `canonical=True`, it takes precedence over `hybrid=True` (hybrid is ignored).

        Temperature / annealing
        -----------------------
        Burn-in is run as a single kernel launch of `burn_in` proposals with geometric annealing:
        kT : kT_high_batch -> kt_low_batch over `burn_in` steps (if different).
        During sampling, each inter-sample advance runs at constant temperature kt_low_batch
        (the wrapper passes kT_high = kT_low for those kernel calls).

        Parameters
        ----------
        lattice_batch : array-like (int32)
            Either a single lattice of shape (H, W, L) or a batch of lattices of shape
            (B, H, W, L). All lattices are assumed to have the same shape.
            H, W, and L must each be multiples of 4.

        num_parallel_proposals : int
            A budget used to compute the inter-sample spacing:
                sample_rate = (num_parallel_proposals - burn_in) // num_samples
            The actual number of proposals performed is:
                burn_in + num_samples * sample_rate
            (This is <= num_parallel_proposals due to floor division.)

        num_samples : int
            Number of samples to collect after burn-in.

        burn_in : int
            Number of proposal steps during burn-in (annealed from kT_high to kT_low if provided).

        num_species : int
            Number of species. Lattice entries must be valid indices: 0 <= s < num_species.

        g_ij_batch : array-like (float)
            Pairwise interaction energies. Accepts shape (num_species, num_species) or
            (B, num_species, num_species). For correctness with the swap kernels, `g_ij`
            is expected to represent symmetric interactions (g_ij[i,j] == g_ij[j,i]).

        g_i_batch : array-like (float), optional
            Per-species energies. Shape (num_species,) or (B, num_species). If omitted, zeros.
            Used by GC updates (and for reported energies). In canonical modes, counts are fixed,
            so `g_i` does not affect acceptance, but it is still included in configuration energy.

        kT_batch : float or array-like (float), optional
            Convenience temperature input. If provided and kT_high_batch/kt_low_batch are not,
            the wrapper sets kT_high = kT_low = kT_batch (i.e., constant temperature).

        kT_high_batch, kT_low_batch : float or array-like (float), optional
            Burn-in annealing endpoints. Each may be scalar or length-B. If neither is provided,
            both default to 1.0.

        free_pos_batch : array-like (bool/int), optional
            Mask of free (updatable) lattice positions. True/1 = free, False/0 = clamped.
            Accepts shape (H, W, L) or (B, H, W, L). If omitted, all positions are free.

        free_species_batch : array-like (int), optional
            Mask of free species IDs, used only in GC (and GC part of hybrid):
            proposals that would change to or from a non-free species are suppressed.
            Shape (num_species,) or (B, num_species). If omitted, all species are free.

        periodic : bool
            Boundary condition for local energy evaluation and neighbor addressing.
            - In the non-ranged canonical case, False here does prevent periodic swap proposals.

        canonical : bool
            If True, use canonical swap updates instead of GC resampling.

        ranged : bool
            Only used if canonical=True. If True, use ranged-swap kernel; else local-swap kernel.

        hybrid : bool
            If True (and canonical=False), use the hybrid kernel (randomly mixes GC and ranged steps).

        neighbor_displacements : tuple/list of (dx, dy, dz)
            Defines the neighborhood used in energy calculations.
            Constraints for correctness / detailed balance with the parallel update schemes:
            - GC mode: must be von Neumann (DEFAULT_NEIGHBOR_DISPLACEMENTS). Using Moore
            or larger neighborhoods causes race conditions which (probably) break detailed balance.
            - Canonical modes: must be a subset of the Moore neighborhood (dx,dy,dz ∈ {-1,0,1})
            to prevent race conditions. In short, for an extended neighborhood, proposals must be 
            changed to a mod-k sublattice scheme (it is currently hardcoded mod-4).

        swap_directions : tuple/list of (dx, dy, dz)
            Possible displacements for swap proposals in canonical kernels.
            Must be a subset of the Moore neighborhood (dx,dy,dz ∈ {-1,0,1}) to prevent
            races/overlapping swaps under the mod-4 sublattice scheme.

        seed : int, optional
            RNG seed for xoroshiro128p states (ignored if `rng_states` is provided).

        save_lattices : bool
            If True, returns a list of sampled lattice configurations (host copies).

        rng_states : numba.cuda.random.Xoroshiro128pStates, optional
            Pre-initialized RNG states. Must have length:
                THREADS_PER_BLOCK_1D * num_lattices
            (One RNG stream per thread across all blocks.)

        Returns
        -------
        If save_lattices is False:
            lattice_final, avg_n_ij, avg_n_i, energies_list

        If save_lattices is True:
            saved_lattices, avg_n_ij, avg_n_i, energies_list

        Where:
        - lattice_final:
            Final lattice configuration(s), shape (H,W,L) if B=1 else (B,H,W,L).
            (Note: `saved_lattices` entries are not squeezed even when B=1.)
        - avg_n_ij:
            Sample-mean interaction counts. Shape (num_species,num_species) if B=1 else
            (B,num_species,num_species).
        - avg_n_i:
            Sample-mean species counts. Shape (num_species,) if B=1 else (B,num_species).
        - energies_list:
            Energies for bookkeeping, shape (num_samples+1,) if B=1 else (B,num_samples+1).
            energies_list[..., 0] is computed from the **initial** configuration (pre-burn-in);
            energies during burn-in are not recorded.

        Critical reminders
        ------------------------------
        - Canonical modes require lattice dimensions H, W, L to be multiples of 4.
        (The mod-4 offset scheme can otherwise index beyond lattice bounds.)
        - Ensure `num_parallel_proposals > burn_in` and that
            (num_parallel_proposals - burn_in) // num_samples >= 1
        or else sample_rate may be 0 (yielding repeated identical samples).
        - All lattices in the batch must share the same shape.
    """
    sample_rate = (num_parallel_proposals - burn_in) // num_samples

    lattice_batch = np.array(lattice_batch, dtype=np.int32)
    if lattice_batch.ndim == 3:
        lattice_batch = np.array([lattice_batch])

    num_lattices = lattice_batch.shape[0]
    
    H, W, L = lattice_batch[0].shape

    g_ij_batch = np.array(g_ij_batch)
    if g_ij_batch.ndim == 2:
        g_ij_batch = [g_ij_batch] * num_lattices

    if g_i_batch is None:
        g_i_batch = np.zeros((num_lattices, num_species), dtype=np.float64)
    else:
        g_i_batch = np.array(g_i_batch)
        if g_i_batch.ndim == 1:
            g_i_batch = np.tile(np.asarray(g_i_batch, dtype=np.float64), (num_lattices, 1))

    if kT_batch is None and kT_high_batch is None and kT_low_batch is None:
        kT_high_batch = np.ones(num_lattices, dtype=np.float64)
        kT_low_batch = np.ones(num_lattices, dtype=np.float64)
    elif kT_high_batch is None and kT_low_batch is None:
        kT_high_batch = np.array(kT_batch)
        kT_low_batch = np.array(kT_batch)
        if np.ndim(kT_high_batch) == 0:
            kT_high_batch = np.full(num_lattices, float(kT_high_batch), dtype=np.float64)
            kT_low_batch = np.full(num_lattices, float(kT_low_batch), dtype=np.float64)
    else:
        kT_high_batch = np.array(kT_high_batch)
        kT_low_batch = np.array(kT_low_batch)
        if np.ndim(kT_high_batch) == 0:
            kT_high_batch = np.full(num_lattices, float(kT_high_batch), dtype=np.float64)
            kT_low_batch = np.full(num_lattices, float(kT_low_batch), dtype=np.float64)

    if free_pos_batch is None:
        free_pos_batch = []
        for lattice in lattice_batch:
            free_pos_batch.append(np.ones_like(lattice))
    else:
        free_pos_batch = np.array(free_pos_batch, dtype=np.bool)
        if free_pos_batch.ndim == 3:
            free_pos_batch = [free_pos_batch] * num_lattices
        
    # Hacky but reduces logic checks in GPU
    # free_positions will be padded to nearest multiple of 4
    # positions outside the lattice but within next multiple of 4 are considered not free
    # Thus the position freeness check also checks that the position is within the lattice
    new_free_pos_list = []
    for free_positions in free_pos_batch:
        # Next powers of 4
        H4 = H + ((4-H) % 4)
        W4 = W + ((4-W) % 4)
        L4 = L + ((4-L) % 4)
        new_free_pos = np.zeros((H4, W4, L4), dtype=np.bool)
        new_free_pos[:H, :W, :L] = free_positions
        new_free_pos_list.append(new_free_pos)
    free_pos_batch = new_free_pos_list

    if free_species_batch is None:
        free_species_batch = np.tile(np.ones(num_species, dtype=np.int32), (num_lattices, 1))
    elif np.ndim(free_species_batch) == 1:
        free_species_batch = np.tile(np.asarray(free_species_batch, dtype=np.int32), (num_lattices, 1))

    if canonical:
        if ranged:
            kernel = steps_kernel_crange
        else: 
            kernel = steps_kernel_c
    elif hybrid:
        kernel = steps_kernel_hybrid_gc_crange
    else:
        kernel = steps_kernel_gc

    if seed is None:
        seed = np.random.randint(0, 2**32)

    if save_lattices:
        saved_lattices = []

    lattices_d = cuda.to_device(lattice_batch)
    g_ijs_d = cuda.to_device(g_ij_batch)
    g_is_d = cuda.to_device(g_i_batch)
    free_pos_list_d = cuda.to_device(free_pos_batch)
    free_species_list_d = cuda.to_device(free_species_batch)
    neighbor_displacements_d = cuda.to_device(np.asarray(neighbor_displacements, dtype=np.int32))
    swap_directions_d = cuda.to_device(np.asarray(swap_directions, dtype=np.int32))

    # Lattices are assumed same shape. Can address later.
    H, W, L = lattice_batch[0].shape

    positions = []

    for i in range(H):
        for j in range(W):
            for k in range(L):
                if i % 4 == 0 and j % 4 == 0 and k % 4 == 0:
                    positions.append((i, j, k))

    positions_d = cuda.to_device(np.asarray(positions, dtype=np.int32))

    even_positions = []
    odd_positions = []

    for i in range(H):
        for j in range(W):
            for k in range(L):
                if (i+j+k) % 2 == 0:
                    even_positions.append((i, j, k))
                else:
                    odd_positions.append((i, j, k))

    even_positions_d = cuda.to_device(np.asarray(even_positions, dtype=np.int32))
    odd_positions_d = cuda.to_device(np.asarray(odd_positions, dtype=np.int32))

    n_threads = THREADS_PER_BLOCK_1D * num_lattices
    blocks_per_grid = (num_lattices,)

    block_dim = (THREADS_PER_BLOCK_1D,)

    if rng_states is None:
        rng_states = create_xoroshiro128p_states(n_threads, seed=seed)

    n_ijs = np.zeros((num_lattices, num_species, num_species), dtype = np.int32)
    n_is = np.zeros((num_lattices, num_species), dtype = np.int32)
    energies_list = np.zeros((num_lattices, num_samples+1), dtype=np.float64)
    for i in range(num_lattices):
        energies_list[i][0] = config_energy(lattice_batch[i], num_species, g_ij_batch[i], g_i_batch[i], periodic, neighbor_displacements)

    kernel[blocks_per_grid, block_dim](burn_in,
                                       lattices_d,
                                       num_species,
                                       kT_high_batch,
                                       kT_low_batch,
                                       free_pos_list_d,
                                       free_species_list_d,
                                       g_ijs_d,
                                       g_is_d,
                                       rng_states,
                                       periodic,
                                       neighbor_displacements_d,
                                       swap_directions_d,
                                       even_positions_d,
                                       odd_positions_d,
                                       positions_d)
    cuda.synchronize()

    for step in tqdm(range(num_samples)):
        kernel[blocks_per_grid, block_dim](sample_rate,
                                           lattices_d,
                                           num_species,
                                           kT_low_batch,
                                           kT_low_batch,
                                           free_pos_list_d,
                                           free_species_list_d,
                                           g_ijs_d,
                                           g_is_d,
                                           rng_states,
                                           periodic,
                                           neighbor_displacements_d,
                                           swap_directions_d,
                                           even_positions_d,
                                           odd_positions_d,
                                           positions_d)
        cuda.synchronize()
        lattice_batch = lattices_d.copy_to_host()

        for i in range(num_lattices):
            n_ijs[i] += get_n_ij(lattice_batch[i], num_species, periodic, neighbor_displacements)
            n_is[i] += get_n_i(lattice_batch[i], num_species)
            energies_list[i][step+1] = config_energy(lattice_batch[i], num_species, g_ij_batch[i], g_i_batch[i], periodic, neighbor_displacements)

        if save_lattices:
            saved_lattices.append(np.copy(lattice_batch))
        
    cuda.synchronize()

    if lattice_batch.shape[0] == 1:
        lattice_batch = lattice_batch[0]

    avg_n_ijs = n_ijs / num_samples

    if avg_n_ijs.shape[0] == 1:
        avg_n_ijs = avg_n_ijs[0]

    avg_n_is = n_is / num_samples

    if avg_n_is.shape[0] == 1:
        avg_n_is = avg_n_is[0]

    if energies_list.shape[0] == 1:
        energies_list = energies_list[0]
    
    if save_lattices:
        return saved_lattices, avg_n_ijs, avg_n_is, energies_list
    else:
        return lattice_batch, avg_n_ijs, avg_n_is, energies_list