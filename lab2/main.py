import numpy as np
import sympy
import time
import os

input_filename = 'input.txt'
output_filename = 'output.txt'

def inverse_permutation(permutation):
    n = len(permutation)
    inverse_perm = [0] * n

    for i, p in enumerate(permutation):
        inverse_perm[p] = i
    
    return inverse_perm

def apply_permutation(matrix, permutation):
    return matrix[:, permutation]

def row_echelon_form(matrix, q):
    A = np.array(matrix, dtype=int)
    rows, cols = A.shape
    
    r = 0
    pivot_columns = []

    for c in range(cols):
        for i in range(r, rows):
            if A[i, c] != 0:
                pivot_columns.append(c)

                if i != r:
                    A[[r, i]] = A[[i, r]]

                pivot_value = A[r, c]
                
                try:
                    inverse = int(sympy.mod_inverse(pivot_value, q))
                    A[r] = (A[r] * inverse) % q
                except:
                    print("Matrix is not invertible!")
                    return np.array([]), []

                for j in range(r + 1, rows):
                    if A[j, c] != 0:
                        mult = A[j, c]
                        A[j] = (A[j] - mult * A[r]) % q 

                r += 1
                break

        if r == rows:
            break

    return A, pivot_columns

def extract_nonsingular_minor(matrix, pivot_cols):
    rank = len(pivot_cols)

    if rank == 0:
        return np.array([])
    
    nonsingular_minor = matrix[:rank, pivot_cols]
    
    return nonsingular_minor

def standard_form_generator(matrix, q):
    echelon, pivot_cols = row_echelon_form(matrix, q)
    rank = len(pivot_cols)
    
    if rank == 0:
        return matrix, list(range(matrix.shape[1])), 0
    
    all_columns = list(range(matrix.shape[1]))
    non_pivot_cols = [col for col in all_columns if col not in pivot_cols]
    permutation = pivot_cols + non_pivot_cols

    reordered = echelon[:, permutation]

    standard_form = np.array(reordered[:rank], dtype=int)

    for i in range(rank-1, -1, -1):
        for j in range(i-1, -1, -1):
            if standard_form[j, i] != 0:
                mult = int(standard_form[j, i])
                standard_form[j] = (standard_form[j] - mult * standard_form[i]) % q

    for i in range(rank):
        if standard_form[i, i] != 1:
            raise ValueError("Algorithm error!")
        
        for j in range(i+1, rank):
            if standard_form[j, i] != 0:
                mult = int(standard_form[j, i])
                standard_form[j] = (standard_form[j] - mult * standard_form[i]) % 1

    return standard_form, permutation, rank

def get_parity_check_matrix(matrix, q):
    cannonical_form, permutation, rank = standard_form_generator(matrix, q)

    if rank == 0 or rank == matrix.shape[1]:
        return np.array([]), permutation
    
    k = rank
    n = cannonical_form.shape[1]

    A = cannonical_form[:, k:]
    H = np.zeros((n-k, n), dtype=int)

    H[:, :k] = (-A.T) % q

    for i in range(n-k):
        H[i, k+i] = 1

    return H, permutation

def coding(matrix, q):
    H, permutation = get_parity_check_matrix(matrix, q)
    inverse_perm = inverse_permutation(permutation)
    final_H = apply_permutation(H, inverse_perm)

    return final_H

def get_matrix_from_file(in_file):
    matrix = []

    with open(in_file, 'r') as file:
        content = file.read().strip()
        content = content[1:-1].strip()
        rows = content.split('],')

        for row in rows:
            clean_row = row.strip().strip('[').strip(']').strip()
            row_values = [int(value) for value in clean_row.split(',')]

            matrix.append(row_values)
    
    np_matrix = np.array(matrix)
    max_value = np.max(np_matrix)

    return np_matrix, sympy.nextprime(max_value)

def write_matrix_to_file(matrix, out_file, time_diff):
    if os.path.exists(out_file):
        os.remove(out_file)

    with open(out_file, 'w') as file:
        str_matrix = str(matrix).replace(']\n ', '],\n').replace(' ', ',')

        file.write(str_matrix)
        file.write('\n\n')
        file.write(f'Time: {time_diff}ms')

def main():
    start = time.perf_counter()
    matrix, q = get_matrix_from_file(input_filename)
    result = coding(matrix, q)
    end = time.perf_counter()

    diff = (end - start) * 1000
    
    write_matrix_to_file(result, output_filename, diff)

if __name__ == "__main__":
    main()
