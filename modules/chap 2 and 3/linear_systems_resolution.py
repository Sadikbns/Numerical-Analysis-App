import numpy as np

def solve_iteratif(A, b, x0, tol, methode="jacobi", max_iter=20):
    D = np.diag(np.diag(A))
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)

    if methode == "jacobi":
        B = np.linalg.inv(D) @ (L + U)
        f_vec = np.linalg.inv(D) @ b
    else:
        B = np.linalg.inv(D - L) @ U
        f_vec = np.linalg.inv(D - L) @ b

    rho_B = max(abs(np.linalg.eigvals(B)))
    x = x0.astype(float)
    history = []

    for k in range(max_iter):
        x_new = B @ x + f_vec
        err = np.linalg.norm(x_new - x, np.inf)
        history.append([k + 1, x_new.copy(), err])
        if err < tol:
            break
        x = x_new

    return B, rho_B, history


def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=100):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    if A.shape != (n, n):
        raise ValueError("A must be a square matrix compatible with b")

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.array(x0, dtype=float)

    history = []
    for k in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            if abs(A[i, i]) < 1e-15:
                raise ValueError("Zero pivot encountered in Gauss-Seidel")

            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1 :], x_old[i + 1 :])
            x[i] = (b[i] - s1 - s2) / A[i, i]

        err = np.linalg.norm(x - x_old, ord=np.inf)
        history.append([k + 1, x.copy(), err])
        if err < tol:
            break

    return x, history


def _back_substitution(U, y):
    n = len(y)
    x = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-15:
            raise ValueError("Singular upper-triangular matrix")
        x[i] = (y[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]

    return x


def _forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n, dtype=float)

    for i in range(n):
        if abs(L[i, i]) < 1e-15:
            raise ValueError("Singular lower-triangular matrix")
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y


def gaussian_elimination_partial_pivot(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    if A.shape != (n, n):
        raise ValueError("A must be a square matrix compatible with b")

    M = np.hstack([A, b.reshape(-1, 1)])

    for k in range(n - 1):
        pivot_row = k + np.argmax(np.abs(M[k:, k]))
        if abs(M[pivot_row, k]) < 1e-15:
            raise ValueError("Matrix is singular or nearly singular")
        if pivot_row != k:
            M[[k, pivot_row]] = M[[pivot_row, k]]

        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k:] -= factor * M[k, k:]

    U = M[:, :n]
    y = M[:, n]
    x = _back_substitution(U, y)
    return x, U, y


def gaussian_elimination_total_pivot(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    if A.shape != (n, n):
        raise ValueError("A must be a square matrix compatible with b")

    M = np.hstack([A.copy(), b.reshape(-1, 1)])
    perm = np.arange(n)

    for k in range(n - 1):
        sub = np.abs(M[k:n, k:n])
        i_rel, j_rel = np.unravel_index(np.argmax(sub), sub.shape)
        i_max, j_max = k + i_rel, k + j_rel

        if abs(M[i_max, j_max]) < 1e-15:
            raise ValueError("Matrix is singular or nearly singular")

        if i_max != k:
            M[[k, i_max]] = M[[i_max, k]]
        if j_max != k:
            M[:, [k, j_max]] = M[:, [j_max, k]]
            perm[[k, j_max]] = perm[[j_max, k]]

        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i, k:] -= factor * M[k, k:]

    U = M[:, :n]
    y = M[:, n]
    x_permuted = _back_substitution(U, y)

    x = np.zeros(n, dtype=float)
    x[perm] = x_permuted
    return x, U, y, perm


def lu_decomposition(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    L = np.eye(n, dtype=float)
    U = A.copy()
    P = np.eye(n, dtype=float)

    for k in range(n - 1):
        pivot = k + np.argmax(np.abs(U[k:, k]))
        if abs(U[pivot, k]) < 1e-15:
            raise ValueError("Matrix is singular or nearly singular")

        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            P[[k, pivot]] = P[[pivot, k]]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U


def solve_lu(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    P, L, U = lu_decomposition(A)
    pb = P @ b
    y = _forward_substitution(L, pb)
    x = _back_substitution(U, y)
    return x, P, L, U


def cholesky_decomposition(A):
    A = np.array(A, dtype=float)

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if not np.allclose(A, A.T, atol=1e-12):
        raise ValueError("A must be symmetric for Cholesky")

    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            s = np.dot(L[i, :j], L[j, :j])
            if i == j:
                val = A[i, i] - s
                if val <= 0:
                    raise ValueError("A must be positive definite for Cholesky")
                L[i, j] = np.sqrt(val)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]

    return L


def solve_cholesky(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    L = cholesky_decomposition(A)
    y = _forward_substitution(L, b)
    x = _back_substitution(L.T, y)
    return x, L
# Convergence Conditions for Iterative Methods
def is_strictly_diagonally_dominant(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    
    for i in range(n):
        diag = abs(A[i, i])
        sum_other = np.sum(np.abs(A[i, :]) - np.abs(A[i, i]))
        if diag <= sum_other:
            return False
    
    return True


def is_symmetric_positive_definite(A):
    
    A = np.array(A, dtype=float)
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    
    # Check symmetry
    if not np.allclose(A, A.T, atol=1e-10):
        return False
    
    # Check positive definiteness using eigenvalues
    eigenvalues = np.linalg.eigvalsh(A)
    
    return np.all(eigenvalues > 1e-10)


def induced_matrix_norm(A, norm_type=2):
  
    A = np.array(A, dtype=float)
    
    return np.linalg.norm(A, ord=norm_type)


def spectral_radius(A):
   
    A = np.array(A, dtype=float)
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    
    eigenvalues = np.linalg.eigvals(A)
    rho = np.max(np.abs(eigenvalues))
    
    return rho
def check_convergence_conditions(A, verbose=False):
    A = np.array(A, dtype=float)
    results = {}
    
    try:
        results['dds'] = is_strictly_diagonally_dominant(A)
    except:
        results['dds'] = False
    
    try:
        results['spd'] = is_symmetric_positive_definite(A)
    except:
        results['spd'] = False
    
    results['norm_1'] = induced_matrix_norm(A, norm_type=1)
    results['norm_2'] = induced_matrix_norm(A, norm_type=2)
    results['norm_inf'] = induced_matrix_norm(A, norm_type=np.inf)
    results['spectral_radius'] = spectral_radius(A)
    results['converges'] = results['spectral_radius'] < 1
    
    if verbose:
        print("=" * 70)
        print("CONVERGENCE CONDITIONS ANALYSIS")
        print("=" * 70)
        print(f"1. Strictly Diagonally Dominant (DDS):     {results['dds']}")
        print(f"   (SUFFICIENT condition)")
        print(f"\n2. Symmetric and Positive Definite (SPD):  {results['spd']}")
        print(f"   (SUFFICIENT condition)")
        print(f"\n3. Induced Matrix Norms:")
        print(f"   ||A||₁ (column norm):    {results['norm_1']:.6f}  (< 1? {results['norm_1'] < 1})")
        print(f"   ||A||₂ (spectral norm):  {results['norm_2']:.6f}  (< 1? {results['norm_2'] < 1})")
        print(f"   ||A||∞ (row norm):       {results['norm_inf']:.6f}  (< 1? {results['norm_inf'] < 1})")
        print(f"   (SUFFICIENT condition if < 1)")
        print(f"\n4. Spectral Radius:                        {results['spectral_radius']:.6f}")
        print(f"   (NECESSARY & SUFFICIENT condition: < 1? {results['converges']})")
        print("=" * 70)
        print(f"CONVERGENCE GUARANTEED: {results['converges']}")
        print("=" * 70)
    
    return results


__all__ = [
    "solve_iteratif",
    "gauss_seidel",
    "gaussian_elimination_partial_pivot",
    "gaussian_elimination_total_pivot",
    "lu_decomposition",
    "solve_lu",
    "cholesky_decomposition",
    "solve_cholesky",
    "is_strictly_diagonally_dominant",
    "is_symmetric_positive_definite",
    "induced_matrix_norm",
    "spectral_radius",
    "check_convergence_conditions",
]
