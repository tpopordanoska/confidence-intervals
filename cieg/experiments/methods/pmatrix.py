import torch


def pmatrix_bounds(eigvals_lower, eigvals_upper, eigvects_lower, eigvects_upper, sigma, eig):
    inv_eigvals_lower, inv_eigvals_upper = get_inverse_eigenvalue_bounds(eigvals_lower, eigvals_upper)
    p = eig.eigenvalues.size()[0]
    lower_bound, upper_bound = get_pmatrix_bounds(sigma,
                                                  eigvects_lower,
                                                  eigvects_upper,
                                                  inv_eigvals_upper,
                                                  inv_eigvals_lower,
                                                  p)

    # prec_emp = torch.inverse(sigma)
    # print_pmatrix_bounds(lower_bound, upper_bound, prec_emp)
    # check_pmatrix_bounds(lower_bound, upper_bound, prec_emp)

    non_zero_precision = (lower_bound > 0) + (upper_bound < 0) > 0

    return lower_bound, upper_bound, non_zero_precision


def get_inverse_eigenvalue_bounds(lambdas_lower, lambdas_upper):
    inv_lambdas_lower = 1.0 / lambdas_upper
    inv_lambdas_upper = 1.0 / lambdas_lower

    return inv_lambdas_lower, inv_lambdas_upper


def get_pmatrix_bounds(sigma, v_lower, v_upper, inv_lambdas_upper, inv_lambdas_lower, p):
    # initialize to zero
    lower_bound = torch.zeros_like(sigma)
    upper_bound = torch.zeros_like(sigma)
    # Now reconstruct the bounds on the precision matrix using interval arithmetic over the matrix multiplication of
    # V*Sig^{-1}*V^T. Doing it naively for now, vectorize it later
    for i in range(p):
        for j in range(p):
            for k in range(p):
                updated_this_round = False
                low_low = v_lower[i, k] * inv_lambdas_lower[k] * v_lower[j, k]
                up_up = v_upper[i, k] * inv_lambdas_upper[k] * v_upper[j, k]
                low_up = v_lower[i, k] * inv_lambdas_upper[k] * v_upper[j, k]
                up_low = v_upper[i, k] * inv_lambdas_lower[k] * v_lower[j, k]

                # CHECK IF ANY OF THE LOWER OR UPPER BOUNDS ARE ZERO!  HAVEN'T IMPLEMENTED THESE CASES
                if (v_lower[i, k] == 0) + (v_upper[i, k] == 0) + (v_lower[j, k] == 0) + (v_upper[j, k] == 0):
                    # lower_bound[i, j] += low_low
                    # upper_bound[i, j] += up_up
                    # updated_this_round = True
                    print("oh no, some value is zero")
                    print(v_lower[i, k])
                    print(v_upper[i, k])
                    print(v_lower[j, k])
                    print(v_upper[j, k])
                    continue
                    # assert False  # die here, fix code for these cases rather than return false result

                # case 1a: i lower is positive; j lower is positive
                if (v_lower[i, k] > 0) * (v_lower[j, k] > 0):
                    lower_bound[i, j] += low_low
                    upper_bound[i, j] += up_up
                    updated_this_round = True
                # case 2a: i lower is negative, upper is positive; j lower is positive
                if (v_lower[i, k] < 0) * (v_upper[i, k] > 0) * (v_lower[j, k] > 0):
                    lower_bound[i, j] += low_up
                    upper_bound[i, j] += up_up
                    updated_this_round = True
                # case 3a: i upper is negative; j lower is positive
                if (v_upper[i, k] < 0) * (v_lower[j, k] > 0):
                    lower_bound[i, j] += low_up
                    upper_bound[i, j] += up_low
                    updated_this_round = True

                # case 1b: i lower and upper are positive; j lower is negative, upper is positive
                if (v_lower[i, k] > 0) * (v_lower[j, k] < 0) * (v_upper[j, k] > 0):
                    lower_bound[i, j] += up_low
                    upper_bound[i, j] += up_up
                    updated_this_round = True
                # case 2b: i lower is negative, upper is positive; j lower is negative, upper is positive
                # this is the complicated one where there are a couple possibilities
                if (v_lower[i, k] < 0) * (v_upper[i, k] > 0) * (v_lower[j, k] < 0) * (v_upper[j, k] > 0):
                    # Lower bound will be negative (unless i==j), and there are two possibilities for this
                    tmp = v_lower[i, k] * inv_lambdas_upper[k] * v_upper[j, k]
                    if tmp > up_low:
                        tmp = up_low
                    if i == j:
                        # in this case, the minimum is actually zero as we are on the diagonal
                        if tmp < 0:  # this condition should always hold, but just being explicit
                            tmp = 0
                    lower_bound[i, j] += tmp
                    # Upper bound will be positive, and there are two possibilities for this
                    tmp = v_lower[i, k] * inv_lambdas_upper[k] * v_lower[j, k]
                    if tmp < v_upper[i, k] * inv_lambdas_upper[k] * v_upper[j, k]:
                        tmp = v_upper[i, k] * inv_lambdas_upper[k] * v_upper[j, k]
                    upper_bound[i, j] += tmp
                    updated_this_round = True
                # case 3b: i lower and upper are negative; j lower is negative, upper is positive
                if (v_upper[i, k] < 0) * (v_lower[j, k] < 0) * (v_upper[j, k] > 0):
                    lower_bound[i, j] += low_up
                    upper_bound[i, j] += low_low
                    updated_this_round = True

                # case 1c: i lower and upper are positive; j lower and upper are negative
                if (v_lower[i, k] > 0) * (v_upper[j, k] < 0):
                    lower_bound[i, j] += up_low
                    upper_bound[i, j] += low_up
                    updated_this_round = True
                # case 2c: i lower is negative, upper is positive; j lower and upper are negative
                if (v_lower[i, k] < 0) * (v_upper[i, k] > 0) * (v_upper[j, k] < 0):
                    lower_bound[i, j] += up_low
                    upper_bound[i, j] += low_low
                    updated_this_round = True
                # case 3c: i lower and upper are negative; j lower and upper are negative
                if (v_upper[i, k] < 0) * (v_upper[j, k] < 0):
                    lower_bound[i, j] += up_up
                    upper_bound[i, j] += low_low
                    updated_this_round = True

                if not updated_this_round:
                    print(i)
                    print(j)
                    print(k)
                    print(v_lower[i, k])
                    print(v_upper[i, k])
                    print(v_lower[j, k])
                    print(v_upper[j, k])
                    print(inv_lambdas_lower[k])
                    print(inv_lambdas_upper[k])
                    assert False  # should have hit at least one of the cases!

    return lower_bound, upper_bound


def print_pmatrix_bounds(lower, upper, prec_emp):
    print(f"Prec. matrix lower: {lower}")
    print(f"Prec. matrix:       {prec_emp}")
    print(f"Prec. matrix upper: {upper}")


def check_pmatrix_bounds(lower, upper, prec_emp):
    print("Verifying bounds")
    # upper bounds and lower bounds should bound the empirical precision matrix
    assert torch.min(upper - prec_emp) >= 0
    assert torch.min(prec_emp - lower) >= 0
