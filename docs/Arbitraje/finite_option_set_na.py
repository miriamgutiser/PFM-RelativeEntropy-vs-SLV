import numpy as np


def finite_option_set_na(factor, forward, strikes, call_prices):
    """
    Decide whether a finite set of European call quotes is arbitrage-free.

    Parameters
    ----------
    factor : float
        Gross compounding factor between t=0 and maturity T
        (e.g. e^{rT} or 1 + rT). 1/factor is the zero-coupon bond price.
    forward : float
        Forward price of the underlying for delivery at T.
    strikes : array-like
        Strikes K_1, ..., K_n of the n quoted European calls.
    call_prices : array-like
        The corresponding (undiscounted) call prices C(K_1), ..., C(K_n).

    Returns
    -------
    bool
        True  -> the quote set admits at least one arbitrage-free model.
        False -> at least one static arbitrage is present.
    """
    # 1. Sort strikes ascending; align call prices descending.
    K = np.sort(np.asarray(strikes, dtype=float).ravel())              # K_1 < ... < K_n
    C = np.sort(np.asarray(call_prices, dtype=float).ravel())[::-1]    # C_1 > ... > C_n
    n = K.size

    # 2. Candidate state nodes for S_T : 0 and the n strikes.
    L = np.concatenate(([0.0], K))                                     # length n+1

    # 3. Payoff block of size n x (n+1): entries max(K_i - L_j, 0).
    K_mat = np.tile(K.reshape(-1, 1), (1, n + 1))
    L_mat = np.tile(L.reshape(1, -1), (n, 1))
    payoff_block = np.maximum(K_mat - L_mat, 0.0)

    # 4. Assemble the (n+2) x (n+2) "instrument vs. state" matrix A.
    A = np.zeros((n + 2, n + 2))
    A[1:n + 1, 1:n + 2] = payoff_block          # central block (calls vs. states)
    A[0:n + 1, 0]       = 1.0                   # first column  (bond column)
    A[n + 1, 1:n + 2]   = 1.0                   # last row      ("tail" state)

    # 5. Price vector V (row vector in MATLAB).
    V = np.concatenate(([1.0 / factor, forward / factor], C))          # length n+2

    # 6. Solve  x @ A = V  for implied state weights x.
    #    MATLAB's  V/A  is  V * inv(A); equivalently  A.T @ x = V.
    try:
        x = np.linalg.solve(A.T, V)
    except np.linalg.LinAlgError:
        return False                            # degenerate / collinear quotes

    # 7. No-arbitrage  <=>  all state weights are strictly positive (FTAP).
    return bool(np.all(x > 0))