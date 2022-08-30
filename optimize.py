import numpy as np


def _fit_newton(score, start_params, score_args=None, tol=1e-8, disp=True,
                maxiter=100, callback=None, retall=False,
                full_output=True, hess=None, hess_args=None, ridge_factor=1e-10):
    """
    Fit using Newton-Raphson algorithm.
    Based on: https://www.statsmodels.org/stable/_modules/statsmodels/base/optimizer.html#_fit_newton

    Parameters
    ----------
    f : function
        Returns negative log likelihood given parameters.
    score : function
        Returns gradient of negative log likelihood with respect to params.
    start_params : array_like, optional
        Initial guess of the solution for the loglikelihood maximization.
        The default is an array of zeros.
    fargs : tuple
        Extra arguments passed to the objective function, i.e.
        objective(x,*args)
    tol : float
        Error tolerance.
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        The maximum number of iterations to perform.
    callback : callable callback(xk)
        Called after each iteration, as callback(xk), where xk is the
        current parameter vector.
    retall : bool
        Set to True to return list of solutions at each iteration.
        Available in Results object's mle_retvals attribute.
    full_output : bool
        Set to True to have all available output in the Results object's
        mle_retvals attribute. The output is dependent on the solver.
        See LikelihoodModelResults notes section for more information.
    hess : str, optional
        Method for computing the Hessian matrix, if applicable.
    ridge_factor : float
        Regularization factor for Hessian matrix.

    Returns
    -------
    xopt : ndarray
        The solution to the objective function
    retvals : dict, None
        If `full_output` is True then this is a dictionary which holds
        information returned from the solver used. If it is False, this is
        None.
    """
    iterations = 0
    oldparams = np.inf
    newparams = np.asarray(start_params)
    if retall:
        history = [oldparams, newparams]
    while (iterations < maxiter and np.any(np.abs(newparams -
            oldparams) > tol)):
        H = np.asarray(hess(newparams, *hess_args))
        # regularize Hessian, not clear what ridge factor should be
        # keyword option with absolute default 1e-10, see #1847
        if not np.all(ridge_factor == 0):
            H[np.diag_indices(H.shape[0])] += ridge_factor
        oldparams = newparams
        newparams = oldparams - np.dot(np.linalg.inv(H),
                score(oldparams, *score_args))
        if retall:
            history.append(newparams)
        if callback is not None:
            callback(newparams)
        iterations += 1
    if iterations == maxiter:
        warnflag = 1
        if disp:
            print("Warning: Maximum number of iterations has been "
                   "exceeded.")
            print("         Iterations: %d" % iterations)
    else:
        warnflag = 0
        if disp:
            print("Optimization terminated successfully.")
            print("         Iterations %d" % iterations)
    if full_output:
        (xopt, niter,
         gopt, hopt) = (newparams, iterations, score(newparams, *score_args),
                        hess(newparams, *hess_args))
        converged = not warnflag
        retvals = {'iterations': niter, 'score': gopt,
                   'Hessian': hopt, 'warnflag': warnflag,
                   'converged': converged}
        if retall:
            retvals.update({'allvecs': history})

    else:
        xopt = newparams
        retvals = None

    return xopt, retvals




