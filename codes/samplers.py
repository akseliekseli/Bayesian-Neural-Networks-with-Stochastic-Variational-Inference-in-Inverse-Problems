# ===============================================================
# Created by:
# Felipe Uribe
# ===============================================================
# Version 02.24
#================================================================
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal



#=========================================================================
#=========================================================================
#=========================================================================
def RTO_MH(Nc, Nb, Nt, theta_0, q_prop):
    """
    randomnize-then-optimize Metropolis-Hastings (RTO-MH): 
    * Bardsley et al. (2014): Randomize-then-Optimize: A Method for Sampling 
      from Posterior Distributions in Nonlinear Inverse Problems.
    """
    K = int((Nt*Nc) + Nb)   # total number of iterations

    # parameter dimension
    if hasattr(theta_0, "__len__"):
        d = len(theta_0)
    else:
        d = 1

    # allocation
    theta = np.empty((d, Nc))
    theta_prop = np.empty((d, Nc))
    logq_RTO_eval = np.empty(Nc)
    acc = np.zeros(Nc, dtype=int)
    nit = np.zeros(Nc, dtype=int)

    # initial state
    theta_k = theta_0
    logq_RTO_k, _, _ = q_prop(theta_0, 'eval')
    acc_k = 1

    # run RTO MCMC
    j = 0
    for k in range(K):
        # propose a state using RTO
        theta_star, nres, it = q_prop(theta_k, 'sample')
        logq_RTO_star, _, _ = q_prop(theta_star, 'eval')

        # ratio and acceptance probability
        log_prop_ratio = logq_RTO_k - logq_RTO_star
        log_alpha = min(0, log_prop_ratio)

        # accept/reject
        log_u = np.log(np.random.rand())
        if (log_u <= log_alpha) and (nres < 1e-8) and (np.isnan(logq_RTO_star) == False):
            theta_k = theta_star
            logq_RTO_k = logq_RTO_star
            acc_k = 1
        else:
            acc_k = 0
        nit_k = it

        # store after burn-in
        if (k >= Nb):
            # store after thinning
            if (np.mod(k, Nt) == 0):
                theta[:, j] = theta_k
                theta_prop[:, j] = theta_star
                logq_RTO_eval[j] = logq_RTO_k
                acc[j] = acc_k
                nit[j] = nit_k
                j += 1
                if (np.mod(j, 100) == 0):
                    print(f"\t Sampling at {j}/{Nc}", end='\r')
        elif (k == 0):
            print("\nBurn-in/Warm-up and adapting proposal... {:d} samples\n".format(Nb))

    return theta, theta_prop, logq_RTO_eval, np.mean(acc), nit



# =========================================================================
# =====adaptive Metropolis=================================================
# =========================================================================
def aRWM(d, Nc, Nb, Nt, theta_0, log_target, eps=1e-6, Na=100):
    """
    adaptive random walk Metropolis (aRWM): 
    * assumes Gaussian proposal
    * uses the adaptive algorithm from Haario et al. (2001)
    * only adapts during the burn-in/warm-up period
    * use for low-dimensional inference problems
    """
    K = int((Nt*Nc) + Nb)   # total number of iterations

    # proposal
    q_sample = lambda x, C: multivariate_normal.rvs(mean=x, cov=C)

    # initial proposal scale
    beta = (2.38**2)/d
    Id = np.eye(d)
    C_kp1 = beta*Id
    beta_eps_Id = beta*eps*Id

    # allocation
    theta = np.empty((Nc, d))
    theta_warm = np.empty((K, d))
    log_target_eval = np.empty(Nc)
    acc = np.zeros(Nc, dtype=int)

    # initial state
    theta_k = theta_0
    log_target_eval_k = log_target(theta_0)
    acc_k = 1

    # store
    theta[0, :] = theta_k
    log_target_eval[0] = log_target_eval_k
    acc[0] = acc_k

    # Metropolis algorithm with adaptation
    j = 0
    for k in range(K):
        # propose state and evaluate logtarget and logproposal
        theta_star = q_sample(theta_k, C_kp1)
        log_target_star = log_target(theta_star)

        # log acceptance probability
        target_ratio = log_target_star - log_target_eval_k
        log_alpha = min(0, target_ratio)

        # accept/reject
        log_u = np.log(np.random.rand())
        if (log_u <= log_alpha) and (np.isnan(log_target_star) == False):
            theta_k = theta_star
            log_target_eval_k = log_target_star
            acc_k = 1
        else:
            acc_k = 0

        # adaptation only within the burn-in/warm-up
        theta_warm[k, :] = theta_k
        if (k >= Nb):
            C_kp1 = beta*np.cov(theta_warm[:k, :].T) + beta_eps_Id

        # store after burn-in
        if (k >= Nb):
            # store after thinning
            if (np.mod(k, Nt) == 0):
                theta[j, :] = theta_k
                log_target_eval[j] = log_target_eval_k
                acc[j] = acc_k
                j += 1
                if (np.mod(j, 100) == 0):
                    print(f"\t Sampling at iteration {j}/{Nc}", end='\r')
        elif (k == 0):
            print("\nBurn-in/Warm-up and adapting proposal... {:d} samples\n".format(Nb))
    #
    print("\n-Acceptance rate:", np.mean(acc))

    return theta, log_target_eval, acc



# =========================================================================
# =====adaptive Metropolis 2=================================================
# =========================================================================
def RWM(d, Nc, Nb, Nt, xi_0, logtarget):
    """
    random walk Metropolis (RWM) with vanishing adaptation
    """
    K = int((Nt*Nc) + Nb)   # total number of iterations
    # d                     # parameter dimension
    # df                    # random field discretization

    # allocation
    xi = np.empty((d, Nc))   # uncertain parameters
    logtarget_eval = np.empty(Nc)   # store target evaluations
    acc = np.zeros(Nc, dtype=int)   # store acceptance rate
    betas = np.empty(Nc)

    # initial state
    xi[:, 0] = xi_0
    logtarget_eval[0] = logtarget(xi_0)
    acc[0] = 1
    #
    xi_kp1 = xi_0
    logtarget_eval_kp1 = logtarget_eval[0]
    acc_kp1 = acc[0]

    # initial proposal scaling parameters
    C, a = 1, 0.5
    acc_star = 0.234
    beta = 0.6

    # RWM
    j = 0
    for k in range(K+Nt):
        # propose state and evaluate loglike
        xi_star = xi_kp1 + beta*np.random.randn(d)
        logtarget_star = logtarget(xi_star)

        # log acceptance probability
        log_alpha = min(0, logtarget_star - logtarget_eval_kp1)  # proposal is symmetric

        # accept/reject
        log_u = np.log(np.random.rand())
        if (log_u <= log_alpha) and (np.isnan(logtarget_star) == False):
            xi_kp1 = xi_star
            logtarget_eval_kp1 = logtarget_star
            acc_kp1 = 1
        else:
            acc_kp1 = 0

        # msg
        if (k > Nb):
            # thinning
            if (np.mod(k, Nt) == 0):
                xi[:, j] = xi_kp1
                logtarget_eval[j] = logtarget_eval_kp1
                acc[j] = acc_kp1
                betas[j] = beta
                j += 1
                if (np.mod(j, 500) == 0):
                    print("\nSample {:d}/{:d}".format(j, Nc))
        else:
            if (k == 0):
                print("\nBurn-in... {:d} samples\n".format(Nb))
                # print('\t relerr so far', e_x[k+1])

        # adapt prop spread using info of past acc probability
        gamma_i = C/((k+1)**a)
        beta = np.exp(np.log(beta) + gamma_i*(np.exp(log_alpha) - acc_star))
 
    # apply burn-in
    print("\n-Acceptance rate:", np.mean(acc))

    return xi, logtarget_eval, betas



# ===================================================================
# =====preconditioned Crank-Nicolson=================================
# ===================================================================
def pCN(d, Nc, Nb, Nt, theta_0, prior_rvs, log_like, acc_star=0.44):
    """
    preconditioned Crank-Nicolson (pCN): 
    * assumes reference measure (e.g., prior) is Gaussian
    * uses the vanishing adaptation algorithm from Andrieu and Thoms (2008) (Alg. 4)
        to define a vanishing scaling so the effect of the adaptation dissapears
        in the long run. Here we don't update the covariance of the proposal, as in 
        the Alg. 4, since this is defined by the prior covariance. Hence, we only
        update a global adaptation parameter.
    * use for high-dimensional inference problems
    """
    K = int((Nt*Nc) + Nb)   # total number of iterations

    # initial scale parameters: in pCN beta \in (0, 1]
    beta = (2.38**2)/d
    if beta > 1:
        beta = 0.6
    else:
        beta = beta

    # allocation
    theta = np.empty((Nc, d))
    log_like_eval = np.empty(Nc)
    acc = np.zeros(Nc, dtype=int)
    betas = np.zeros(Nc)

    # initial state
    theta_k = theta_0
    log_like_eval_k = log_like(theta_0)
    acc_k = 1

    # store
    theta[0, :] = theta_k
    log_like_eval[0] = log_like_eval_k
    acc[0] = acc_k

    # pCN
    j = 0
    for k in range(K):
        # propose state and evaluate loglike
        xi = prior_rvs(1).flatten()   # sample from the prior
        theta_star = np.sqrt(1 - beta**2)*theta_k + beta*xi
        loglike_star = log_like(theta_star)

        # log acceptance probability: in pCN this only depends on the like
        log_alpha = min(0, loglike_star - log_like_eval_k)

        # accept/reject
        log_u = np.log(np.random.rand())
        if (log_u <= log_alpha) and (np.isnan(loglike_star) == False):
            theta_k = theta_star
            log_like_eval_k = loglike_star
            acc_k = 1
        else:
            acc_k = 0

        # adaptation of global scale
        beta = vanishing_adapt(np.exp(log_alpha), k+1, beta, acc_star)

        # store after burn-in
        if (k >= Nb):
            # store after thinning
            if (np.mod(k, Nt) == 0):
                theta[j, :] = theta_k
                log_like_eval[j] = log_like_eval_k
                acc[j] = acc_k
                betas[j] = beta
                j += 1
                if (np.mod(j, 100) == 0):
                    print(f"\t Sampling at iteration {j}/{Nc}", end='\r')
        elif (k == 0):
            print("\nBurn-in/Warm-up... {:d} samples\n".format(Nb))
    #
    print("\n-Acceptance rate:", np.mean(acc))

    return theta, log_like_eval, acc#, betas
# =========================================================================
def vanishing_adapt(alpha, k, lambd, acc_star, C=1.0, a=0.5):
    # Lipschitz constant C > 0 and a \in (0, 1]
    gamma_i = C / (k**a)   # deterministic stepsizes: ensures that the variations vanish
    beta = sp.special.expit(sp.special.logit(lambd) + gamma_i*(alpha - acc_star))
    # beta = np.exp(np.log(lambd) + gamma_i*(alpha - acc_star))
    return beta



# ===================================================================
# =====truncated unadjusted Langevin algorithm=======================
# ===================================================================
def ULA(d, Nc, Nb, Nt, theta_0, log_target, beta_k=None, eps2=1e-6):
    """
    unadjusted Langevin algorithm (ULA): 
    * approximated samples from the target
    * uses the truncated version that modifies the drift term to make it stable
        as in Atchade (2006)
    * acceptance rate is 100%
    * here the user should pass a scale beta_k since we cannot adapt it, otherwise
        we use the default scale value
    """
    K = int((Nt*Nc) + Nb)   # total number of iterations

    # initial scale parameters
    if (beta_k == None):
        beta_k = (2.38**2)/d
    Lambda_k = eps2*np.eye(d)

    # allocation
    theta = np.empty((Nc, d))
    log_target_eval = np.empty(Nc)
    acc = np.ones(Nc, dtype=int)

    # initial state
    theta_k = theta_0
    log_target_eval_k, grad_log_target_k = log_target(theta_0)
    D_k = drift(grad_log_target_k)  # truncation

    # store
    theta[0, :] = theta_k
    log_target_eval[0] = log_target_eval_k

    # T-ULA
    j = 0
    for k in range(K):
        # approximate Langevin diffusion
        theta_star = q_langevin_rvs(theta_k, D_k, beta_k, Lambda_k)
        log_target_star, grad_log_target_star = log_target(theta_star)
        D_star = drift(grad_log_target_star)
        
        # compute the acc prob
        log_alpha = (log_target_star - log_target_eval_k) + \
                (q_langevin_logpdf(theta_star, theta_k) - q_langevin_logpdf(theta_k, theta_star))
                
        # accept/reject
        log_u = np.log(np.random.rand())
        if (log_u <= log_alpha) and (np.isnan(log_target_star) == False):
            theta_k = theta_star
            log_target_eval_k = log_target_star.copy()
            acc_k = 1
        else:
            acc_k = 0

        # accept without Metropolis-Hastings correction
        theta_k = theta_star.copy()
        log_target_eval_k = log_target_star.copy()
        D_k = D_star.copy()

        # store after burn-in
        if (k >= Nb):
            # store after thinning
            if (np.mod(k, Nt) == 0):
                theta[j, :] = theta_k
                log_target_eval[j] = log_target_eval_k
                j += 1
                if (np.mod(j, 100) == 0):
                    print(f"\t Sampling at iteration {j}/{Nc}", end='\r')
        elif (k == 0):
            print("\nBurn-in/Warm-up... {:d} samples\n".format(Nb))
    #
    print("\nAcceptance rate:", np.mean(acc), "\n")

    return theta, log_target_eval, acc
# ===================================================================
def drift(grad, delta=1000):
    return delta*grad / max(delta, np.linalg.norm(grad)) 
# ===================================================================
def q_langevin_rvs(theta_k, D_k, beta, Lambda_k):
    mu = theta_k + ((beta**2)/2)*(Lambda_k @ D_k)
    cov = (beta**2)*Lambda_k
    return multivariate_normal.rvs(mean=mu, cov=cov)
def q_langevin_logpdf(theta_k, D_k, beta, Lambda_k):
    mu = theta_k + ((beta**2)/2)*(Lambda_k @ D_k)
    cov = (beta**2)*Lambda_k
    return multivariate_normal.logpdf(mean=mu, cov=cov)