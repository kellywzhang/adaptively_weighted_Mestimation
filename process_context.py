import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
from scipy.linalg import sqrtm
import os
import argparse
import json
import time
import pickle
from scipy.stats import chi2
from itertools import combinations
from scipy.special import expit, logit

from optimize import _fit_newton

parser = argparse.ArgumentParser()
parser.add_argument('--strategy', type=str, default='TS', \
        choices=['TS', 'independent'])
parser.add_argument('--N', type=int, default=100000,
        help='Number of monte carlo simulations')
parser.add_argument('--T', type=int, default=5,
        help='Number of batches')
parser.add_argument('--d', type=int, default=3,
        help='Number of dimension')
parser.add_argument('--params', type=str, default='1,1,1,0,0,0',
        help='Expected rewards for each arm')
parser.add_argument('--var', type=float, default=1.0,
        help='Reward variance (sigma^2)')
parser.add_argument('--clipping', type=float, default=0.0,
        help='Clipping value in [0, 1)')
parser.add_argument('--reward', type=str, default='normal', \
        choices=['bernoulli', 'normal', 'poisson', 'uniform', 't-dist'])
parser.add_argument('--no_zeros', type=int, default=0)
parser.add_argument('--Z_dist', type=str, default='uniform',
         choices=['normal', 'uniform', 'bernoulli', 'ones'] )

# non-save arguments
parser.add_argument('--path', type=str, default='./simulations_context',
        help='Where to save results' )
parser.add_argument('--load_results', type=int, default=0,
        help='Only load results from a previous run of process.py')
parser.add_argument('--verbose', type=int, default=0,
        help='Prints more details')
parser.add_argument('--estvar', type=int, default=1,
        help='Estimate the variance')
parser.add_argument('--Wdecor', type=int, default=0,
        help='Use W decorrelated estimator')
parser.add_argument('--snmb', type=int, default=1,
        help='Self-Normalized Martingale bound:' + \
                'https://papers.nips.cc/paper/2011/file/e1d5be1c7f2f456670de3d53c7b54f4a-Paper.pdf')
parser.add_argument('--sparseT', type=str, default=None,
        help='Evaluate estimators not at every batch')

args = parser.parse_args()
print( vars(args) )

plt.rcParams.update({'font.size': 20})
path = args.path
nonsave_args = ['path', 'load_results', 'estvar', \
        'verbose', 'sparseT', 'mle', 'least_squares', 'Wmle', 'Wdecor', 'snmb']
if not args.no_zeros:
    nonsave_args.append( 'no_zeros' )
save_args = [ '{}={}'.format(key, val) for key, val in vars(args).items() if key not in nonsave_args ]
save_str = '_'.join( save_args )
save_f_load = os.path.join( path, save_str)

save_f = os.path.join(path, save_str)
if not os.path.isdir( save_f ):
    os.mkdir( save_f )

alphas = [ 0.1 ]
if args.sparseT is None:
    Tvals = [t for t in range(1, args.T+1)]
else:
    Tvals = [ int(x) for x in args.sparseT.split(",") ]

params_str = args.params.replace('n', '-', args.d*2)
params = np.array( [float(x) for x in params_str.split(',')] )
print("Params", params)



###########
# Methods #
###########


def confidence_region(est_param, true_param, invcov, t, results_dict, suffix, df, cutoffs=None):
    # =============================================================
    # Hyper-Ellipsoid ==================================================
    # =============================================================
        # centered at MLE
        # ellipsoid shape described by quadratic equation
    
    dim = len(true_param)
    
    # Chi-Squared Test (prob confidence ellipsoid contains params)
    results_dict['ellipsoid'+suffix] = { a : {} for a in alphas }
    results_dict['ellipsoid{}_vol'.format(suffix)] = { a : {} for a in alphas }
    results_dict['ellipsoid{}_root_vol'.format(suffix)] = { a : {} for a in alphas }

    # Computing semi-axes of ellipsoid
        # np.linalg.eig outputs Q, L, where original matrix is Q L Q.T
    invcov = invcov + np.eye(dim)*1e-6
    eigvals, eigvectors = np.linalg.eig( invcov )

    tmp = np.einsum( 'ijk,ik->ij', invcov, est_param-true_param )
    chi_stat = np.einsum('ij,ij->i', tmp, est_param-true_param)
    for i, alpha in enumerate(alphas):
        if cutoffs == None:
            if args.estvar == 0:
                # Chi-squared cutoffs (not estimating noise variance)
                cutoff = scipy.stats.chi2.ppf(1-alpha, df=df)
            else:
                # F distribution cutoffs (since estimating noise variance)
                cutoff_scale = df * (t-1) / (t-df)
                cutoff = cutoff_scale * scipy.stats.f.ppf(1-alpha, dfn=df, dfd=t-df)
        else:
            cutoff = cutoffs[i]
            empirical_cutoff = cutoff

        # Computing coverage probabilities
        contains = chi_stat <= cutoff
        print('ellipsoid'+suffix, alpha, np.mean(contains), np.std(contains)/np.sqrt(args.N), sep=',')
      
        # Volume = pi^{dim/2} / Gamma( dim/2 + 1) \prod 1/sqrt(lambda) 
               # = pi^{dim/2} / (dim/2)! \prod 1/sqrt(lambda) 
        volume = np.pi**(dim/2) / scipy.special.gamma( dim/2 + 1 ) 
        volume = np.product( 1/np.sqrt(eigvals), axis=1) * volume
        volume = volume * np.power( np.sqrt(cutoff), dim )
            # power of dim in above line because in dim dimensional space
            # https://math.stackexchange.com/questions/2431159/how-to-obtain-the-equation-of-the-projection-shadow-of-an-ellipsoid-into-2d-plan

        print("vol", np.max(np.product( 1/np.sqrt(eigvals), axis=1) ) )
        print("median", np.median(volume), "max", np.max(volume))

        root_volume = np.float_power( volume, 1/dim )
        print('ellipsoid{} volume'.format(suffix), alpha, np.mean(volume), np.var(volume), sep=',' )
        
        results_dict['ellipsoid'+suffix][alpha] = ( np.mean(contains), np.std(contains)/np.sqrt(args.N) )
        results_dict['ellipsoid{}_vol'.format(suffix)][alpha] = ( np.mean(volume), np.std(volume)/np.sqrt(args.N) )
        results_dict['ellipsoid{}_root_vol'.format(suffix)][alpha] = ( np.mean(root_volume), np.std(root_volume)/np.sqrt(args.N) )
        
    return None


def snmb_inference(simulation_dict, alphas, CI_dict):
    all_actions = simulation_data[ 'all_actions' ]
    all_rewards = simulation_data[ 'all_rewards' ]
    all_pis = simulation_data[ 'all_pis' ]
    all_Z = simulation_data[ 'all_Z' ]      # N x T x d
    
    CI_dict['snmb'] = { t: {} for t in Tvals }

    lam = 1
    S = 2*args.d

    all_RZA_base = np.expand_dims(all_rewards, 2)*all_Z
    all_RZA_adv = np.expand_dims(all_actions*all_rewards, 2)*all_Z
    all_RZA = np.concatenate( [all_RZA_base, all_RZA_adv], axis=2 )
    
    all_ZA = np.expand_dims(all_actions, 2)*all_Z
    all_Z_vec = np.concatenate( [all_Z, all_ZA], axis=2 )
    all_ZZ = np.einsum('ijk,ijl->ijkl', all_Z_vec, all_Z_vec)
    
    print( '\n\nSelf-Normalized Martingale Bound' )
    for t in Tvals:
        print("")
        print(t)
       
        # L2 regularized least squares

        # Estimator ##############################
        design = np.sum(all_ZZ[:,:t], axis=1) + lam*np.eye(2*args.d)
        l2ls_est = np.einsum('ijk,ik->ij', np.linalg.inv(design), np.sum(all_RZA[:,:t], axis=1) )
        invcov = design

        # Compute projected inverse covariance matrix for advantage
        invcov_12 = invcov[:,:args.d,args.d:]
        invcov_11_inv = np.linalg.inv( invcov[:,:args.d,:args.d] )
        invcov_tmp = np.einsum('ijk,ikl->ijl', invcov_12, np.einsum('ijk,ikl->ijl', invcov_11_inv, invcov_12))
        invcov_adv = invcov[:,args.d:,args.d:] - invcov_tmp
    
        # Cutoffs #################################
        cutoffs = []
        for alpha in alphas:
            # det(lam*I) = lam**d
            sqrtDetLamI = np.sqrt( np.power(lam, 2*args.d) )
            sqrtDetVt = np.sqrt( np.linalg.det(design) )
            
            # Theorem 2 of https://papers.nips.cc/paper/2011/file/e1d5be1c7f2f456670de3d53c7b54f4a-Paper.pdf 
            cutoff_pt1 = np.sqrt( 2 * np.log( sqrtDetVt / ( sqrtDetLamI * alpha ) ) ) 
            cutoff_pt2 = np.sqrt( lam ) * S
            cutoff = np.square( cutoff_pt1 + cutoff_pt2 )
            
            cutoffs.append(cutoff)
       
        # Confidence Region #######################
        mse_vec = np.mean(np.square(l2ls_est-params), axis=1)
        print("MSE", np.mean(mse_vec), np.std(mse_vec) / np.sqrt(args.N) )
       
        confidence_region(est_param=l2ls_est, true_param=params, \
                invcov=invcov, t=t, results_dict=CI_dict['snmb'][t], \
                cutoffs=cutoffs, df=2*args.d, suffix='')
    
        confidence_region(est_param=l2ls_est[:,args.d:], true_param=params[args.d:], \
                invcov=invcov_adv, t=t, results_dict=CI_dict['snmb'][t], \
                cutoffs=cutoffs, df=2*args.d, suffix='_adv')
   
    return None 
        

def Wdecor_inference(simulation_data, alphas, CI_dict):
    all_actions = simulation_data[ 'all_actions' ]
    all_rewards = simulation_data[ 'all_rewards' ]
    all_pis = simulation_data[ 'all_pis' ]
    all_Z = simulation_data[ 'all_Z' ]      # N x T x d
    
    CI_dict['Wdecor'] = { t: {} for t in Tvals }

    all_RZA_base = np.expand_dims(all_rewards, 2)*all_Z
    all_RZA_adv = np.expand_dims(all_actions*all_rewards, 2)*all_Z
    all_RZA = np.concatenate( [all_RZA_base, all_RZA_adv], axis=2 )
    
    all_ZA = np.expand_dims(all_actions, 2)*all_Z
    all_Z_vec = np.concatenate( [all_Z, all_ZA], axis=2 )
    all_ZZ = np.einsum('ijk,ijl->ijkl', all_Z_vec, all_Z_vec)
    
    print( '\n\nW-Decorrelated' )
    print('Computing weights...')
    
    # W weights ##############################
    all_W_dict = {}
    Z_vec_norm = np.square( np.linalg.norm(all_Z_vec, axis=2) )
    all_lam = {}
    
    for Ttmp in Tvals:
        eigval, eigvec = np.linalg.eig( np.sum(all_ZZ[:,:Ttmp], axis=1) )
        min_eigval = np.min(eigval, axis=1)
        lam = np.percentile( min_eigval, 0.01 ) / np.log(Ttmp)
        #lam = np.percentile( min_eigval, 0.1 ) / np.log(Ttmp)
        assert lam >= 0
        print(Ttmp, 'lambda', lam)
        all_lam[str(Ttmp)] = lam

        all_Z_vec_normed = all_Z_vec / np.expand_dims(all_lam[str(Ttmp)] + Z_vec_norm, axis=2)
        for t in range(Ttmp):
            if t == 0:
                w_t = all_Z_vec_normed[:,t]
                all_W = np.expand_dims( w_t, axis=1 )
            else:
                WX = np.einsum('ijk,ijl->ikl', all_W, all_Z_vec[:,:t] )
                w_t = np.einsum('ijk,ik->ij', np.eye(args.d*2)-WX, all_Z_vec_normed[:,t])
                all_W = np.concatenate([all_W, np.expand_dims(w_t, axis=1)], axis=1)

        all_W_dict[Ttmp] = all_W

    for t in Tvals:
        print("")
        print(t)
       
        # OLS Estimator ##############################
        design = np.sum(all_ZZ[:,:t], axis=1) + 1e-6*np.eye(2*args.d)
        ols_est = np.einsum('ijk,ik->ij', np.linalg.inv(design), np.sum(all_RZA[:,:t], axis=1))
       
        # W_decor estimator ##############################
        residuals = all_rewards[:,:t] - np.einsum('ijk,ik->ij', all_Z_vec[:,:t], ols_est)
        adjust = np.einsum('ijk,ij->ik', all_W_dict[t], residuals)
        Wdecor_est = ols_est + adjust

        # W_decor variance ###############################
        WW = np.einsum('ijk,ijl->ikl', all_W_dict[t], all_W_dict[t])
        varest = np.mean( np.square(residuals), axis=1 )
        cov = WW * np.expand_dims(varest, axis=(1,2))
        invcov = np.linalg.inv( cov + np.eye(args.d*2)*1e-6 )

        # Compute inverse covariance matrix for advantage
        cov = np.linalg.inv( invcov + np.eye(args.d*2)*1e-6 )
        invcov_adv = np.linalg.inv( cov[:,args.d:,args.d:] )

        # Confidence Region #######################
        mse_vec = np.mean(np.square(Wdecor_est-params), axis=1)
        CI_dict['Wdecor'][t]['MSE'] = ( np.mean(mse_vec), np.std(mse_vec) / np.sqrt(args.N) )

        print("error", np.mean(mse_vec), np.std(mse_vec) / np.sqrt(args.N) )
        confidence_region(est_param=Wdecor_est, true_param=params, invcov=invcov, t=t, results_dict=CI_dict['Wdecor'][t],
                suffix='', df=2*args.d)
       
        confidence_region(est_param=Wdecor_est[:,args.d:], true_param=params[args.d:], invcov=invcov_adv, 
                t=t, results_dict=CI_dict['Wdecor'][t], suffix='_adv', df=args.d)

    return None 



def scoref(theta, invlink, RZ_vec, Z_vec, weights):
    """
    theta: param value; dimension 2*args.d
    Z_vec: dimension 2*args.d
    RZ_vec: dimension 2*args.d
    """
    # 0 = Sum R_t Z_t - g^{-1} ( Z_t . theta) Z_t
    natural_param = np.matmul(Z_vec, theta)
    weights_expanded = np.expand_dims(weights, axis=1)
    pt1 = np.sum( weights_expanded * RZ_vec, axis=0 ) 
    pt2 = np.sum( np.expand_dims( weights * invlink(natural_param), axis=1) * Z_vec, axis=0 )
    return pt1 - pt2


def loglikelihood_hessian(theta, deriv_invlink, Z_vec, ZZ_matrix, weights):
    # - Sum g^{-1}' ( Z_t . theta) Z_t Z_t 
    natural_param = np.matmul(Z_vec, theta)
    tmp = np.expand_dims( weights * deriv_invlink(natural_param), axis=(1,2) ) * ZZ_matrix
    return -np.sum(tmp, axis=0)


def mle_inference(simulation_dict, alphas, CI_dict, weight_type=None, least_squares=False):
    all_actions = simulation_data[ 'all_actions' ]
    all_rewards = simulation_data[ 'all_rewards' ]
    all_pis = simulation_data[ 'all_pis' ]
    all_Z = simulation_data[ 'all_Z' ]      # N x T x d

    # Weighting
    if weight_type == None:
        print( '\nMLE' )
        weights = np.ones(all_pis.shape)
        est_type = 'least_squares' if least_squares else 'MLE'
    elif weight_type == "isrpw": # inverse square-root propensity weighting
        print( '\nMLE '+weight_type )
        weights = 1/np.sqrt(all_pis) * all_actions + 1/np.sqrt(1-all_pis) * (1-all_actions)
        est_type = 'W_least_squares' if least_squares else 'W-MLE'
    else:
        raise ValueError("Invalid weight type")
    
    CI_dict[est_type] = { t: {} for t in Tvals }

    all_RZA_base = np.expand_dims(all_rewards, 2)*all_Z
    all_RZA_adv = np.expand_dims(all_actions*all_rewards, 2)*all_Z
    all_RZA = np.concatenate( [all_RZA_base, all_RZA_adv], axis=2 )
    
    all_ZA = np.expand_dims(all_actions, 2)*all_Z
    all_Z_vec = np.concatenate( [all_Z, all_ZA], axis=2 )
    all_ZZ = np.einsum('ijk,ijl->ijkl', all_Z_vec, all_Z_vec)
    
    for t in Tvals:
        print("")
        print(t)
        
        if args.reward == 'normal' or least_squares:
            print("least squares")
            # Least Squares
            tmp_RZA = all_RZA[:,:t] * np.expand_dims(weights[:,:t], axis=2)
            tmp_ZZ = all_ZZ[:,:t] * np.expand_dims(weights[:,:t], axis=(2,3))

            gram_matrix = np.sum(tmp_ZZ, axis=1) # N x 2*args.d x 2*args.d
            mle = np.einsum('ijk,ik->ij', \
                    np.linalg.inv( gram_matrix + np.eye(args.d*2)*1e-6 ), 
                    np.sum(tmp_RZA, axis=1) )
          
            # Reward residuals
            residuals = all_rewards[:,:t] - np.einsum('ijk,ik->ij', all_Z_vec[:,:t], mle)
            residuals_square = np.square(residuals)
            if args.estvar == 0:
                residuals_square = np.ones(residuals.shape)
            elif args.estvar == 1:
                residuals_square = np.expand_dims( np.mean( np.square(residuals), axis=1 ), axis=1 )

            if weight_type == None:
                # Sandwich
                residual_ZZ = np.expand_dims( residuals_square, axis=(2,3) ) * all_ZZ[:,:t]
                Sigma_hat_inv = np.linalg.inv( np.mean( residual_ZZ, axis=1 ) + np.eye(args.d*2)*1e-6 )  # middle
                bread = 1/np.sqrt(t) * gram_matrix
                
                # bread x middle x bread
                invcov = np.einsum('ijk,ikl->ijl', np.transpose(bread, (0,2,1)), \
                        np.einsum('ijk,ikl->ijl', Sigma_hat_inv, bread) )

                # Compute inverse covariance matrix for advantage
                cov = np.linalg.inv( invcov + np.eye(args.d*2)*1e-6 )
                invcov_adv = np.linalg.inv( cov[:,args.d:,args.d:] )
             

            elif weight_type == 'isrpw':
                #1/all_pis
                ipw_weights = all_actions / all_pis + (1-all_actions) / (1-all_pis)
                mdot_sum = np.mean( all_ZZ[:,:t] * np.expand_dims(residuals_square * ipw_weights[:,:t], axis=(2,3)), axis=1 )
                mdot_sum_inv = np.linalg.inv( mdot_sum + np.eye(args.d*2)*1e-6 )

                bread = 1/np.sqrt(t) * gram_matrix
                invcov = np.einsum('ijk,ikl->ijl', np.transpose(bread, (0,2,1)), \
                        np.einsum('ijk,ikl->ijl', mdot_sum_inv, bread) )

                # Compute projected inverse covariance matrix for advantage
                invcov_12 = invcov[:,:args.d,args.d:]
                invcov_11_inv = np.linalg.inv( invcov[:,:args.d,:args.d] )
                invcov_tmp = np.einsum('ijk,ikl->ijl', np.transpose(invcov_12, (0,2,1)), \
                        np.einsum('ijk,ikl->ijl', invcov_11_inv, invcov_12))
                invcov_proj_adv = invcov[:,args.d:,args.d:] - invcov_tmp

                # Compute inverse covariance matrix for advantage (not projected)
                cov = np.linalg.inv( invcov + np.eye(args.d*2)*1e-6 )
                invcov_adv = np.linalg.inv( cov[:,args.d:,args.d:] )


        elif args.reward == 'bernoulli':
            print("bernoulli")
            # MLE: newton-raphson optimization #################################
            all_mle = []
            all_converged = []
            for i in range(args.N):
                theta0 = np.zeros(args.d*2)
                
                # expit(x) = 1/(1+np.exp(-x)) ) = exp(x) / ( 1 + exp(x) )
                def b_deriv_invlink(x):
                    # exp(x) / ( 1+exp(x) )**2
                    # return np.divide( expit(x), (1 + np.exp(x)) )
                    tmp = np.log( expit(x) ) - np.log( 1 + np.exp(x) )
                    return np.exp(tmp)

                # See page 122 of Agresti
                xopt, retvals = _fit_newton(score=scoref, start_params=theta0, 
                        score_args = (expit, all_RZA[i][:t], all_Z_vec[i][:t], weights[i][:t]),
                        hess=loglikelihood_hessian,
                        hess_args = (b_deriv_invlink, all_Z_vec[i][:t], all_ZZ[i][:t], weights[i][:t]),
                        tol=1e-10, disp=False, maxiter=200, callback=None, retall=False,
                        full_output=True, ridge_factor=1e-10)
                
                all_mle.append(xopt)
                all_converged.append(retvals['converged'])

                if i % 1000 == 0:
                    print(i, xopt)

            mle = np.array(all_mle)
            print("Estimate", np.mean(mle, 0), np.var(mle, 0))
            print("proportion converged", np.mean(all_converged)) 

            print( np.mean( np.square(mle-params) ) )
            #if np.mean( np.square(mle-params) ) > 10:
            #    import ipdb; ipdb.set_trace()

            # Variance ############################################
            if weight_type == None:
                # Sum g^{-1}' ( Z_t . theta) Z_t Z_t 
                natural_param = np.einsum('ijk,ik->ij', all_Z_vec[:,:t], mle)
                tmp = np.expand_dims( b_deriv_invlink(natural_param[:,:t]), axis=(2,3) ) * all_ZZ[:,:t]
                invcov = np.sum(tmp, axis=1) 
                
                # Compute inverse covariance matrix for advantage
                cov = np.linalg.inv( invcov + np.eye(2*args.d)*1e-6 )
                invcov_adv = np.linalg.inv( cov[:,args.d:,args.d:] ) 
           
            elif weight_type == 'isrpw':
                natural_param = np.einsum('ijk,ik->ij', all_Z_vec[:,:t], mle)
                tmp = np.expand_dims( weights[:,:t] * b_deriv_invlink(natural_param[:,:t]), \
                        axis=(2,3) ) * all_ZZ[:,:t]
                bread = 1/np.sqrt(t) * np.sum(tmp, axis=1) 
                
                #1/all_pis
                ipw_weights = all_actions / all_pis + (1-all_actions) / (1-all_pis)
                tmp = np.expand_dims( ipw_weights[:,:t] * b_deriv_invlink(natural_param[:,:t]), \
                        axis=(2,3) ) * all_ZZ[:,:t]
                middle = np.linalg.inv( np.mean(tmp, axis=1) + np.eye(2*args.d)*1e-3 ) 

                invcov = np.einsum('ijk,ikl->ijl', np.transpose(bread, (0,2,1)), \
                        np.einsum('ijk,ikl->ijl', middle, bread) ) 
                
                # Compute projected inverse covariance matrix for advantage
                invcov_12 = invcov[:,:args.d,args.d:] 
                invcov_11_inv = np.linalg.inv( invcov[:,:args.d,:args.d] )
                invcov_tmp = np.einsum('ijk,ikl->ijl', np.transpose(invcov_12, (0,2,1)), \
                        np.einsum('ijk,ikl->ijl', invcov_11_inv, invcov_12))
                invcov_proj_adv = invcov[:,args.d:,args.d:] - invcov_tmp 

                # Compute inverse covariance matrix for advantage (not projected)
                cov = np.linalg.inv( invcov + np.eye(2*args.d)*1e-6 )
                invcov_adv = np.linalg.inv( cov[:,args.d:,args.d:] ) 
               
        elif args.reward == 'poisson':
            print("poisson")
            # MLE: newton-raphson optimization #################################
            all_mle = []
            all_converged = []
            for i in range(args.N):
                theta0 = np.ones(args.d*2)
             
                xopt, retvals = _fit_newton(score=scoref, start_params=theta0, 
                        score_args = (np.exp, all_RZA[i][:t], all_Z_vec[i][:t], weights[i][:t]),
                        hess=loglikelihood_hessian,
                        hess_args = (np.exp, all_Z_vec[i][:t], all_ZZ[i][:t], weights[i][:t]),
                        tol=1e-8, disp=False, maxiter=200, callback=None, retall=False,
                        full_output=True, ridge_factor=1e-6)
             
                all_mle.append(xopt)
                all_converged.append(retvals['converged'])

                if i % 1000 == 0:
                    print(i, xopt)

            mle = np.array(all_mle)
            print("Estimate", np.mean(mle, 0), np.var(mle, 0))
            print("proportion converged", np.mean(all_converged)) 
            
            # Variance ############################################
            if weight_type == None:
                # Sum g^{-1}' ( Z_t . theta) Z_t Z_t 
                natural_param = np.einsum('ijk,ik->ij', all_Z_vec[:,:t], mle)
                tmp = np.expand_dims( np.exp(natural_param[:,:t]), axis=(2,3) ) * all_ZZ[:,:t]
                invcov = np.sum(tmp, axis=1)
                
                # Compute inverse covariance matrix for advantage
                cov = np.linalg.inv( invcov + np.eye(args.d*2)*1e-3 )
                invcov_adv = np.linalg.inv( cov[:,args.d:,args.d:] )
            
            elif weight_type == 'isrpw':
                natural_param = np.einsum('ijk,ik->ij', all_Z_vec[:,:t], mle)
                tmp = np.expand_dims( weights[:,:t] * np.exp(natural_param[:,:t]), \
                        axis=(2,3) ) * all_ZZ[:,:t]
                bread = 1/np.sqrt(t) * np.sum(tmp, axis=1)
                
                #1/all_pis
                ipw_weights = all_actions / all_pis + (1-all_actions) / (1-all_pis)
                tmp = np.expand_dims( ipw_weights[:,:t] * np.exp(natural_param[:,:t]), \
                        axis=(2,3) ) * all_ZZ[:,:t]
                middle = np.linalg.inv( np.mean(tmp, axis=1) + np.eye(2*args.d)*1e-2 )
                invcov = np.einsum('ijk,ikl->ijl', np.transpose(bread, (0,2,1)), \
                        np.einsum('ijk,ikl->ijl', middle, bread) )
                
                # Compute projected inverse covariance matrix for advantage
                invcov_12 = invcov[:,:args.d,args.d:]
                invcov_11_inv = np.linalg.inv( invcov[:,:args.d,:args.d] )
                invcov_tmp = np.einsum('ijk,ikl->ijl', np.transpose(invcov_12, (0,2,1)), \
                        np.einsum('ijk,ikl->ijl', invcov_11_inv, invcov_12))
                invcov_proj_adv = invcov[:,args.d:,args.d:] - invcov_tmp
                
                # Compute inverse covariance matrix for advantage (not projected)
                cov = np.linalg.inv( invcov + np.eye(args.d*2)*1e-6 )
                invcov_adv = np.linalg.inv( cov[:,args.d:,args.d:] )
        
        else:
            raise ValueError('Invalid Reward Type')

        CI_dict[est_type][t]['Estimate'] = ( np.mean(mle, axis=1), np.std(mle, axis=1) / np.sqrt(args.N) )
        mse_vec = np.mean(np.square(mle-params), axis=1)
        CI_dict[est_type][t]['MSE'] = ( np.mean(mse_vec), np.std(mse_vec) / np.sqrt(args.N) )

        # CI for All Parameters ######################################################
        print("error", np.mean(mse_vec), np.std(mse_vec) / np.sqrt(args.N) )
        confidence_region(est_param=mle, true_param=params, invcov=invcov, t=t, results_dict=CI_dict[est_type][t],
                suffix='', df=2*args.d)
      
        # CI for Advantage ###########################################################
        confidence_region(est_param=mle[:,args.d:], true_param=params[args.d:], invcov=invcov_adv, 
                t=t, results_dict=CI_dict[est_type][t], suffix='_adv', df=args.d)
        if weight_type != None:
            confidence_region(est_param=mle[:,args.d:], true_param=params[args.d:], invcov=invcov_proj_adv, 
                    t=t, results_dict=CI_dict[est_type][t], suffix='_adv_proj', df=2*args.d)

    return None




########################
# Load simulation data #
########################
if not args.load_results:
    print('Loading...')
    with open(save_f_load+'/simulation_data.p', 'rb') as fp:
        simulation_data = pickle.load(fp)
    print('Done loading!')

    # Perform inference
    least_squares=True if args.reward in ['uniform', 't-dist'] else False

    CI_dict = {}
    mle_inference(simulation_data, alphas, CI_dict, least_squares=least_squares)
    mle_inference(simulation_data, alphas, CI_dict, weight_type='isrpw', least_squares=least_squares)

    if least_squares:
        if args.Wdecor:
            Wdecor_inference(simulation_data, alphas, CI_dict)
        if args.snmb:
            snmb_inference(simulation_data, alphas, CI_dict)
   
    pickle.dump( CI_dict, open( os.path.join( save_f, 'CI_dict.p'), 'wb' ) )

else:
    CI_dict = pickle.load( open( os.path.join( save_f, 'CI_dict.p'), 'rb' ) )


strategy2name = {
            'TS': 'Thompson Sampling',
            'independent': 'Independently Sampled',
        }
key2color = { 
    'MLE': 'blue',
    'W-MLE': 'red', 
    'W-MLE_proj': 'purple', 
    'snmb': 'green',
    'Wdecor': 'yellow',
    'least_squares': 'blue',
    'W_least_squares': 'red',
    'W_least_squares_proj': 'purple',
    
    'UW': 'blue',
    'AW': 'red',
    'AW_proj': 'purple',
}
#https://davidmathlogic.com/colorblind/#%23648FFF-%23785EF0-%23DC267F-%23FE6100-%23FFB000
key2color = { 
    'MLE': '#648FFF',
    'W-MLE': '#FFB000', 
    'W-MLE_proj': '#FE6100', 
    'snmb': '#785EF0',
    'Wdecor': '#DC267F',
    'least_squares': '#648FFF',
    'W_least_squares': '#FFB000',
    'W_least_squares_proj': '#FE6100',
    
    'UW': '#648FFF',
    'AW': '#FFB000',
    'AW_proj': '#FE6100',
}
key2name = {
    'MLE': 'MLE',
    'W-MLE': 'AW-MLE', 
    'Wdecor': 'W-Decorrelated', 
    'snmb': 'Self-Normalized\nMartingale Bound',
    'least_squares': 'OLS',
    'W_least_squares': 'AW-LS',
    
    'UW': 'OLS / MLE',
    'AW': 'AW-LS / AW-MLE',
    'AW_proj': 'AW-LS / AW-MLE\n(Projected)',
}
order = {
        'MLE': 40,
        'W-MLE': 0, 
        'W-MLE_proj': 20, 
        'snmb': 60,
        'Wdecor': 70,
        'least_squares': 50,
        'W_least_squares': 10,
        'W_least_squares_proj': 30,

        'UW': 3,
        'AW': 1,
        'AW_proj': 2,
        }
reward2model = {
            'bernoulli': 'Logistic Model',
            't-dist': 'Linear Model',
        }

###########################################################################################
###########################################################################################
# Plot MSE

import pandas as pd
from plotnine import ggplot, ggsave, aes, theme, geom_line, geom_point, geom_errorbar, labs, xlim, theme_classic, theme_seaborn, element_text, ylim, element_blank, geom_hline, scale_y_log10, scale_fill_manual, scale_color_manual, scale_linetype_manual, facet_wrap, facet_grid, element_line, element_text

setting2name = { 'allparam': 'All Parameters', 'advantage': 'Advantage Parameters'}
save_summary = '{}_R={}_Z={}_d={}'.format( args.strategy, args.reward, args.Z_dist, args.d )

print("\n\nPlotting MSE...\n")

mse_data = { 
        'Estimator': [],
        'Timesteps': [],
        'MSE': [],
        'mse_se': [],
        'Reward': [],
        }

keys = []
for key, val in CI_dict.items():
    if 'MSE' not in val[Tvals[0]].keys():
        continue
    keys.append(key)

    for t in Tvals:
        mse, mse_se = val[t]['MSE']
        mse_data['Estimator'].append(key2name[key])
        mse_data['Timesteps'].append(t)
        mse_data['MSE'].append(mse)
        mse_data['mse_se'].append(mse_se)
        mse_data['Reward'].append(args.reward)

title_str = "Mean Squared Error (All Parameters)"
keys.sort(key = lambda x: order[x])
colors=[key2color[x] for x in keys]


#ymax = 0.31
ymax = 0.35
if args.reward == 'bernoulli':
    #ymax = 0.4
    ymax = 0.5
elif args.reward == 'poisson':
    #ymax = 0.05
    ymax = 0.1

df_mse = pd.DataFrame.from_dict(mse_data)
gplot = ggplot(df_mse, aes(x='Timesteps', y='MSE', color='Estimator')) \
        + geom_line(size=2) \
        + geom_point(size=2) \
        + geom_errorbar( aes(x="Timesteps", ymin="MSE-mse_se",ymax="MSE+mse_se"), width=20, size=1 ) \
        + labs(y="Mean Squared Error", title=title_str) \
        + xlim(0, args.T+10) \
        + ylim(0, ymax) \
        + theme_seaborn() \
        + scale_color_manual(values=colors) \
        + scale_fill_manual( values =colors ) \
        + theme(figure_size=(10,5), text=element_text(size=20), legend_title = element_blank())
gplot.save( os.path.join( save_f, 'MSE_{}.png'.format(save_summary) ) )


###########################################################################################
###########################################################################################
# Plot ellipsoid confidence region

print("\nPlotting ellipsoid confidence region coverage...\n")

    
for alpha in alphas:
    CP_data = { 
            'Parameters': [],
            'Method': [],
            'Timesteps': [],
            'CP': [],
            'CP_se': [],
            'Reward': [],
            }
    
    for setting in ['allparam', 'advantage']:
        keys = []
        for key, val in CI_dict.items():
            keys.append(key)
            if key in ['W_least_squares', 'W-MLE']:
                suffix_list = [''] if setting=='allparam' else ['_adv', '_adv_proj']
                if setting == 'advantage':
                    keys.append(key+'_proj')
            else:
                suffix_list = [''] if setting=='allparam' else ['_adv']

            for suffix in suffix_list:
                for t in Tvals:
                    CI, CI_se = val[t]['ellipsoid'+suffix][alpha]
                    CP_data['Parameters'].append( setting2name[setting] )
                    if 'proj' in suffix:
                        CP_data['Method'].append( key2name[key] + " (Projected)" )
                    else:
                        CP_data['Method'].append( key2name[key] )
                    CP_data['Timesteps'].append(t)
                    CP_data['CP'].append(CI)
                    CP_data['CP_se'].append(CI_se)
                    CP_data['Reward'].append(args.reward)

    title_str = "Empirical Coverage Probabilities"
    
    keys.sort(key = lambda x: order[x])
    colors=[key2color[x] for x in keys]

    df_CP = pd.DataFrame.from_dict(CP_data)
   
    ymin = 0.68
    if args.reward == 'Poisson':
        ymin = 0.8

    # Reorder plots
    df_CP['Parameters'] = df_CP['Parameters'].astype("category")
    df_CP['Parameters'] = df_CP['Parameters'].cat.reorder_categories(['All Parameters', 'Advantage Parameters'])

    gplot = (ggplot(df_CP, aes(x='Timesteps', y='CP', color='Method')) \
            + geom_line(size=2) \
            + geom_point(size=2) \
            + geom_errorbar( aes(x="Timesteps", ymin="CP-CP_se",ymax="CP+CP_se"), width=20, size=1 ) \
            + labs(y="Coverage Probability", title=title_str) \
            + xlim(0, args.T+10) \
            + ylim(ymin, 1) \
            + theme_seaborn() \
            + scale_color_manual(values=colors) \
            + scale_fill_manual( values =colors ) \
            + geom_hline(yintercept=1-alpha, color='black', size=1) \
            + theme(figure_size=(10,5), text=element_text(size=20), legend_title = element_blank()) \
            + facet_wrap('~Parameters'))
    gplot.save( os.path.join( save_f, 'CP_alpha={}_{}.png'.format(int(alpha*100), save_summary) ) )



###########################################################################################
###########################################################################################
# Plot ellipsoid confidence volume


print("\nPlotting ellipsoid confidence region volume...\n")

for alpha in alphas:
    V_data = { 
            'Parameters': [],
            'Method': [],
            'Timesteps': [],
            'Volume': [],
            'vol_se': [],
            'Reward': [],
            }
    
    for setting in ['allparam', 'advantage']:
        keys = []
        for key, val in CI_dict.items():
            keys.append(key)
            if key in ['W_least_squares', 'W-MLE']:
                suffix_list = [''] if setting=='allparam' else ['_adv', '_adv_proj']
                if setting == 'advantage':
                    keys.append(key+'_proj')
            else:
                suffix_list = [''] if setting=='allparam' else ['_adv']
            

            for suffix in suffix_list:
                for t in Tvals:
                    CI, CI_se = val[t]['ellipsoid{}_vol'.format(suffix)][alpha]
                    V_data['Parameters'].append( setting2name[setting] )
                    if 'proj' in suffix:
                        V_data['Method'].append( key2name[key] + " (Projected)" )
                    else:
                        V_data['Method'].append( key2name[key] )
                    V_data['Timesteps'].append(t)
                    V_data['Volume'].append(CI)
                    V_data['vol_se'].append(CI_se)
                    V_data['Reward'].append(args.reward)

    title_str = "Volume of Confidence Ellipsoids"
    keys.sort(key = lambda x: order[x])
    colors=[key2color[x] for x in keys]
    
    df_V = pd.DataFrame.from_dict(V_data)
    
    # Reorder plots
    df_V['Parameters'] = df_V['Parameters'].astype("category")
    df_V['Parameters'] = df_V['Parameters'].cat.reorder_categories(['All Parameters', 'Advantage Parameters'])
    
    gplot = ( ggplot(df_V, aes(x='Timesteps', y='Volume', color='Method')) \
            + geom_line(size=2) \
            + geom_point(size=2) \
            + geom_errorbar( aes(x="Timesteps", ymin="Volume-vol_se",ymax="Volume+vol_se"), width=20, size=1 ) \
            + labs(y="Volume (log scale)", title=title_str) \
            + xlim(0, args.T+100) \
            + scale_y_log10() \
            + theme_seaborn() \
            + scale_color_manual(values=colors) \
            + scale_fill_manual( values =colors ) \
            + theme(figure_size=(10,5), text=element_text(size=20), legend_title = element_blank()) \
            + facet_wrap('~Parameters'))
    try:
        gplot.save( os.path.join( save_f, 'vol_alpha={}_{}.png'.format(int(alpha*100), save_summary) ) )
    except:
        import ipdb; ipdb.set_trace()

###########################################################################################
###########################################################################################
dict_data = {
            "mse_data": mse_data,
            "CP_data": CP_data,
            "V_data": V_data,
        }
pickle.dump( dict_data, open( os.path.join( save_f, 'dict_data.p'), 'wb' ) )

# Consolidate plots
if args.reward == 'bernoulli':
    tdist_dict_data = pickle.load( open( os.path.join( save_f.replace("bernoulli", "t-dist"), 'dict_data.p'), 'rb' ) )
    
    tdist_CP_data = tdist_dict_data['CP_data']
    tdist_V_data = tdist_dict_data['V_data']

    #########################
    # Coverage Probabilities
    #########################
    all_CP_data = {
                "Parameters": CP_data["Parameters"]+tdist_CP_data["Parameters"],
                "Timesteps": CP_data["Timesteps"]+tdist_CP_data["Timesteps"],
                "CP": CP_data["CP"]+tdist_CP_data["CP"],
                "CP_se": CP_data["CP_se"]+tdist_CP_data["CP_se"],
                "Reward": CP_data["Reward"]+tdist_CP_data["Reward"],
                "Method": [],
                "Model": [reward2model[r] for r in CP_data["Reward"]] + [reward2model[r] for r in tdist_CP_data["Reward"]],
            }
    
    for x in CP_data['Method']:
        if x == 'MLE':
            all_CP_data['Method'].append(key2name['UW'])
        elif x == 'AW-MLE':
            all_CP_data['Method'].append(key2name['AW'])
        elif x == 'AW-MLE (Projected)':
            all_CP_data['Method'].append(key2name['AW_proj'])
        elif x == 'Self-Normalized Bound':
            all_CP_data['Method'].append(key2name['snmb'])
        else:
            all_CP_data['Method'].append(x)

    for x in tdist_CP_data['Method']:
        if x == 'OLS':
            all_CP_data['Method'].append(key2name['UW'])
        elif x == 'AW-LS':
            all_CP_data['Method'].append(key2name['AW'])
        elif x == 'AW-LS (Projected)':
            all_CP_data['Method'].append(key2name['AW_proj'])
        elif x == 'Self-Normalized Bound':
            all_CP_data['Method'].append(key2name['snmb'])
        else:
            all_CP_data['Method'].append(x)

    df_all_CP = pd.DataFrame.from_dict(all_CP_data)

    # Reorder Categories
    df_all_CP['Parameters'] = df_all_CP['Parameters'].astype("category")
    df_all_CP['Parameters'] = df_all_CP['Parameters'].cat.reorder_categories(['All Parameters', 'Advantage Parameters'])
    
    df_all_CP['Model'] = df_all_CP['Model'].astype("category")
    df_all_CP['Model'] = df_all_CP['Model'].cat.reorder_categories(['Linear Model', 'Logistic Model'])
    
    keys = ['UW', 'AW', 'AW_proj', 'Wdecor', 'snmb']
    df_all_CP['Method'] = df_all_CP['Method'].astype("category")
    df_all_CP['Method'] = df_all_CP['Method'].cat.reorder_categories([key2name[x] for x in keys])

    colors = [key2color[x] for x in keys]

    gplot = (ggplot(df_all_CP, aes(x='Timesteps', y='CP', color='Method')) \
            + geom_line(size=2) \
            + geom_point(size=2) \
            + labs(y="Coverage Probability", title="{}% Confidence Regions".format( int((1-alpha)*100)) ) \
            + xlim(0, args.T+10) \
            + ylim(0.78, 1) \
            + theme_seaborn() \
            + geom_hline(yintercept=1-alpha, color='black', size=1) \
            + scale_color_manual(values=colors) \
            + scale_fill_manual( values =colors ) \
            + theme(figure_size=(10,2), text=element_text(size=12), legend_title = element_blank(),
                axis_ticks_minor_x = element_line(color='white') ) \
            + facet_grid('. ~ Model + Parameters'))
    gplot.save( os.path.join( save_f, 'all_CP_alpha={}_{}.png'.format(int(alpha*100), save_summary) ) )
    

    #########################
    # Volumes
    #########################
    all_V_data = {
                "Parameters": V_data["Parameters"]+tdist_V_data["Parameters"],
                "Timesteps": V_data["Timesteps"]+tdist_V_data["Timesteps"],
                "Volume": V_data["Volume"]+tdist_V_data["Volume"],
                "vol_se": V_data["vol_se"]+tdist_V_data["vol_se"],
                "Reward": V_data["Reward"]+tdist_V_data["Reward"],
                "Method": [],
                "Model": [reward2model[r] for r in V_data["Reward"]] + [reward2model[r] for r in tdist_V_data["Reward"]],
            }
   
    all_V_data['Method'] = all_CP_data['Method']
    df_all_V = pd.DataFrame.from_dict(all_V_data)

    # Reorder Categories
    df_all_V['Parameters'] = df_all_V['Parameters'].astype("category")
    df_all_V['Parameters'] = df_all_V['Parameters'].cat.reorder_categories(['All Parameters', 'Advantage Parameters'])
    
    df_all_V['Model'] = df_all_V['Model'].astype("category")
    df_all_V['Model'] = df_all_V['Model'].cat.reorder_categories(['Linear Model', 'Logistic Model'])
    
    keys = ['UW', 'AW', 'AW_proj', 'Wdecor', 'snmb']
    df_all_V['Method'] = df_all_V['Method'].astype("category")
    df_all_V['Method'] = df_all_V['Method'].cat.reorder_categories([key2name[x] for x in keys])

    colors = [key2color[x] for x in keys]

    gplot = (ggplot(df_all_V, aes(x='Timesteps', y='Volume', color='Method')) \
            + geom_line(size=2) \
            + geom_point(size=2) \
            + geom_errorbar( aes(x="Timesteps", ymin="Volume-vol_se",ymax="Volume+vol_se"), width=20, size=1 ) \
            + labs(y="Volume (log scale)") \
            + scale_y_log10() \
            + xlim(0, args.T+10) \
            + theme_seaborn() \
            + scale_color_manual(values=colors) \
            + scale_fill_manual( values =colors ) \
            + theme(figure_size=(10,2), text=element_text(size=12), legend_title = element_blank()) \
            + facet_grid('. ~ Model + Parameters'))
    gplot.save( os.path.join( save_f, 'all_Vol_alpha={}_{}.png'.format(int(alpha*100), save_summary) ) )
   
    #########################
    # CP and Volumes
    #########################
    all_data = {
                "Parameters": all_CP_data["Parameters"] + all_V_data["Parameters"],
                "Method":  all_CP_data["Method"] + all_V_data["Method"],
                "Timesteps": all_CP_data["Timesteps"] + all_V_data["Timesteps"],
                "stat": all_CP_data["CP"] + all_V_data["Volume"],
                "stat_se": all_CP_data["CP_se"] + all_V_data["vol_se"],
                "stat_type": ["Coverage\nProbability" for x in range(len(all_CP_data["CP"]))] \
                        + ["Volume\n(log scale)" for x in range(len(all_V_data["Volume"]))],
                "Reward": all_CP_data["Reward"] + all_V_data["Reward"],
                "Model": all_CP_data["Model"] + all_V_data["Model"],
            }
    df_all = pd.DataFrame.from_dict(all_data)
    
    # Reorder Categories
    df_all['Parameters'] = df_all['Parameters'].astype("category")
    df_all['Parameters'] = df_all['Parameters'].cat.reorder_categories(['All Parameters', 'Advantage Parameters'])
    
    df_all['Model'] = df_all['Model'].astype("category")
    df_all['Model'] = df_all['Model'].cat.reorder_categories(['Linear Model', 'Logistic Model'])
    
    keys = ['UW', 'AW', 'AW_proj', 'Wdecor', 'snmb']
    df_all['Method'] = df_all['Method'].astype("category")
    df_all['Method'] = df_all['Method'].cat.reorder_categories([key2name[x] for x in keys])
    
    gplot = (ggplot(df_all, aes(x='Timesteps', y='stat', color='Method')) \
            + geom_line(size=2) \
            + geom_point(size=2) \
            + geom_errorbar( aes(x="Timesteps", ymin="stat-stat_se",ymax="stat+stat_se"), width=20, size=1 ) \
            + xlim(0, args.T+10) \
            #+ ylim(0.68, 1) \
            + ylim(0.79, 1) \
            + labs(y="", title="{}% Confidence Regions".format( int((1-alpha)*100) ) ) \
            + theme_seaborn() \
            + geom_hline(yintercept=1-alpha, color='black', size=1) \
            + scale_color_manual(values=colors) \
            + scale_fill_manual( values =colors ) \
            + theme(figure_size=(12,5), text=element_text(size=15), legend_title = element_blank(), \
                plot_title=element_text(size=20), axis_title_x=element_text(size=20), \
                legend_text_legend=element_text(size=18), legend_entry_spacing_y=20) \
            + facet_grid('stat_type ~ Model + Parameters', scales="free_y"))
    gplot.save( os.path.join( save_f, 'all_alpha={}_{}.png'.format(int(alpha*100), save_summary) ) )
    
    gplot = (ggplot(df_all, aes(x='Timesteps', y='stat', color='Method')) \
            + geom_line(size=2) \
            + geom_point(size=2) \
            + geom_errorbar( aes(x="Timesteps", ymin="stat-stat_se",ymax="stat+stat_se"), width=20, size=1 ) \
            + xlim(0, args.T+10) \
            + labs(y="", title="{}% Confidence Regions".format( int((1-alpha)*100) ) ) \
            + theme_seaborn() \
            + scale_y_log10() \
            + scale_color_manual(values=colors) \
            + scale_fill_manual( values =colors ) \
            + theme(figure_size=(12,5), text=element_text(size=15), legend_title = element_blank(), \
                plot_title=element_text(size=20), axis_title_x=element_text(size=20), \
                legend_text_legend=element_text(size=18), legend_entry_spacing_y=20) \
            + facet_grid('stat_type ~ Model + Parameters', scales="free_y"))
    gplot.save( os.path.join( save_f, 'all2_alpha={}_{}.png'.format(int(alpha*100), save_summary) ) )

