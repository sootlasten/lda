"""Standard (i.e. uncollapsed) Gibbs sampler for LDA."""
import numpy as np
from numpy.random import dirichlet
from scipy.special import gamma
from utils import randinit_z, calc_counts


def _std_lda_logjoint(alpha, beta, A, B, theta, phi):
    """Calculates the log-joint of a standard (i.e. uncollapsed) LDA."""
    (D, K), W = A.shape, B.shape[1]

    logjoint = 0
    logjoint += K*np.log(gamma(W*beta))
    logjoint += -W*K*np.log(gamma(beta))
    logjoint += D*np.log(gamma(K*alpha))
    logjoint += -K*D*np.log(gamma(alpha))
    logjoint += ((A + alpha - 1)*np.log(theta)).sum()
    logjoint += ((B + beta - 1)*np.log(phi)).sum()
    return logjoint


def _std_lda_logpred(Z, theta, phi, testc):
    """Calculate the log-predictive probs for standard LDA."""
    p = np.array([np.dot(theta[d], phi[:, w]) for d in Z for w in Z[d]])
    return np.dot(testc, np.log(p))
    

def std_gibbs(data, K, alpha=1, beta=1, nb_iters=200):
    """Gibbs sampler for the standard (i.e. uncollapsed) LDA."""
    Z = randinit_z(data, K)
    A, B, M = calc_counts(Z, K, len(np.unique(data[:, 1])))
    theta = np.array([dirichlet(doc_c + alpha) for doc_c in A])
    phi = np.array([dirichlet(topic_c + beta) for topic_c in B])

    logjoints, logpreds = [], []
    logjoints.append(_std_lda_logjoint(alpha, beta, A, B, theta, phi))
    logpreds.append(_std_lda_logpred(Z, theta, phi, data[:, -1]))

    for i in range(nb_iters):
        for d in Z:
            for w in Z[d]:
                for wi, old_k in enumerate(Z[d][w]):
                    A[d, old_k] -= 1; B[old_k, w] -= 1; M[old_k] -= 1
                    probs = np.zeros(K)
                    # for k in range(K):
                    #     probs[k] = phi[k, w] * theta[d, k]
                    # probs /= np.sum(probs)
                    probs[k] = phi[:, w] * theta[d, :]

                    new_k = np.random.choice(len(probs), p=probs)
                    Z[d][w][wi] = new_k
                    A[d, new_k] += 1; B[new_k, w] += 1; M[new_k] += 1

            theta[d] = dirichlet(A[d] + alpha)

        phi = np.array([dirichlet(topic_c + beta) for topic_c in B])
        
        logjoints.append(_std_lda_logjoint(alpha, beta, A, B, theta, phi))
        logpreds.append(_std_lda_logpred(Z, theta, phi, data[:, -1]))
    
    return logjoints, logpreds, Z, theta, phi, A, B, M 

