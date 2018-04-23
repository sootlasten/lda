"""Collapsed Gibbs sampler for LDA."""
import time
import numpy as np
from scipy.special import gammaln
from utils import randinit_z, calc_counts


def _col_lda_logjoint(alpha, beta, A, B, M):
    """Calculates the log-joint of a collapsed LDA."""
    (D, K), W = A.shape, B.shape[1]
    N = np.sum(A, axis=1)  # array of size D, d-th elem holds the total number of words in document d

    logjoint = 0
    logjoint += D*gammaln(K*alpha)
    logjoint += -K*D*gammaln(alpha)
    logjoint += K*gammaln(W*beta)
    logjoint += -W*K*gammaln(beta)
    logjoint += gammaln(A + alpha).sum() - gammaln(N + K*alpha).sum()
    logjoint += gammaln(B + beta).sum() - gammaln(M + W*beta).sum()
    return logjoint


def _col_lda_logpred(Z, alpha, beta, A, B, M, testc):
    """Calculates the log-preditive probs for collapsed LDA."""
    Pdk = alpha + A
    Pdk = Pdk / Pdk.sum(axis=1)[:, np.newaxis]

    Pkw = beta + B
    Pkw = Pkw / Pkw.sum(axis=1)[:, np.newaxis]

    p = np.array([np.dot(Pdk[d], Pkw[:, w]) for d in Z for w in Z[d]])
    return np.dot(testc, np.log(p))
    

def col_gibbs(data, K, idx_to_word=dict(), alpha=100, beta=0.1, nb_iters=200):
    """Gibbs sampler for the collapsed LDA."""
    Z = randinit_z(data, K)
    W = len(np.unique(data[:, 1]))
    A, B, M = calc_counts(Z, K, W)

    logjoints, logpreds = [], []
    logjoints.append(_col_lda_logjoint(alpha, beta, A, B, M))
    logpreds.append(_col_lda_logpred(Z, alpha, beta, A, B, M, data[:, -1]))

    print("Starting logjoint: {}".format(logjoints[-1]))
    for i in range(nb_iters):
        start_time = time.time()

        for d in Z:
            for w in Z[d]:
                for wi, old_k in enumerate(Z[d][w]):
                    A[d, old_k] -= 1; B[old_k, w] -= 1; M[old_k] -= 1
                    
                    probs = (A[d, :] + alpha) * ((B[:, w] + beta) / (M + W*beta))
                    probs /= np.sum(probs)
                                        
                    new_k = np.random.choice(len(probs), p=probs)
                    Z[d][w][wi] = new_k
                    A[d, new_k] += 1; B[new_k, w] += 1; M[new_k] += 1
        
        # output topics
        if idx_to_word:
             a = np.flip(np.argsort(B, axis=1), axis=1)[:, :5]
             f = np.vectorize(lambda i: idx_to_word[i+1])
             print(f(a));

        # monitor logjoint
        logjoints.append(_col_lda_logjoint(alpha, beta, A, B, M))
        logpreds.append(_col_lda_logpred(Z, alpha, beta, A, B, M, data[:, -1]))
    
        iter_time = time.time() - start_time
        print("Iter {}/{} ({:.2f} s). Logjoint: {:.2f}".format(i+1, nb_iters, iter_time, logjoints[-1]))
        print()

    return logjoints, logpreds, Z, A, B, M 

