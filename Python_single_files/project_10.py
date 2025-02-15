from utils import *
from project_07 import compute_minDCF_actDCF

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def logpdf_GMM(X, gmm):
    S = np.array([logpdf_GAU_ND(X, mu, C) + np.log(w) for w, mu, C in gmm])
    return scipy.special.logsumexp(S, axis=0)

def smooth_covariance_matrix(C, psi):
    U, s, _ = np.linalg.svd(C)
    s = np.maximum(s, psi)
    return U @ (vcol(s) * U.T)

def train_GMM_EM_Iteration(X, gmm, covType, psiEig):
    
    # E-step
    S = np.array([logpdf_GAU_ND(X, mu, C) + np.log(w) for w, mu, C in gmm])
    logdens = scipy.special.logsumexp(S, axis=0)
    gammaAllComponents = np.exp(S - logdens)

    # M-step
    gmmUpd = []
    for gIdx in range(len(gmm)): 
        gamma = gammaAllComponents[gIdx]
        Z = gamma.sum()
        F = vcol((vrow(gamma) * X).sum(1))
        S = (vrow(gamma) * X) @ X.T
        muUpd = F/Z
        CUpd = S/Z - muUpd @ muUpd.T
        wUpd = Z / X.shape[1]
        if covType == 'diagonal':
            CUpd  = CUpd * np.eye(X.shape[0])
        gmmUpd.append((wUpd, muUpd, CUpd))

    if covType == 'tied':
        CTied = 0
        for w, mu, C in gmmUpd:
            CTied += w * C
        gmmUpd = [(w, mu, CTied) for w, mu, C in gmmUpd]
    gmmUpd = [(w, mu, smooth_covariance_matrix(C, psiEig)) for w, mu, C in gmmUpd]
        
    return gmmUpd

def train_GMM_EM(X, gmm, covType, psiEig=None, epsLLAverage = 1e-6):
    ll_old  = logpdf_GMM(X, gmm).mean()
    ll_delta = None 

    while(ll_delta  is None or ll_delta  > epsLLAverage):
        gmm_updated = train_GMM_EM_Iteration(X, gmm, covType, psiEig)
        ll_new  = logpdf_GMM(X, gmm_updated).mean()
        ll_delta  = ll_new - ll_old 
        gmm = gmm_updated
        ll_old  = ll_new 

    return gmm

def split_GMM_LBG(gmm, alpha = 0.1):
    gmmOut = []
    for (w, mu, C) in gmm:
        U, s, _ = np.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        gmmOut.append((0.5 * w, mu - d, C))
        gmmOut.append((0.5 * w, mu + d, C))
    return gmmOut

def train_GMM_LBG_EM(X, numComponents, covType, psiEig, epsLLAverage = 1e-6, lbgAlpha = 0.1):

    mu, C = compute_mu_C(X)
    if covType == 'diagonal':
        C = C * np.eye(X.shape[0])
    
    gmm = [(1.0, mu, smooth_covariance_matrix(C, psiEig))]
    while len(gmm) < numComponents:
        gmm = split_GMM_LBG(gmm, lbgAlpha)
        gmm = train_GMM_EM(X, gmm, covType = covType, psiEig = psiEig, epsLLAverage = epsLLAverage)
    return gmm

def main():
    features, classes = loadDataset('/trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(features, classes)
    prior, Cfn, Cfp = 0.1, 1.0, 1.0

    DTR_0 = DTR[:, LTR==0]
    DTR_1 = DTR[:, LTR==1]
    
    best_results = {'covType' : "", 'numC0' : 0, 'numC1' : 0, 'minDCF': float('inf'), 'actDCF': float('inf'), 'scores': ""}

    numComponents = [1, 2, 4, 8, 16, 32]
    for covType in ['full', 'diagonal', 'tied']:
        print('GMM Type:', covType)
        for components_c0 in numComponents:
            for components_c1 in numComponents:
                gmm_C0 = train_GMM_LBG_EM(DTR_0, components_c0, covType = covType, psiEig = 0.01)
                gmm_C1 = train_GMM_LBG_EM(DTR_1, components_c1, covType = covType, psiEig = 0.01)

                SLLR = logpdf_GMM(DVAL, gmm_C1) - logpdf_GMM(DVAL, gmm_C0)
                minDCF, actDCF = compute_minDCF_actDCF(SLLR, LVAL, prior)

                if minDCF < best_results['minDCF']:
                    best_results = {
                        'covType': covType,
                        'numC0': components_c0, 'numC1': components_c1,
                        'minDCF': minDCF, 'actDCF': actDCF,
                        'scores': SLLR
                    }

                print(f'C0 components: {components_c0} - C1 components: {components_c1} \tminDCF: {minDCF:.4f} - actDCF: {actDCF:.4f}')
        print()
    
    print(f"Best model: {best_results['covType']} - numC0: {best_results['numC0']} - numC1: {best_results['numC1']} - "
          f"minDCF: {best_results['minDCF']:.4f} - actDCF: {best_results['actDCF']:.4f}")
    
    save_scores(best_results['scores'], './scores_GMM.json')

if __name__ == "__main__":
    main()