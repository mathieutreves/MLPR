from utils import *
from project_07 import compute_minDCF_actDCF

best_results = {'name' : "", 'lambda' : 0, 'minDCF': float('inf'), 'actDCF': float('inf'), 'scores': ""}

def trainLogReg(DTR, LTR, l):

    ZTR = 2.0 * LTR - 1.0
    def logreg_obj(v):
        w, b = v[:-1], v[-1]
        S = np.dot(vcol(w).T, DTR).ravel() + b

        loss = np.logaddexp(0, -ZTR * S)

        G = - ZTR / (1.0 + np.exp(ZTR * S))
        GW = (vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()
        return loss.mean() + l / 2 * np.linalg.norm(w)**2, np.hstack([GW, np.array(Gb)])

    x0 = np.zeros(DTR.shape[0] + 1)
    vf, _, _ = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, x0=x0, approx_grad=False)
    return vf[:-1], vf[-1]

def trainWeightedLogRegBinary(DTR, LTR, l, pT):

    ZTR = LTR * 2.0 - 1.0
    
    wTrue = pT / np.sum(ZTR > 0)
    wFalse = (1 - pT) / np.sum(ZTR < 0)
    weights = np.where(ZTR > 0, wTrue, wFalse)

    def logreg_obj_with_grad(v):
        w, b = v[:-1], v[-1]
        s = DTR.T @ w + b

        loss = np.logaddexp(0, -ZTR * s)
        loss_weighted = np.sum(weights * loss)
        G = -ZTR * weights / (1.0 + np.exp(ZTR * s))
        
        GW = DTR @ G + l * w
        Gb = np.sum(G)
        return loss_weighted + 0.5 * l * np.linalg.norm(w)**2, np.hstack([GW, np.array(Gb)])

    x0 = np.zeros(DTR.shape[0] + 1)
    vf, _, _ = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = x0, approx_grad=False)
    return vf[:-1], vf[-1]

def expand_features(DTR, DVAL):
    _, N = DTR.shape
    quad_terms = np.array([np.outer(DTR[:, i], DTR[:, i]).reshape(-1) for i in range(N)]).T
    DTR_expanded = np.vstack([quad_terms, DTR])
    
    _, N = DVAL.shape
    quad_terms = np.array([np.outer(DVAL[:, i], DVAL[:, i]).reshape(-1) for i in range(N)]).T
    DVAL_expanded = np.vstack([quad_terms, DVAL])
    return DTR_expanded, DVAL_expanded

def center_data(DTR, DVAL):
    mean = DTR.mean(axis=1, keepdims=True)
    DTR_centered = DTR - mean
    DVAL_centered = DVAL - mean
    return DTR_centered, DVAL_centered

def z_normalize_data(DTR, DVAL):
    mean = DTR.mean(axis=1, keepdims=True)
    std = DTR.std(axis=1, keepdims=True)
    DTR_normalized = (DTR - mean) / std
    DVAL_normalized = (DVAL - mean) / std
    return DTR_normalized, DVAL_normalized

def plot_dcf_vs_lambda(lambdas, minDCF, actDCF, title):
    plt.figure()
    plt.xscale('log', base=10)
    plt.plot(lambdas, minDCF, label="minDCF", marker='o')
    plt.plot(lambdas, actDCF, label="actDCF", marker='s')
    plt.xlabel("Lambda")
    plt.ylabel("DCF")
    plt.title(title)
    plt.legend()
    plt.grid(True)

def compute_dcf_foreach_lambda(DTR, LTR, DVAL, LVAL, title, weightPrior=False):
    global best_results
    best_results_local = {'name' : "", 'lambda' : 0, 'minDCF': float('inf'), 'actDCF': float('inf'), 'scores': ""}
    minDCF = []
    actDCF = []

    pT = 0.1
    lambdas = np.logspace(-4, 2, 13)
    for lamb in lambdas:
        w, b = trainLogReg(DTR, LTR, lamb) if not weightPrior else trainWeightedLogRegBinary(DTR, LTR, lamb, pT = pT)
        scores = np.dot(w.T, DVAL) + b
        pEmp = (LTR == 1).sum() / LTR.size
        adjusted_scores = scores - np.log(pEmp / (1 - pEmp)) if not weightPrior else scores - np.log(pT / (1 - pT))

        mDCF, aDCF = compute_minDCF_actDCF(adjusted_scores, LVAL, pT)
        minDCF.append(mDCF)
        actDCF.append(aDCF)
        
        if (minDCF[-1] < best_results["minDCF"]):
            best_results = { 'name': title, 'lambda': lamb, 'minDCF': minDCF[-1],
                            'actDCF': actDCF[-1], 'scores' : adjusted_scores }
            
        if (minDCF[-1] < best_results_local["minDCF"]):
            best_results_local = {  'name': title, 'lambda': lamb, 'minDCF': minDCF[-1],
                                  'actDCF': actDCF[-1], 'scores' : adjusted_scores }
            
    print(f"Model: {best_results_local['name']} - lambda: {best_results_local['lambda']} - "
        f"minDCF: {best_results_local['minDCF']:.4f} - actDCF: {best_results_local['actDCF']:.4f}")

    return lambdas, minDCF, actDCF

def evaluate_model(DTR, LTR, DVAL, LVAL, title, transform_fn):    
    DTR_proc, DVAL_proc = transform_fn(DTR, DVAL)
    lambdas, minDCF, actDCF = compute_dcf_foreach_lambda(DTR_proc, LTR, DVAL_proc, LVAL, title)    
    return lambdas, minDCF, actDCF

def main(plot=False):
    features, classes = loadDataset('/trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(features, classes)

    # Use whole dataset
    title = "whole dataset"
    lambdas, minDCF, actDCF = compute_dcf_foreach_lambda(DTR, LTR, DVAL, LVAL, title) 
    if plot: plot_dcf_vs_lambda(lambdas, minDCF, actDCF, "minDCF and actDCF vs Lambda - " + title)

    # Use less training examples
    title = "smaller dataset"
    lambdas, minDCF, actDCF = compute_dcf_foreach_lambda(DTR[:, ::50], LTR[::50], DVAL, LVAL, title)
    if plot: plot_dcf_vs_lambda(lambdas, minDCF, actDCF, "minDCF and actDCF vs Lambda - smaller dataset")

    # Prior-weighted version of the model
    title = "prior-weighted model"
    lambdas, minDCF, actDCF = compute_dcf_foreach_lambda(DTR, LTR, DVAL, LVAL, title, weightPrior=True)
    if plot: plot_dcf_vs_lambda(lambdas, minDCF, actDCF, "minDCF and actDCF vs Lambda - prior-weighted model")

    results = {
        'Expanded features': evaluate_model(DTR, LTR, DVAL, LVAL, 'Expanded features', expand_features),
        'Centering': evaluate_model(DTR, LTR, DVAL, LVAL, 'Centering', center_data),
        'Z-normalization': evaluate_model(DTR, LTR, DVAL, LVAL, 'Z-normalization', z_normalize_data)
    }

    # Plotting results
    if plot:
        for _, (name, (lambdas, minDCF, actDCF)) in enumerate(results.items()):
            plot_dcf_vs_lambda(lambdas, minDCF, actDCF, f'Linear Logistic Regression: {name}')
        plt.show()
    
    print(f"Best model: {best_results['name']} - lambda: {best_results['lambda']} - "
          f"minDCF: {best_results['minDCF']:.4f} - actDCF: {best_results['actDCF']:.4f}")
    
    # Save best model scores
    save_scores(best_results['scores'], '/scores_logReg.json')

if __name__ == "__main__":
    main(plot=True)