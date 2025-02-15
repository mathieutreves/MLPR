from utils import *
from project_07 import compute_minDCF_actDCF
from project_08 import center_data

def train_dual_SVM_linear(DTR, LTR, C, K = 1):
    ZTR = LTR * 2.0 - 1.0
    DTR_EXT = np.vstack([DTR, np.ones((1,DTR.shape[1])) * K])
    H = (DTR_EXT.T @ DTR_EXT) * vcol(ZTR) * vrow(ZTR)

    def fOpt(alpha):
        Ha = H @ alpha
        loss = 0.5 * alpha @ Ha - np.sum(alpha)
        grad = Ha - 1
        return loss, grad
    
    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(
        fOpt, 
        np.zeros(DTR_EXT.shape[1]), 
        bounds=[(0, C)] * len(LTR), 
        factr=1.0
    )
    
    w_hat = (alphaStar * ZTR) @ DTR_EXT.T
    w, b = w_hat[:DTR.shape[0]], w_hat[-1] * K
    
    return w, b

def polyKernel(degree, c):
    def polyKernelFunc(D1, D2):
        return (D1.T @ D2 + c)**degree
    return polyKernelFunc

def rbfKernel(gamma):
    def rbfKernelFunc(D1, D2):
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
        return np.exp(-gamma * Z)
    return rbfKernelFunc

def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps = 1.0):
    ZTR = LTR * 2.0 - 1.0
    K = kernelFunc(DTR, DTR) + eps
    H = vcol(ZTR) * vrow(ZTR) * K

    def fOpt(alpha):
        Ha = H @ alpha
        loss = 0.5 * alpha @ Ha - np.sum(alpha)
        grad = Ha - 1
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(
        fOpt, 
        np.zeros(DTR.shape[1]), 
        bounds=[(0, C)] * len(LTR), 
        factr=1.0
    )
    
    def fScore(DTE):
        K = kernelFunc(DTR, DTE) + eps
        H = vcol(alphaStar) * vcol(ZTR) * K
        return H.sum(0)

    return fScore

def plot_dcf_vs_C(Cs, minDCF, actDCF, title):
    plt.figure()
    plt.xscale('log', base=10)
    plt.plot(Cs, minDCF, label="minDCF", marker='o')
    plt.plot(Cs, actDCF, label="actDCF", marker='s')
    plt.xlabel("C")
    plt.ylabel("DCF")
    plt.title(title)
    plt.legend()
    plt.grid(True)

def process_svm(DTR, LTR, DVAL, LVAL, config, gamma=None, gamma_name=''):
    best_results_local = {'svmType': "", 'C': 0, 'centered': False, 'd': 0, 'c': 0, 'eps': 0, 'gamma': 0, 'minDCF': float('inf'), 'actDCF': float('inf'), 'scores': None}
    minDCFs, actDCFs = [], []
    for C in config['Cs']:
        if config['name'].startswith("Linear"):
            if config['centered']:
                DTR_centered, DVAL_centered = center_data(DTR, DVAL)
                w, b = train_dual_SVM_linear(DTR_centered, LTR, C, K=1)
                SVAL = (vrow(w) @ DVAL_centered + b).ravel()
            else:
                w, b = train_dual_SVM_linear(DTR, LTR, C, K=1)
                SVAL = (vrow(w) @ DVAL + b).ravel()
        elif config['name'].startswith("Polynomial"):
            fScore = train_dual_SVM_kernel(DTR, LTR, C, polyKernel(config['d'], config['c']), config['eps'])
            SVAL = fScore(DVAL)
        elif config['name'].startswith("RBF"):
            fScore = train_dual_SVM_kernel(DTR, LTR, C, rbfKernel(gamma), config['eps'])
            SVAL = fScore(DVAL)
        
        mDCF, aDCF = compute_minDCF_actDCF(SVAL, LVAL, 0.1)
        minDCFs.append(mDCF)
        actDCFs.append(aDCF)
    
    update_best_results(best_results_local, config, minDCFs, actDCFs, gamma_name)
    print_best_results(best_results_local)
    
    return minDCFs, actDCFs

def update_best_results(best_results, config, minDCFs, actDCFs, gamma_name=''):
    best_idx = np.argmin(minDCFs)
    if minDCFs[best_idx] < best_results['minDCF']:
        best_results.update({
            'svmType': config['name'],
            'C': config['Cs'][best_idx],
            'centered': config.get('centered', False),
            'd': config.get('d', 0),
            'c': config.get('c', 0),
            'eps': config.get('eps', 0),
            'gamma': gamma_name,
            'minDCF': minDCFs[best_idx],
            'actDCF': actDCFs[best_idx]
        })

def print_best_results(best_results, overall=False):
    if overall:
        print("BEST MODEL OVERALL")
        print("==================")

    print(f"Model: {best_results['svmType']} - C: {best_results['C']} - centered: {best_results['centered']}", end=" ")
    if best_results['svmType'].startswith("Polynomial"):
        print(f"d: {best_results['d']} - c: {best_results['c']} - eps: {best_results['eps']}", end=" ")
    if best_results['svmType'].startswith("RBF"):
        print(f"- gamma: {best_results['gamma']} - eps: {best_results['eps']}", end=" ")
    print(f"- minDCF: {best_results['minDCF']:.4f} - actDCF: {best_results['actDCF']:.4f}")

def main(plot=False):
    features, classes = loadDataset('/trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(features, classes)
    
    best_results = {'svmType': "", 'C': 0, 'centered': False, 'd': 0, 'c': 0, 'eps': 0, 'gamma': 0, 'minDCF': float('inf'), 'actDCF': float('inf'), 'scores': None}

    svm_configs = [
        # {'name': "Linear SVM", 'Cs': np.logspace(-5, 0, 11), 'centered': False},
        # {'name': "Linear SVM - centered", 'Cs': np.logspace(-5, 0, 11), 'centered': True},
        # {'name': "Polynomial SVM", 'Cs': np.logspace(-5, 0, 11), 'd': 2.0, 'c': 1.0, 'eps': 0},
        {'name': "RBF SVM", 'Cs': np.logspace(-3, 2, 11), 'eps': 1, 'gammas': [("e-4", np.exp(-4)), ("e-3", np.exp(-3)), ("e-2", np.exp(-2)), ("e-1", np.exp(-1))]},
        # {'name': "Polynomial SVM - optional", 'Cs': np.logspace(-5, 0, 11), 'd': 4.0, 'c': 1.0, 'eps': 0}
    ]

    for config in svm_configs:
        minDCFs, actDCFs = [], []
        
        if config['name'].startswith("RBF"):
            if plot:
                plt.figure()
                plt.xscale('log', base=10)
            
            for gamma_name, gamma in config['gammas']:
                minDCFs, actDCFs = process_svm(DTR, LTR, DVAL, LVAL, config, gamma=gamma, gamma_name=gamma_name)
                update_best_results(best_results, config, minDCFs, actDCFs)
                if plot:
                    plt.plot(config['Cs'], minDCFs, label=f"minDCF - gamma: {gamma_name}", marker='o')
                    plt.plot(config['Cs'], actDCFs, label=f"actDCF - gamma: {gamma_name}", marker='s')
                
            if plot:
                plt.xlabel("C")
                plt.ylabel("DCF")
                plt.title("RBF Kernel SVM")
                plt.legend()
                plt.grid(True)
        else:
            minDCFs, actDCFs = process_svm(DTR, LTR, DVAL, LVAL, config)
            update_best_results(best_results, config, minDCFs, actDCFs)
            
            if plot:
                plot_dcf_vs_C(config['Cs'], minDCFs, actDCFs, config['name'])
        
    if plot:
        plt.show()
    
    print_best_results(best_results, overall=True)

if __name__ == "__main__":
    main()