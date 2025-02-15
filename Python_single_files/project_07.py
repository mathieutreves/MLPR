from utils import *

def compute_confusion_matrix(predictions, labels):
    nClasses = labels.max() + 1
    M = np.zeros((nClasses, nClasses))
    for i in range(labels.size):
        M[predictions[i], labels[i]] += 1
    return M

def compute_optimal_Bayes(llr, prior, Cfn, Cfp):
    thresh = -np.log((prior * Cfn) / ((1 - prior) * Cfp))
    return np.int32(llr > thresh)

def compute_empirical_Bayes_risk(predictions, labels, prior, Cfn, Cfp):
    M = compute_confusion_matrix(predictions, labels)
    Pfn = M[0,1] / (M[0,1] + M[1,1])
    Pfp = M[1,0] / (M[0,0] + M[1,0])
    bayesError = prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp
    return bayesError / np.minimum(prior * Cfn, (1 - prior) * Cfp)
                
def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = np.argsort(llr)
    llrSorted = llr[llrSorter]
    classLabelsSorted = classLabels[llrSorter]
    
    nTrue = np.sum(classLabelsSorted)
    nFalse = len(classLabelsSorted) - nTrue
    
    cumsum = np.cumsum(classLabelsSorted)
    nFalseNegative = cumsum
    nFalsePositive = nFalse - (np.arange(len(classLabelsSorted)) + 1 - cumsum)
    
    Pfn = np.concatenate(([0], nFalseNegative / nTrue))
    Pfp = np.concatenate(([1], nFalsePositive / nFalse))
    
    unique_indices = np.concatenate(([0], np.where(np.diff(llrSorted))[0] + 1, [len(llrSorted)]))
    
    return Pfn[unique_indices], Pfp[unique_indices]

def compute_actDCF(llrs, classLabels, effPrior, Cfn, Cfp):
    predictions = compute_optimal_Bayes(llrs, effPrior, Cfn, Cfp)
    return compute_empirical_Bayes_risk(predictions, classLabels, effPrior, Cfn, Cfp)

def compute_minDCF(llr, classLabels, prior, Cfn, Cfp):
    Pfn, Pfp = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / np.minimum(prior * Cfn, (1 - prior) * Cfp)
    return minDCF[np.argmin(minDCF)]

def compute_minDCF_actDCF(llrs, LVAL, targetPrior, Cfn = 1.0, Cfp = 1.0):
    minDCF = compute_minDCF(llrs, LVAL, targetPrior, Cfn, Cfp)
    actDCF = compute_actDCF(llrs, LVAL, targetPrior, Cfn, Cfp)
    return minDCF, actDCF

def compute_likelihoods(PCA_DVAL, mu_0, mu_1, C_0, C_1):
    S_MVG = np.array([
        np.exp(logpdf_GAU_ND(PCA_DVAL, mu_0, C_0)),
        np.exp(logpdf_GAU_ND(PCA_DVAL, mu_1, C_1))
    ])
    S_Tied = np.array([
        np.exp(logpdf_GAU_ND(PCA_DVAL, mu_0, 1/2 * (C_0 + C_1))),
        np.exp(logpdf_GAU_ND(PCA_DVAL, mu_1, 1/2 * (C_0 + C_1)))
    ])
    S_NB = np.array([
        np.exp(logpdf_GAU_ND(PCA_DVAL, mu_0, C_0 * np.eye(C_0.shape[0], C_0.shape[1]))),
        np.exp(logpdf_GAU_ND(PCA_DVAL, mu_1, C_1 * np.eye(C_1.shape[0], C_1.shape[1])))])

    return [("MVG", S_MVG), ("Tied MVG", S_Tied), ("Naive Bayes", S_NB)]

def compute_metrics(S, LVAL, effPrior, Cfn, Cfp):
    logSJoint = np.log(S) + vcol(np.log(0.5))
    llrs = logSJoint[1] - logSJoint[0]
    actDCF = compute_actDCF(llrs, LVAL, effPrior, Cfn, Cfp)
    minDCF = compute_minDCF(llrs, LVAL, effPrior, Cfn, Cfp)
    return llrs, minDCF, actDCF

def create_comparison_table(priors, Cfn, Cfp, DTR, DVAL, LTR, LVAL):
    table_data = []
    best_results = {'overall': {'minDCF': float('inf')}, 'prior_01': {'minDCF': float('inf')}}

    for prior in priors:
        effPrior = (prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)

        for i in range(6):
            U_pca = PCA(DTR, m=i+1, project=False)
            PCA_DTR = np.dot(U_pca.T, DTR)
            PCA_DVAL = np.dot(U_pca.T, DVAL)
            
            D_0 = PCA_DTR[:, LTR == 0]
            D_1 = PCA_DTR[:, LTR == 1]
            mu_0 = vcol(D_0.mean(1))
            mu_1 = vcol(D_1.mean(1))
            C_0 = ((D_0 - mu_0) @ (D_0 - mu_0).T) / D_0.shape[1]
            C_1 = ((D_1 - mu_1) @ (D_1 - mu_1).T) / D_1.shape[1]

            # Compute likelihoods for each model
            models = compute_likelihoods(PCA_DVAL, mu_0, mu_1, C_0, C_1)

            for model_name, S in models:
                _, min_dcf, act_dcf = compute_metrics(S, LVAL, effPrior, Cfn, Cfp)
                perc = act_dcf * 100 / min_dcf - 100

                table_data.append([
                    model_name, f"{prior:.2f}", i + 1,
                    f"{min_dcf:.4f}", f"{act_dcf:.4f}", f"{perc:05.2f} %"
                ])

                # Update best results
                if min_dcf < best_results['overall']['minDCF']:
                    best_results['overall'] = {
                        'minDCF': min_dcf, 'model': model_name,
                        'PCA': i+1, 'prior': prior
                    }
                if prior == 0.1 and min_dcf < best_results['prior_01']['minDCF']:
                    best_results['prior_01'] = {
                        'minDCF': min_dcf, 'model': model_name, 'PCA': i+1
                    }


    model_order = ["MVG", "Tied MVG", "Naive Bayes"]
    table_data.sort(key=lambda row: model_order.index(row[0]))

    headers = ["Model", "Prior", "PCA Dim", "Min DCF", "Act DCF", "Perc"]
    latex_table = "\\begin{tabular}{|c|c|c|c|c|c|}\n\\hline\n"
    latex_table += " & ".join(headers) + " \\\\\n\\hline\\hline\n"

    for row in table_data:
        row_str = " & ".join(map(str, row))
        latex_table += row_str + " \\\\\n\\hline\n"

    latex_table += "\\end{tabular}"

    print(latex_table)

    print(f"Best model: {best_results['overall']['model']} - PCA: {best_results['overall']['PCA']} - "
          f"Prior: {best_results['overall']['prior']} with minDCF: {best_results['overall']['minDCF']}")
    print(f"Best model Prior 0.1: {best_results['prior_01']['model']} - PCA: {best_results['prior_01']['PCA']} "
          f"with minDCF: {best_results['prior_01']['minDCF']}")

def plot_bayes_error(Cfn, Cfp, DTR, DVAL, LTR, LVAL):
    effPriorLogOdds = np.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))

    D_0, D_1 = DTR[:, LTR == 0], DTR[:, LTR == 1]
    mu_0, mu_1 = vcol(D_0.mean(1)), vcol(D_1.mean(1))
    C_0, C_1 = np.cov(D_0), np.cov(D_1)

    models = compute_likelihoods(DVAL, mu_0, mu_1, C_0, C_1)

    for model_name, S in models:
        actDCF = []
        minDCF = []
        for effPrior in effPriors:
            _, min_dcf, act_dcf = compute_metrics(S, LVAL, effPrior, Cfn, Cfp)
            actDCF.append(act_dcf)
            minDCF.append(min_dcf)

        plt.figure()
        plt.title(f"Model: {model_name}")
        plt.plot(effPriorLogOdds, actDCF, label='DCF')
        plt.plot(effPriorLogOdds, minDCF, label='minDCF')
        plt.ylim([0, 1.0])
        plt.legend()
        plt.xlabel('Prior log-odds')
        plt.ylabel('DCF')
    plt.show()

def main(plot=False):
    features, classes = loadDataset('/trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(features, classes)

    priors = [0.5, 0.9, 0.1]
    Cfn, Cfp = 1.0, 1.0

    create_comparison_table(priors, Cfn, Cfp, DTR, DVAL, LTR, LVAL)
    if plot:
        plot_bayes_error(Cfn, Cfp, DTR, DVAL, LTR, LVAL)

if __name__ == "__main__":
    main(plot=True)