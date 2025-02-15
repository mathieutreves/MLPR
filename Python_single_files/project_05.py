from utils import *
from project_04 import compute_gaussian_parameters

def compute_log_likelihood_ratio(DVAL, mu_0, C_0, mu_1, C_1, Pc=0.5):
    S_0 = np.exp(logpdf_GAU_ND(DVAL, mu_0, C_0))
    S_1 = np.exp(logpdf_GAU_ND(DVAL, mu_1, C_1))
    S = np.array([S_0, S_1])
    logS = np.log(S)
    logSJoint = logS + vcol(np.log(Pc))
    return logSJoint[1] - logSJoint[0]

def compute_error_rate(llr, LVAL, thresh=0):
    predictions = np.where(llr >= thresh, 1, 0)
    return 1 - np.sum(predictions == LVAL) / LVAL.size

def mvg_classifier(DTR, LTR, DVAL, LVAL):
    D_0, D_1 = DTR[:, LTR == 0], DTR[:, LTR == 1]
    mu_0, C_0 = compute_gaussian_parameters(D_0)
    mu_1, C_1 = compute_gaussian_parameters(D_1)
    llr = compute_log_likelihood_ratio(DVAL, mu_0, C_0, mu_1, C_1)
    err = compute_error_rate(llr, LVAL)
    return err, C_0, C_1

def tied_classifier(DTR, LTR, DVAL, LVAL):
    D_0, D_1 = DTR[:, LTR == 0], DTR[:, LTR == 1]
    mu_0, C_0 = compute_gaussian_parameters(D_0)
    mu_1, C_1 = compute_gaussian_parameters(D_1)
    C_tied = (C_0 + C_1) / 2
    llr = compute_log_likelihood_ratio(DVAL, mu_0, C_tied, mu_1, C_tied)
    err = compute_error_rate(llr, LVAL)
    return err

def naive_bayes_classifier(DTR, LTR, DVAL, LVAL):
    D_0, D_1 = DTR[:, LTR == 0], DTR[:, LTR == 1]
    mu_0, C_0 = compute_gaussian_parameters(D_0)
    mu_1, C_1 = compute_gaussian_parameters(D_1)
    C_0_NB = C_0 * np.eye(C_0.shape[0])
    C_1_NB = C_1 * np.eye(C_1.shape[0])
    llr = compute_log_likelihood_ratio(DVAL, mu_0, C_0_NB, mu_1, C_1_NB)
    err = compute_error_rate(llr, LVAL)
    return err

def compute_pearson_correlation(C):
    return C / (vcol(C.diagonal()**0.5) * vrow(C.diagonal()**0.5))

def pca_analysis(DTR, LTR, DVAL, LVAL):
    for m in range(1, 7):
        U_pca = PCA(DTR, m=m, project=False)
        PCA_DTR = U_pca.T @ DTR
        PCA_DVAL = U_pca.T @ DVAL
        
        print(f"Applied PCA with m = {m}")
        mvg_err = mvg_classifier(PCA_DTR, LTR, PCA_DVAL, LVAL)[0]
        nb_err = naive_bayes_classifier(PCA_DTR, LTR, PCA_DVAL, LVAL)
        tied_err = tied_classifier(PCA_DTR, LTR, PCA_DVAL, LVAL)
        
        print(f"\tMVG Error rate is:\t{mvg_err:.4%}")
        print(f"\tNB Error rate is:\t{nb_err:.4%}")
        print(f"\tTied Error rate is:\t{tied_err:.4%}")

def feature_subset_analysis(DTR, LTR, DVAL, LVAL):
    feature_subsets = [
        slice(None),  # All features
        slice(0, 4),  # Feature 1 to 4
        slice(0, 2),  # Feature 1, 2
        slice(2, 4)   # Feature 3, 4
    ]
    
    for i, feature_subset in enumerate(feature_subsets):
        DTR_SUB = DTR[feature_subset, :]
        DVAL_SUB = DVAL[feature_subset, :]
        
        print(f"Processing with features {feature_subset}")
        mvg_err = mvg_classifier(DTR_SUB, LTR, DVAL_SUB, LVAL)[0]
        nb_err = naive_bayes_classifier(DTR_SUB, LTR, DVAL_SUB, LVAL)
        tied_err = tied_classifier(DTR_SUB, LTR, DVAL_SUB, LVAL)
        
        print(f"\tMVG Error rate is:\t{mvg_err:.4%}")
        print(f"\tNB Error rate is:\t{nb_err:.4%}")
        print(f"\tTied Error rate is:\t{tied_err:.4%}")

def main():
    features, classes = loadDataset('/trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(features, classes)
    
    print("Full dataset analysis:")
    mvg_err, C_0, C_1 = mvg_classifier(DTR, LTR, DVAL, LVAL)
    tied_err = tied_classifier(DTR, LTR, DVAL, LVAL)
    nb_err = naive_bayes_classifier(DTR, LTR, DVAL, LVAL)
    
    print(f"MVG Error rate is {mvg_err:.4%}")
    print(f"Tied Error rate is {tied_err:.4%}")
    print(f"Naive Bayes Error rate is {nb_err:.4%}")
    
    print("\nCovariance matrices:")
    print("Class 0:")
    print(C_0)
    print("Class 1:")
    print(C_1)
    
    print("\nPearson correlation coefficients:")
    print(f"Class 0:\n{compute_pearson_correlation(C_0).round(2)}")
    print(f"Class 1:\n{compute_pearson_correlation(C_1).round(2)}")
    
    print("\nPCA analysis:")
    pca_analysis(DTR, LTR, DVAL, LVAL)
    
    print("\nFeature subset analysis:")
    feature_subset_analysis(DTR, LTR, DVAL, LVAL)

if __name__ == "__main__":
    main()