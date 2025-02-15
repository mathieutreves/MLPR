from utils import *
from project_07 import compute_minDCF_actDCF
from project_08 import trainLogReg, expand_features, trainWeightedLogRegBinary
from project_09 import train_dual_SVM_kernel, rbfKernel
from project_10 import train_GMM_LBG_EM, logpdf_GMM

KFOLD = 5
TARGET_PRIOR = 0.1

def extract_train_val_folds(X, idx):
    return np.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]

def shuffle_data(X, L):
    perm = np.random.permutation(X.shape[1])
    return X[:, perm], L[perm]

def bayesPlot(S, L, left = -3, right = 3, npts = 21):
    effPriorLogOdds = np.linspace(left, right, npts)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    minDCF, actDCF = np.array([compute_minDCF_actDCF(S, L, p) for p in effPriors]).T
    return effPriorLogOdds, actDCF, minDCF

def calibrate_scores(model_name, scores, labels, priors):
    best_minDCF, best_actDCF, best_prior = float('inf'), float('inf'), 0
    calibrated_scores, calibrated_labels = [], []

    for pT in priors:
        fold_scores, fold_labels = [], []
        for fold in range(KFOLD):
            SCAL, SVAL = extract_train_val_folds(scores, fold)
            LCAL, LVAL = extract_train_val_folds(labels, fold)

            w, b = trainWeightedLogRegBinary(vrow(SCAL), LCAL, 0, pT)
            calibrated_SVAL = (w.T @ vrow(SVAL) + b - np.log(pT / (1 - pT))).ravel()

            fold_scores.append(calibrated_SVAL)
            fold_labels.append(LVAL)

        fold_scores = np.hstack(fold_scores)
        fold_labels = np.hstack(fold_labels)

        minDCF, actDCF = compute_minDCF_actDCF(fold_scores, fold_labels, TARGET_PRIOR)
        if actDCF < best_actDCF:
            best_actDCF, best_minDCF, best_prior = actDCF, minDCF, pT
            calibrated_scores, calibrated_labels = fold_scores, fold_labels

    print(f'Model: {model_name} - Best prior: {best_prior:.1f} - minDCF: {best_minDCF:.4f} - actDCF: {best_actDCF:.4f} - mis-calibration: {best_actDCF * 100 / best_minDCF - 100:.2f}%')
    return calibrated_scores, calibrated_labels

def train_and_evaluate_model(model_name, train_func, eval_func, DTR, LTR, DVAL, LVAL, plot=False):
    scores = eval_func(train_func(DTR, LTR), DVAL)
    calibrated_scores, calibrated_labels = calibrate_scores(model_name, scores, LVAL, np.linspace(0.1, 0.9, 9))

    if plot:
        plt.figure()
        logOdds, actDCF, minDCF = bayesPlot(scores, LVAL)
        plt.plot(logOdds, minDCF, color='C0', linestyle='--', label='minDCF (pre-cal.)')
        plt.plot(logOdds, actDCF, color='C0', linestyle=':', label='actDCF (pre-cal.)')
        
        logOdds, actDCF, minDCF = bayesPlot(calibrated_scores, calibrated_labels)
        plt.plot(logOdds, actDCF, color='C0', linestyle='-', label='actDCF (cal.)')
        plt.legend()
        plt.title(f'System - {model_name}')
        plt.ylim(0, 0.8)

    return scores, calibrated_scores, calibrated_labels

def main(plot=False):
    features, classes = loadDataset('/trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(features, classes)
    DTR, LTR = shuffle_data(DTR, LTR)
    DVAL, LVAL = shuffle_data(DVAL, LVAL)

    # Logistic Regression
    lr_train = lambda DTR, LTR: trainLogReg(expand_features(DTR, DVAL)[0], LTR, 0.03162)
    lr_eval = lambda model, DVAL: np.dot(model[0].T, expand_features(DTR, DVAL)[1]) + model[1] - np.log((LTR == 1).mean() / (1 - (LTR == 1).mean()) )
    lr_scores, lr_calib_scores, lr_labels = train_and_evaluate_model("Logistic Regression", lr_train, lr_eval, DTR, LTR, DVAL, LVAL, plot)

    # RBF Kernel SVM
    svm_train = lambda DTR, LTR: train_dual_SVM_kernel(DTR, LTR, 31.62, rbfKernel(np.exp(-2)), 1)
    svm_scores, svm_calib_scores, svm_labels = train_and_evaluate_model("RBF Kernel SVM", svm_train, lambda model, DVAL: model(DVAL), DTR, LTR, DVAL, LVAL, plot)

    # GMM
    def gmm_train(DTR, LTR):
        gmm_C0 = train_GMM_LBG_EM(DTR[:, LTR==0], 8, covType='diagonal', psiEig=0.01)
        gmm_C1 = train_GMM_LBG_EM(DTR[:, LTR==1], 32, covType='diagonal', psiEig=0.01)
        return (gmm_C0, gmm_C1)

    gmm_eval = lambda model, DVAL: logpdf_GMM(DVAL, model[1]) - logpdf_GMM(DVAL, model[0])
    gmm_scores, gmm_calib_scores, gmm_labels = train_and_evaluate_model("GMM", gmm_train, gmm_eval, DTR, LTR, DVAL, LVAL, plot)

    # Fusion
    def fuse_scores(scores_list, labels, priors):
        best_minDCF, best_actDCF, best_prior = float('inf'), float('inf'), 0
        best_fused_scores, best_fused_labels = None, None

        for pT in priors:
            fused_scores, fused_labels = [], []
            for fold in range(KFOLD):
                SCAL = np.vstack([extract_train_val_folds(scores, fold)[0] for scores in scores_list])
                SVAL = np.vstack([extract_train_val_folds(scores, fold)[1] for scores in scores_list])
                LCAL, LVAL = extract_train_val_folds(labels, fold)

                w, b = trainWeightedLogRegBinary(SCAL, LCAL, 0, pT)
                calibrated_SVAL = (w.T @ SVAL + b - np.log(pT / (1-pT))).ravel()

                fused_scores.append(calibrated_SVAL)
                fused_labels.append(LVAL)

            fused_scores = np.hstack(fused_scores)
            fused_labels = np.hstack(fused_labels)

            minDCF, actDCF = compute_minDCF_actDCF(fused_scores, fused_labels, TARGET_PRIOR)
            if actDCF < best_actDCF:
                best_actDCF, best_minDCF, best_prior = actDCF, minDCF, pT
                best_fused_scores, best_fused_labels = fused_scores, fused_labels

        print(f'Best prior Fusion: {best_prior:.1f} - minDCF: {best_minDCF:.4f} - actDCF: {best_actDCF:.4f}')
        print(f'Relative improvement: {best_actDCF * 100 / best_minDCF - 100:.2f}%')

        return best_fused_scores, best_fused_labels

    fused_scores, fused_labels = fuse_scores([lr_scores, svm_scores, gmm_scores], LVAL, np.linspace(0.1, 0.9, 9))

    if plot:
        plt.figure()
        plt.title('Fusion - validation')
        for scores, labels, name, color in zip([lr_calib_scores, svm_calib_scores, gmm_calib_scores, fused_scores],
                                               [lr_labels, svm_labels, gmm_labels, fused_labels],
                                               ['LR', 'SVM', 'GMM', 'LR + SVM + GMM'],
                                               ['C0', 'C1', 'C2', 'C3']):
            logOdds, actDCF, minDCF = bayesPlot(scores, labels)
            plt.plot(logOdds, minDCF, color=color, linestyle='--', label=f'{name} - minDCF')
            plt.plot(logOdds, actDCF, color=color, linestyle='-', label=f'{name} - actDCF')
        plt.ylim(0.0, 0.8)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main(plot=True)