import numpy as np
import matplotlib.pyplot as plt
from utils import loadDataset, split_db_2to1, vrow
from project_07 import compute_minDCF, compute_actDCF
from project_08 import trainLogReg, expand_features, trainWeightedLogRegBinary
from project_09 import train_dual_SVM_kernel, rbfKernel
from project_10 import train_GMM_LBG_EM, logpdf_GMM
from project_11 import bayesPlot, shuffle_data

TARGET_PRIOR = 0.1

def load_and_preprocess_data():
    features, classes = loadDataset('/trainData.txt')
    evalFeatures, eval_labels = loadDataset('/evalData.txt')

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(features, classes)

    DTR, LTR = shuffle_data(DTR, LTR)
    DVAL, LVAL = shuffle_data(DVAL, LVAL)

    return DTR, LTR, DVAL, LVAL, evalFeatures, eval_labels

def logOdds(L):
    pEmp = (L == 1).mean()
    return np.log(pEmp / (1 - pEmp))

def train_logistic_regression(DTR, LTR, DVAL, l):
    DTR_expanded, DVAL_expanded = expand_features(DTR, DVAL)
    w, b = trainLogReg(DTR_expanded, LTR, l)
    scores = np.dot(w.T, DVAL_expanded) + b - logOdds(LTR)
    return w, b, scores

def train_rbf_kernel_svm(DTR, LTR, DVAL, C, gamma, eps):
    fScore = train_dual_SVM_kernel(DTR, LTR, C, rbfKernel(gamma), eps)
    scores = fScore(DVAL)
    return fScore, scores

def train_gmm(DTR, LTR, DVAL, components_c0, components_c1):
    gmm_C0 = train_GMM_LBG_EM(DTR[:, LTR==0], components_c0, covType='diagonal', psiEig=0.01)
    gmm_C1 = train_GMM_LBG_EM(DTR[:, LTR==1], components_c1, covType='diagonal', psiEig=0.01)
    scores = logpdf_GMM(DVAL, gmm_C1) - logpdf_GMM(DVAL, gmm_C0)
    return gmm_C0, gmm_C1, scores

def evaluate_model(scores, labels, p_target, Cfn_target, Cfp_target, model_name):
    minDCF = compute_minDCF(scores, labels, p_target, Cfn_target, Cfp_target)
    actDCF = compute_actDCF(scores, labels, p_target, Cfn_target, Cfp_target)
    print(f'Model {model_name} \t- minDCF: {minDCF:.4f} - actDCF: {actDCF:.4f} - mis-calibration: {actDCF * 100 / minDCF - 100:.2f}%')
    return minDCF, actDCF

def fusion_models(scores_list, LVAL, pT_fusion):
    SMatrix = np.vstack(scores_list)
    w_fusion, b_fusion = trainWeightedLogRegBinary(SMatrix, LVAL, 0, pT_fusion)
    return w_fusion, b_fusion

def apply_fusion(w_fusion, b_fusion, scores_list, pT_fusion):
    SMatrixEval = np.vstack(scores_list)
    fused_scores = (w_fusion.T @ SMatrixEval + b_fusion - np.log(pT_fusion / (1 - pT_fusion))).ravel()
    return fused_scores

def plot_bayes_error(scores_dict, labels):
    plt.figure(figsize=(12, 8))
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    
    for i, (model_name, scores) in enumerate(scores_dict.items()):
        logOdds, actDCF, minDCF = bayesPlot(scores, labels)
        plt.plot(logOdds, minDCF, color=colors[i], linestyle='--', label=f'{model_name} - minDCF')
        plt.plot(logOdds, actDCF, color=colors[i], linestyle='-', label=f'{model_name} - actDCF')
    
    plt.title('Model Comparison - Bayes Error Plot')
    plt.xlabel('Log Odds')
    plt.ylabel('DCF')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    DTR, LTR, DVAL, LVAL, evalFeatures, eval_labels = load_and_preprocess_data()

    p_target, Cfn_target, Cfp_target = 0.1, 1.0, 1.0
    
    # Train models
    w_lr, b_lr, scores_sys_1 = train_logistic_regression(DTR, LTR, DVAL, l=0.03162)
    fScore_svm, scores_sys_2 = train_rbf_kernel_svm(DTR, LTR, DVAL, C=31.62, gamma=np.exp(-2), eps=1)
    gmm_C0, gmm_C1, scores_sys_3 = train_gmm(DTR, LTR, DVAL, components_c0=8, components_c1=32)

    # Evaluate models on evaluation set
    eval_scores_sys_1 = np.dot(w_lr.T, expand_features(DTR, evalFeatures)[1]) + b_lr - logOdds(eval_labels)
    eval_scores_sys_2 = fScore_svm(evalFeatures)
    eval_scores_sys_3 = logpdf_GMM(evalFeatures, gmm_C1) - logpdf_GMM(evalFeatures, gmm_C0)

    # Calibrate scores
    pT_LR, pT_SVM, pT_GMM = 0.7, 0.2, 0.9
    w, b = trainWeightedLogRegBinary(vrow(scores_sys_1), LVAL, 0, pT_LR)
    calibrated_eval_scores_sys_1 = (w.T @ vrow(eval_scores_sys_1) + b - np.log(pT_LR / (1 - pT_LR))).ravel()
    
    w, b = trainWeightedLogRegBinary(vrow(scores_sys_2), LVAL, 0, pT_SVM)
    calibrated_eval_scores_sys_2 = (w.T @ vrow(eval_scores_sys_2) + b - np.log(pT_SVM / (1 - pT_SVM))).ravel()
    
    w, b = trainWeightedLogRegBinary(vrow(scores_sys_3), LVAL, 0, pT_GMM)
    calibrated_eval_scores_sys_3 = (w.T @ vrow(eval_scores_sys_3) + b - np.log(pT_GMM / (1 - pT_GMM))).ravel()

    # Evaluate individual models
    evaluate_model(calibrated_eval_scores_sys_1, eval_labels, p_target, Cfn_target, Cfp_target, "LR")
    evaluate_model(calibrated_eval_scores_sys_2, eval_labels, p_target, Cfn_target, Cfp_target, "SVM")
    evaluate_model(calibrated_eval_scores_sys_3, eval_labels, p_target, Cfn_target, Cfp_target, "GMM")

    # Fusion of all models
    pT_fusion = 0.1
    w_fusion, b_fusion = fusion_models([scores_sys_1, scores_sys_2, scores_sys_3], LVAL, pT_fusion)
    fused123_eval_scores = apply_fusion(w_fusion, b_fusion, [eval_scores_sys_1, eval_scores_sys_2, eval_scores_sys_3], pT_fusion)
    evaluate_model(fused123_eval_scores, eval_labels, p_target, Cfn_target, Cfp_target, "Fusion 1-2-3")

    # Different fusion models
    fusion_combinations = [
        ("1-2-3", [scores_sys_1, scores_sys_2, scores_sys_3], [eval_scores_sys_1, eval_scores_sys_2, eval_scores_sys_3], 0.1),
        ("1-2", [scores_sys_1, scores_sys_2], [eval_scores_sys_1, eval_scores_sys_2], 0.9),
        ("2-3", [scores_sys_2, scores_sys_3], [eval_scores_sys_2, eval_scores_sys_3], 0.7),
        ("1-3", [scores_sys_1, scores_sys_3], [eval_scores_sys_1, eval_scores_sys_3], 0.1)
    ]

    for name, fusion_score_functions, fusion_eval_score_functions, pT_fusion in fusion_combinations:        
        w_fusion, b_fusion = fusion_models(fusion_score_functions, LVAL, pT_fusion)
        fused_eval_scores = apply_fusion(w_fusion, b_fusion, fusion_eval_score_functions, pT_fusion)
        evaluate_model(fused_eval_scores, eval_labels, p_target, Cfn_target, Cfp_target, f"Fusion {name}")

    # Plot Bayes error
    scores_dict = {
        "LR": calibrated_eval_scores_sys_1,
        "SVM": calibrated_eval_scores_sys_2,
        "GMM": calibrated_eval_scores_sys_3,
        "Fusion 1-2-3": fused123_eval_scores
    }
    plot_bayes_error(scores_dict, eval_labels)
    
    # Try different hyperparameter for LR
    # ------------------------------
    lambdas = np.logspace(-4, 2, 13)
    best_minDCF, best_actDCF, best_lambda, best_model = float('inf'), float('inf'), 0, ""

    for l in lambdas:
        print()
        print(f'lambda: {l:.4f}')

        # Expanded features
        w, b = trainLogReg(expand_features(DTR, DVAL)[0], LTR, l) 

        _, N = evalFeatures.shape
        quad_terms = np.array([np.outer(evalFeatures[:, i], evalFeatures[:, i]).reshape(-1) for i in range(N)]).T
        evalFeatures_expanded = np.vstack([quad_terms, evalFeatures])

        eval_adjusted_scores = np.dot(w.T, evalFeatures_expanded) + b - logOdds(eval_labels)
        minDCF, actDCF = evaluate_model(eval_adjusted_scores, eval_labels, p_target, Cfn_target, Cfp_target, "QLR EF")
        
        if (minDCF < best_minDCF):
            best_actDCF, best_minDCF, best_lambda, best_model = actDCF, minDCF, l, "QLR"

        # Prior-weighted
        w, b = trainWeightedLogRegBinary(DTR, LTR, l, pT = TARGET_PRIOR)        
        eval_scores_sys_1 = np.dot(w.T, evalFeatures) + b
        eval_adjusted_scores = eval_scores_sys_1 - logOdds(eval_labels)
        minDCF, actDCF = evaluate_model(eval_adjusted_scores, eval_labels, p_target, Cfn_target, Cfp_target, "Weight-Prior")
        
        if (minDCF < best_minDCF):
            best_actDCF, best_minDCF, best_lambda, best_model = actDCF, minDCF, l, "WP"

        # Linear
        w, b = trainLogReg(DTR, LTR, l)        
        eval_scores_sys_1 = np.dot(w.T, evalFeatures) + b
        eval_adjusted_scores = eval_scores_sys_1 - logOdds(LTR)
        minDCF, actDCF = evaluate_model(eval_adjusted_scores, eval_labels, p_target, Cfn_target, Cfp_target, "Linear")
        
        if (minDCF < best_minDCF):
            best_actDCF, best_minDCF, best_lambda, best_model = actDCF, minDCF, l, "LR"

    print(f'Best lambda: {best_lambda:.4f}, best model: {best_model} - minDCF: {best_minDCF:.4f} - actDCF: {best_actDCF:.4f}')


if __name__ == "__main__":
    main()