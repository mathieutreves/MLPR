from utils import *

def plot_histograms(data, labels, title, num_rows, num_cols):
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]
    
    for i, ax in enumerate(axs):
        if i < data.shape[0]:
            for j, label in enumerate(['Fake', 'Genuine']):
                ax.hist(data[i, labels == j], bins=10, density=True, alpha=0.4, label=label)
            ax.set_title(f"{title} ({i+1}st direction)")
            ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_comparison(DTR, DVAL, LTR, LVAL, title):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    for i, (data, labels, subtitle) in enumerate([(DTR, LTR, "Model training set (DTR, LTR)"),
                                                  (DVAL, LVAL, "Model validation set (DVAL, LVAL)")]):
        for j, label in enumerate(['Fake', 'Genuine']):
            axs[i].hist(data[0, labels == j], bins=5, density=True, alpha=0.4, label=label)
        axs[i].set_title(subtitle)
        axs[i].legend()
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def perform_pca_analysis(features, classes):
    PCA_features = PCA(features, 6)
    plot_histograms(PCA_features, classes, "PCA", 2, 3)

def perform_lda_analysis(features, classes):
    LDA_features = LDA(features, classes, 1)
    plot_histograms(LDA_features, classes, "LDA", 1, 1)

def perform_split_analysis(features, classes):
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(features, classes)
    
    # LDA on training set
    W = LDA(DTR, LTR, m=1, project=False)
    DTR_lda = np.dot(W.T, DTR)
    DVAL_lda = np.dot(W.T, DVAL)
    
    plot_comparison(DTR_lda, DVAL_lda, LTR, LVAL, "LDA Comparison")
    calculateThresh(DTR_lda, LTR, DVAL_lda, LVAL)
    
    # PCA then LDA
    U_pca = PCA(DTR, m=2, project=False)
    DTR_pca = np.dot(U_pca.T, DTR)
    DVAL_pca = np.dot(U_pca.T, DVAL)
    
    U_lda = LDA(DTR_pca, LTR, m=1, project=False)
    DTR_lda = np.dot(U_lda.T, DTR_pca)
    DVAL_lda = np.dot(U_lda.T, DVAL_pca)
    
    plot_comparison(DTR_pca, DVAL_pca, LTR, LVAL, "PCA then LDA Comparison")
    calculateThresh(DTR_lda, LTR, DVAL_lda, LVAL)
    
def main():
    features, classes = loadDataset('/trainData.txt')

    perform_pca_analysis(features, classes)
    perform_lda_analysis(features, classes)
    perform_split_analysis(features, classes)

if __name__ == "__main__":
    main()