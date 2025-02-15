from utils import *

def compute_gaussian_parameters(D):
    mu = vcol(D.mean(1))
    C = ((D - mu) @ (D - mu).T) / D.shape[1]
    return mu, C

def plot_feature_histogram_and_gaussian(ax, DTR_0, DTR_1, feature_index):
    m_ML_0, C_ML_0 = compute_gaussian_parameters(DTR_0)
    m_ML_1, C_ML_1 = compute_gaussian_parameters(DTR_1)

    x_plot = np.linspace(-8, 12, 1000)
    
    ax.hist(DTR_0.ravel(), bins=50, density=True, alpha=0.4, label="Fake")
    ax.hist(DTR_1.ravel(), bins=50, density=True, alpha=0.4, label="Genuine")
    ax.plot(x_plot, np.exp(logpdf_GAU_ND(vrow(x_plot), m_ML_0, C_ML_0)), label="Fake (Gaussian)")
    ax.plot(x_plot, np.exp(logpdf_GAU_ND(vrow(x_plot), m_ML_1, C_ML_1)), label="Genuine (Gaussian)")

    ax.set_xlim(-5, 5)
    ax.set_title(f'Feature {feature_index + 1}')
    ax.legend()

def plot_all_features(DTR, LTR):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()

    for i in range(6):
        DTR_0 = DTR[i:i+1, LTR == 0]
        DTR_1 = DTR[i:i+1, LTR == 1]
        plot_feature_histogram_and_gaussian(axs[i], DTR_0, DTR_1, i)

    plt.tight_layout()
    plt.show()

def main():
    features, classes = loadDataset('/trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(features, classes)
    plot_all_features(DTR, LTR)

if __name__ == "__main__":
    main()