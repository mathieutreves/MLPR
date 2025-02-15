from utils import *

def plot_scatter(data, classes, feature_start):
    feature_names = [f"Feature {i}" for i in range(1, 7)]
    data_by_class = [data[:, classes == i] for i in range(2)]
    
    fig, axs = plt.subplots(1, 4, figsize=(25, 5))
    
    for i, ax in enumerate(axs):
        f1, f2 = feature_start + i % 2, feature_start + (i + 1) % 2
        
        if i in [0, 2]:
            for j, (d, label) in enumerate(zip(data_by_class, ['Fake', 'Genuine'])):
                ax.hist(d[f1, :], bins=10, density=True, alpha=0.4, label=label)
            ax.set_title(feature_names[f1])
            ax.legend()
        else:
            for j, (d, label) in enumerate(zip(data_by_class, ['Fake', 'Genuine'])):
                ax.scatter(d[f1, :], d[f2, :], label=label, s=5, alpha=0.4)
            ax.set_xlabel(feature_names[f1])
            ax.set_ylabel(feature_names[f2])
            ax.legend()
    
    plt.tight_layout()
    plt.show()

def compute_statistics(data, classes, feature_start):
    data_by_class = [data[:, classes == i] for i in range(2)]
    stats = []
    
    for i, d in enumerate(data_by_class):
        mean = d.mean(1)
        var = d.var(1)
        stats.append((mean, var))
    
    print(f'Features: {feature_start+1}-{feature_start+2}')
    for i, (mean, var) in enumerate(stats):
        print(f'Class C{i}')
        for j in range(2):
            f = feature_start + j
            print(f'Feature {f+1}: Mean: {mean[f]:.4f} - Variance: {var[f]:.4f}')

def analyze_features(data, classes, feature_start):
    compute_statistics(data, classes, feature_start)
    plot_scatter(data, classes, feature_start)

def main():
    features, classes = loadDataset('/trainData.txt')
    
    for start in range(0, 5, 2):
        analyze_features(features, classes, start)

if __name__ == "__main__":
    main()