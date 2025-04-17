import matplotlib.pyplot as plt
import numpy as np

def plot_pr(precision, recall, label: str, out):
    plt.figure()
    plt.plot(recall, precision, label=label)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend(); plt.tight_layout(); plt.savefig(out)