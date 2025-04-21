from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def save_bar(df: pd.DataFrame, out: Path, metrics):
    out.mkdir(parents=True, exist_ok=True)
    for col in metrics:
        if col not in df.columns:
            continue
        plt.figure()
        df.plot.bar(x="run", y=col, legend=False)
        plt.title(col)
        plt.tight_layout()
        plt.savefig(out / f"{col.replace('/', '_')}.png")
        plt.close()