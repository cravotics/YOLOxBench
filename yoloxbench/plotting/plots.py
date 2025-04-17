from pathlib import Path
import matplotlib.pyplot as plt

def save_bar(df, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for col in [c for c in df.columns if c.startswith("metrics/mAP")]:
        plt.figure()
        df.plot.bar(x="run", y=col, legend=False)
        plt.title(col)
        plt.tight_layout()
        plt.savefig(out / f"{col.replace('/', '_')}.png")
        plt.close()