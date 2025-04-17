from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable
from ..plotting.plots import save_bar


def compare(run_dirs: Iterable[Path], out: Path):
    dfs = []
    for rd in run_dirs:
        csv = Path(rd) / "results.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["run"] = rd.name
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No results.csv found in given runs")
    df_all = pd.concat(dfs)
    save_bar(df_all, out)