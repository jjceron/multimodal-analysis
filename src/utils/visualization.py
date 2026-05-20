from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_fold_training_history(
    history_csv: str | Path,
    output_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    prefix: str = "loss",
) -> list[Path]:
    """
    Genera un PNG separado por fold.

    Espera columnas:
        epoch, fold, train_loss, val_loss

    También acepta:
        val_epoch_loss

    Si hay split_seed e init_seed, los usa en el nombre del archivo.
    """

    history_csv = Path(history_csv)

    if output_dir is None:
        if output_path is None:
            output_dir = history_csv.parent / "plots"
        else:
            output_path = Path(output_path)

            # Compatibilidad con código viejo que pasaba un archivo .png
            if output_path.suffix:
                output_dir = output_path.parent
            else:
                output_dir = output_path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(history_csv)

    if "fold" not in df.columns:
        df["fold"] = 1

    if "val_loss" in df.columns:
        val_col = "val_loss"
    elif "val_epoch_loss" in df.columns:
        val_col = "val_epoch_loss"
    else:
        raise ValueError(
            "No encontré columna de validación. "
            "Debe existir 'val_loss' o 'val_epoch_loss'."
        )

    required_cols = {"epoch", "fold", "train_loss", val_col}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Faltan columnas en {history_csv}: {sorted(missing)}")

    group_cols = []

    if "split_seed" in df.columns:
        group_cols.append("split_seed")

    if "init_seed" in df.columns:
        group_cols.append("init_seed")

    group_cols.append("fold")

    saved_paths: list[Path] = []

    for group_key, group_df in df.groupby(group_cols):
        group_df = group_df.sort_values("epoch")

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_info = dict(zip(group_cols, group_key))

        fold = int(group_info["fold"])

        name_parts = [prefix]

        if "split_seed" in group_info:
            name_parts.append(f"splitseed_{int(group_info['split_seed'])}")

        if "init_seed" in group_info:
            name_parts.append(f"initseed_{int(group_info['init_seed'])}")

        name_parts.append(f"fold_{fold:02d}")

        output_file = output_dir / ("_".join(name_parts) + ".png")

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            group_df["epoch"],
            group_df["train_loss"],
            marker="o",
            linestyle="-",
            label="Train loss",
        )

        ax.plot(
            group_df["epoch"],
            group_df[val_col],
            marker="s",
            linestyle="--",
            label="Validation loss",
        )

        title = f"Fold {fold} - Training / validation loss"

        if "split_seed" in group_info and "init_seed" in group_info:
            title += (
                f" | split_seed={int(group_info['split_seed'])}"
                f" | init_seed={int(group_info['init_seed'])}"
            )

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        saved_paths.append(output_file)

    print("\nSaved fold plots:")
    for path in saved_paths:
        print(f"  {path}")

    return saved_paths