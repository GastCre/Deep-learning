# %%
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import itk
from dataclasses import dataclass
import numpy as np

# Reading data
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DATA_FOLDER = "/Users/gastoncrecikeinbaum/Library/CloudStorage/SynologyDrive-XDMD/Sources/2025 Xyall pilot project"
data_path = Path(DATA_FOLDER)

# Pixels sampled per image for intensity stats (full 4K images are overkill)
SAMPLE_PER_IMAGE = 50_000
_rng = np.random.default_rng(0)

# %% Data features


def _normalize_format(image):
    if isinstance(image, itk.Image):
        return itk.array_from_image(image)
    else:
        raise TypeError("Image format is not array or itk.image")


def _to_rgb(arr):
    """Standardize any image array to (H, W, 3): expand grayscale, drop alpha."""
    if arr.ndim == 2:                       # grayscale -> replicate to 3 channels
        return np.repeat(arr[..., None], 3, axis=-1)
    return arr[..., :3]                     # RGB stays, RGBA drops alpha


CHANNELS = ("red", "green", "blue")


@dataclass
class DatasetFingerprint:
    n_images: int
    sizes: list[list[int]]
    channel_counts: dict[int, int]       # {n_channels: n_images}
    shape_min: np.ndarray                # (H, W) spatial extremes / median
    shape_median: np.ndarray
    shape_max: np.ndarray
    # pixels sampled per channel (the "count")
    n_pixels: int
    intensity_min: np.ndarray            # per-channel, shape (3,)
    intensity_max: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    quartiles: np.ndarray                # rows [25%, 50%, 75%], shape (3, 3)
    # per-channel histogram, shape (3, nbins)
    intensity_counts: np.ndarray
    intensity_bins: np.ndarray           # shared edges, shape (nbins + 1,)

    def describe(self) -> str:
        """Full text report: dataset meta + per-channel intensity table."""
        def shp(a): return list(map(int, a))
        rows = {
            "count": [self.n_pixels] * 3,
            "mean": self.mean,
            "std": self.std,
            "min": self.intensity_min,
            "25%": self.quartiles[0],
            "50%": self.quartiles[1],
            "75%": self.quartiles[2],
            "max": self.intensity_max,
        }

        meta = [
            "Dataset fingerprint",
            "=" * 40,
            f"images         : {self.n_images}",
            f"channel counts : {self.channel_counts}",
            f"shape (H, W)   : min {shp(self.shape_min)}  "
            f"median {shp(self.shape_median)}  max {shp(self.shape_max)}",
            "",
            "Per-channel intensities (sampled)",
            "-" * 40,
        ]

        header = f"{'':>6}" + "".join(f"{c:>12}" for c in CHANNELS)
        table = [header]
        for name, vals in rows.items():
            table.append(f"{name:>6}" +
                         "".join(f"{float(v):>12.2f}" for v in vals))

        return "\n".join(meta + table)

    def plot_summary(self):
        fig, (ax_hist, ax_size) = plt.subplots(1, 2, figsize=(13, 5))

        # left: per-channel intensity distribution
        for c, color in enumerate(CHANNELS):
            ax_hist.stairs(self.intensity_counts[c], self.intensity_bins,
                           color=color, label=f"{color} (mean {self.mean[c]:.1f})")
            ax_hist.axvline(self.mean[c], color=color, ls="--", alpha=0.5)
        ax_hist.set(xlabel="Intensity", ylabel="Pixel count",
                    title="Per-channel intensity distribution")
        ax_hist.legend()

        # middle: size distribution
        shape_counts = Counter(tuple(s) for s in self.sizes)
        labels = [str(list(s)) for s in shape_counts]
        ax_size.bar(range(len(shape_counts)), list(shape_counts.values()))
        ax_size.set_xticks(range(len(shape_counts)))
        ax_size.set_xticklabels(labels, rotation=45, ha="right")
        ax_size.set(xlabel="Image shape", ylabel="Count",
                    title=f"Size distribution ({len(shape_counts)} unique)")

        fig.tight_layout()
        plt.show()


def fingerprint(data_folder) -> DatasetFingerprint:
    n_images = 0
    sizes = []
    channels = []
    voxels = []                             # each entry is (n_pixels, 3)
    for file in Path(data_folder).rglob("*"):
        if file.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            img_array = _normalize_format(itk.imread(str(file)))
        except Exception as err:            # skip unreadable/corrupt images
            print(f"skipping {file.name}: {err}")
            continue
        sizes.append(list(img_array.shape))
        channels.append(img_array.shape[-1] if img_array.ndim == 3 else 1)
        vx = _to_rgb(img_array).reshape(-1, 3)
        if len(vx) > SAMPLE_PER_IMAGE:      # subsample large images
            vx = vx[_rng.integers(0, len(vx), SAMPLE_PER_IMAGE)]
        voxels.append(vx)
        n_images += 1

    if not voxels:
        raise ValueError(f"No readable images found under {data_folder}")

    all_vx = np.concatenate(voxels)         # (total_voxels, 3)

    # Per-channel intensity stats
    int_min = all_vx.min(axis=0)
    int_max = all_vx.max(axis=0)
    mean = all_vx.mean(axis=0)
    std = all_vx.std(axis=0)
    quartiles = np.percentile(all_vx, [25, 50, 75], axis=0)
    # Shared bin edges so the three channel histograms are comparable
    bin_edges = np.histogram_bin_edges(all_vx, bins=100)
    counts = np.stack([np.histogram(all_vx[:, c], bins=bin_edges)[0]
                       for c in range(3)])

    # Spatial size stats (first two dims = H, W)
    hw = np.array([s[:2] for s in sizes])

    return DatasetFingerprint(n_images=n_images,
                              sizes=sizes,
                              channel_counts=dict(Counter(channels)),
                              shape_min=hw.min(axis=0),
                              shape_median=np.median(hw, axis=0).astype(int),
                              shape_max=hw.max(axis=0),
                              n_pixels=len(all_vx),
                              intensity_min=int_min,
                              intensity_max=int_max,
                              mean=mean,
                              std=std,
                              quartiles=quartiles,
                              intensity_counts=counts,
                              intensity_bins=bin_edges)


# %% Trial
fp = fingerprint(DATA_FOLDER)
print(fp.describe())
fp.plot_summary()

# %%
