# %%
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import itk
from dataclasses import dataclass, field
import numpy as np

# Reading data
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
# Label formats, in priority order when a case has more than one. .mhd is the
# MetaImage header; its .zraw payload is read through the header, never directly.
# Trailing comma matters: (".mhd") is a str, and iterating it globs for "." etc.
LABEL_EXTS = (".mhd",)
DATASET = Path(
    "/Users/gastoncrecikeinbaum/Library/CloudStorage/SynologyDrive-XDMD/DaVinci/Sources/Xyall")
DATA_FOLDER = DATASET / "Omniseq"
LABEL_FOLDER = Path(
    "/Users/gastoncrecikeinbaum/Library/CloudStorage/SynologyDrive-XDMD/Annotations/2025 Xyall/Omniseq")
# "<value>: <name>" per line, 0 = background
LABELS_FILE = DATASET / "labels.txt"
data_path = Path(DATA_FOLDER)

# Pixels sampled per image for intensity stats
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


CHANNELS = ("R", "G", "B")
# matplotlib names for the same channels
CHANNEL_COLORS = ("r", "g", "b")


@dataclass
class IntensityStats:
    """Per-channel intensity summary over one population of sampled pixels."""
    n_pixels: int                        # pixels sampled per channel
    intensity_min: np.ndarray            # per-channel, shape (3,)
    intensity_max: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    quartiles: np.ndarray                # rows [25%, 50%, 75%], shape (3, 3)
    # per-channel histogram, shape (3, nbins)
    intensity_counts: np.ndarray
    intensity_bins: np.ndarray           # shared edges, shape (nbins + 1,)

    def table(self) -> list[str]:
        rows = {
            "count": [self.n_pixels] * 3,
            "mean": self.mean,
            "std": self.std,
            "min": self.intensity_min,
            "5%": self.quartiles[0],
            "25%": self.quartiles[1],
            "50%": self.quartiles[2],
            "75%": self.quartiles[3],
            "95%": self.quartiles[4],
            "max": self.intensity_max,
        }
        lines = [f"{'':>6}" + "".join(f"{c:>12}" for c in CHANNELS)]
        for name, vals in rows.items():
            lines.append(f"{name:>6}" +
                         "".join(f"{float(v):>12.2f}" for v in vals))
        return lines


def _stats(vx, bin_edges=None) -> IntensityStats:
    """Summarize an (n_pixels, 3) sample. Reuse bin_edges to compare populations."""
    if bin_edges is None:
        # Shared bin edges so the three channel histograms are comparable
        bin_edges = np.histogram_bin_edges(vx, bins=100)
    return IntensityStats(
        n_pixels=len(vx),
        intensity_min=vx.min(axis=0),
        intensity_max=vx.max(axis=0),
        mean=vx.mean(axis=0),
        std=vx.std(axis=0),
        quartiles=np.percentile(vx, [5, 25, 50, 75, 95], axis=0),
        intensity_counts=np.stack([np.histogram(vx[:, c], bins=bin_edges)[0]
                                   for c in range(3)]),
        intensity_bins=bin_edges,
    )


@dataclass
class DatasetFingerprint:
    n_images: int
    sizes: list[list[int]]
    channel_counts: dict[int, int]       # {n_channels: n_images}
    shape_min: np.ndarray                # (H, W) spatial extremes / median
    shape_median: np.ndarray
    shape_max: np.ndarray
    all: IntensityStats                  # every pixel
    # pixels under a labelled class; None when no labels were paired
    foreground: IntensityStats | None
    n_labelled: int = 0                  # images that had a matching label
    label_map: dict[int, str] = field(default_factory=dict)
    class_pixels: dict[int, int] = field(default_factory=dict)

    def describe(self) -> None:
        """Print the full report: dataset meta + per-channel intensity tables."""
        def shp(a): return list(map(int, a))
        lines = [
            "Dataset fingerprint",
            "=" * 42,
            f"images         : {self.n_images}",
            f"labelled       : {self.n_labelled}",
            f"channel counts : {self.channel_counts}",
            f"shape (H, W)   : min {shp(self.shape_min)}  "
            f"median {shp(self.shape_median)}  max {shp(self.shape_max)}",
            "",
            "All pixels (sampled)",
            "-" * 42,
            *self.all.table(),
        ]

        if self.foreground is not None:
            lines += ["",
                      "Foreground pixels (labelled classes, sampled)",
                      "-" * 42,
                      *self.foreground.table()]
            if self.class_pixels:
                total = sum(self.class_pixels.values())
                lines += ["", "Class balance within foreground", "-" * 42]
                for value, count in sorted(self.class_pixels.items(),
                                           key=lambda kv: -kv[1]):
                    name = self.label_map.get(value, "?")
                    lines.append(f"{value:>3} {name:<12}{count:>14,}"
                                 f"{count / total:>8.1%}")
        else:
            lines += ["", "Foreground pixels : no labels found"]

        print("\n".join(lines))

    def _plot_intensities(self, ax, stats, title):
        for c, (name, color) in enumerate(zip(CHANNELS, CHANNEL_COLORS)):
            ax.stairs(stats.intensity_counts[c], stats.intensity_bins,
                      color=color, alpha=0.3, fill=True)
            ax.stairs(stats.intensity_counts[c], stats.intensity_bins,
                      color=color, lw=1.5,
                      label=f"{name} (mean {stats.mean[c]:.1f})")
            ax.axvline(stats.mean[c], color=color, ls="--", alpha=0.5)
        ax.set(xlabel="Intensity", ylabel="Pixel count", title=title)
        ax.legend(fontsize=8)

    def plot_summary(self):
        # Separate axes for all vs foreground: they differ by orders of magnitude
        # in pixel count, so each keeps its own y scale and stays a true count.
        has_fg = self.foreground is not None
        fig, axes = plt.subplots(1, 3 if has_fg else 2,
                                 figsize=(18 if has_fg else 13, 5))
        ax_all, *rest = axes
        self._plot_intensities(ax_all, self.all, "All pixels")
        if has_fg:
            ax_fg = rest.pop(0)
            self._plot_intensities(ax_fg, self.foreground, "Foreground pixels")
            # Same x range on both so the distributions stay comparable by eye
            lo = min(ax_all.get_xlim()[0], ax_fg.get_xlim()[0])
            hi = max(ax_all.get_xlim()[1], ax_fg.get_xlim()[1])
            ax_all.set_xlim(lo, hi)
            ax_fg.set_xlim(lo, hi)
        ax_size = rest[0]

        # right: size distribution
        shape_counts = Counter(tuple(s) for s in self.sizes)
        labels = [str(list(s)) for s in shape_counts]
        ax_size.bar(range(len(shape_counts)), list(shape_counts.values()))
        ax_size.set_xticks(range(len(shape_counts)))
        ax_size.set_xticklabels(labels, rotation=45, ha="right")
        ax_size.set(xlabel="Image shape", ylabel="Count",
                    title=f"Size distribution ({len(shape_counts)} unique)")

        fig.tight_layout()
        plt.show()


def _subsample(vx):
    if len(vx) > SAMPLE_PER_IMAGE:          # subsample large images
        return vx[_rng.integers(0, len(vx), SAMPLE_PER_IMAGE)]
    return vx


def read_label_map(labels_file) -> dict[int, str]:
    """Parse "<value>: <name>" lines. 0 is background and stays out of the map."""
    label_map = {}
    for line in Path(labels_file).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        value, name = line.split(":", 1)
        label_map[int(value)] = name.strip()
    return label_map


def label_index(label_folder) -> dict[str, Path]:
    """Map stem -> label file. Labels carry their image's name, extension aside,
    but live in their own per-case subfolders, so match on stem rather than path.
    Exact stem matching also keeps siblings like `<case>-0-overlay.jpg` out."""
    index = {}
    for ext in LABEL_EXTS:                  # earlier extensions win
        for file in Path(label_folder).rglob(f"*{ext}"):
            index.setdefault(file.stem, file)
    return index


def fingerprint(data_folder, label_folder=None,
                labels_file=None) -> DatasetFingerprint:
    label_map = read_label_map(labels_file) if labels_file else {}
    labels = label_index(label_folder) if label_folder else {}
    n_images = 0
    n_labelled = 0
    sizes = []
    channels = []
    voxels = []                             # each entry is (n_pixels, 3)
    fg_voxels = []
    # {label value: n_pixels}, unsampled
    class_pixels = Counter()
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
        rgb = _to_rgb(img_array)
        voxels.append(_subsample(rgb.reshape(-1, 3)))
        n_images += 1

        label_file = labels.get(file.stem)
        if label_file is None:
            continue
        try:
            seg = _normalize_format(itk.imread(str(label_file)))
        except Exception as err:
            print(f"skipping label {label_file.name}: {err}")
            continue
        if seg.ndim == 3:                   # label stored with channels: take one
            seg = seg[..., 0]
        if seg.shape != rgb.shape[:2]:
            print(f"skipping label {label_file.name}: shape {tuple(seg.shape)} "
                  f"!= image {rgb.shape[:2]}")
            continue

        # Foreground = the values named in labels.txt, so anything unmapped
        # (background, stray values) is excluded rather than assumed foreground
        mask = np.isin(seg, list(label_map)) if label_map else seg > 0
        n_labelled += 1
        for value, count in zip(*np.unique(seg[mask], return_counts=True)):
            class_pixels[int(value)] += int(count)
        fg = rgb[mask]                      # (n_foreground, 3)
        if len(fg):
            fg_voxels.append(_subsample(fg))

    if not voxels:
        raise ValueError(f"No readable images found under {data_folder}")

    all_vx = np.concatenate(voxels)         # (total_voxels, 3)
    all_stats = _stats(all_vx)
    # Reuse the all-pixel bin edges so the two histograms line up
    fg_stats = (_stats(np.concatenate(fg_voxels), all_stats.intensity_bins)
                if fg_voxels else None)

    # Spatial size stats (first two dims = H, W)
    hw = np.array([s[:2] for s in sizes])

    return DatasetFingerprint(n_images=n_images,
                              sizes=sizes,
                              channel_counts=dict(Counter(channels)),
                              shape_min=hw.min(axis=0),
                              shape_median=np.median(hw, axis=0).astype(int),
                              shape_max=hw.max(axis=0),
                              all=all_stats,
                              foreground=fg_stats,
                              n_labelled=n_labelled,
                              label_map=label_map,
                              class_pixels=dict(class_pixels))


# %%
fp = fingerprint(DATA_FOLDER, LABEL_FOLDER, LABELS_FILE)
# %%
fp.describe()
fp.plot_summary()
# %%
