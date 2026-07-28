# %%
from pathlib import Path
from collections import Counter, defaultdict
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
DATA_FOLDER = DATASET / "AZ"
LABEL_FOLDER = Path(
    "/Users/gastoncrecikeinbaum/Library/CloudStorage/SynologyDrive-XDMD/Annotations/2025 Xyall/AZ")
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


def _number_of_channels(arr):
    """Return the number of channels in an image array, or None if unknown.
    A replicated RGB array (R == G == B) is treated as a single channel, so
    grayscale saved as RGB groups with true grayscale rather than on its own."""
    if arr.ndim == 2:
        return 1
    elif arr.ndim == 3:
        # Check if channels are repeated (grayscale) or not (RGB).
        if np.all(arr[..., 0] == arr[..., 1]) and np.all(arr[..., 0] == arr[..., 2]):
            return 1
        else:
            return arr.shape[-1]
    elif arr.ndim == 4:
        # If alpha channel is present, ignore it and check the first three channels.
        if np.all(arr[..., 0] == arr[..., 1]) and np.all(arr[..., 0] == arr[..., 2]):
            return 1
        else:
            return arr.shape[-1]-1  # Exclude alpha channel
    else:
        return None


def _pixels(arr, n_channels):
    """Flatten an image to (n_pixels, n_channels). Grayscale stored as 2-D or as
    replicated 3-D is reduced to a single channel; multi-channel keeps its first
    n_channels (drops alpha)."""
    if n_channels == 1:
        chan0 = arr if arr.ndim == 2 else arr[..., 0]
        return chan0.reshape(-1, 1)
    return arr[..., :n_channels].reshape(-1, n_channels)


def _masked_pixels(arr, mask, n_channels):
    """Foreground pixels under a (H, W) mask, as (n_foreground, n_channels)."""
    px = arr[mask]                          # (n_fg,) or (n_fg, orig_channels)
    if px.ndim == 1:                        # grayscale stored 2-D
        px = px[:, None]
    return px[:, :n_channels]


def _channel_names(n_channels) -> tuple:
    """Display names per channel: I for single-channel, R/G/B for colour,
    ch0.. otherwise."""
    return {1: ("I",), 3: ("R", "G", "B")}.get(
        n_channels, tuple(f"ch{c}" for c in range(n_channels)))


def _channel_colors(n_channels) -> tuple:
    """matplotlib colours matching _channel_names."""
    return {1: ("k",), 3: ("r", "g", "b")}.get(
        n_channels, tuple(f"C{c % 10}" for c in range(n_channels)))


@dataclass
class IntensityStats:
    """Per-channel intensity summary over one population of sampled pixels.
    Every per-channel array has shape (C,), where C is the channel count of the
    population this summary belongs to."""
    n_pixels: int                        # pixels sampled per channel
    channel_names: tuple                 # length C, labels the channel axis
    intensity_min: np.ndarray            # per-channel, shape (C,)
    intensity_max: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    quartiles: np.ndarray                # rows [5,25,50,75,95%], shape (5, C)
    # per-channel histogram, shape (C, nbins)
    intensity_counts: np.ndarray
    intensity_bins: np.ndarray           # shared edges, shape (nbins + 1,)

    def table(self) -> list[str]:
        rows = {
            "count": [self.n_pixels] * len(self.channel_names),
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
        lines = [f"{'':>6}" + "".join(f"{c:>12}" for c in self.channel_names)]
        for name, vals in rows.items():
            lines.append(f"{name:>6}" +
                         "".join(f"{float(v):>12.2f}" for v in vals))
        return lines


def _stats(vx, bin_edges=None) -> IntensityStats:
    """Summarize an (n_pixels, C) sample. Reuse bin_edges to compare populations."""
    n_channels = vx.shape[1]
    if bin_edges is None:
        # Shared bin edges so the per-channel histograms are comparable
        bin_edges = np.histogram_bin_edges(vx, bins=100)
    return IntensityStats(
        n_pixels=len(vx),
        channel_names=_channel_names(n_channels),
        intensity_min=vx.min(axis=0),
        intensity_max=vx.max(axis=0),
        mean=vx.mean(axis=0),
        std=vx.std(axis=0),
        quartiles=np.percentile(vx, [5, 25, 50, 75, 95], axis=0),
        intensity_counts=np.stack([np.histogram(vx[:, c], bins=bin_edges)[0]
                                   for c in range(n_channels)]),
        intensity_bins=bin_edges,
    )


@dataclass
class DatasetFingerprint:
    """Fingerprint of one channel-count population (all images with n_channels)."""
    n_images: int
    n_channels: int                      # the channel count this group holds
    sizes: list[list[int]]
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
            f"Dataset fingerprint - {self.n_channels}-channel",
            "=" * 42,
            f"images         : {self.n_images}",
            f"labelled       : {self.n_labelled}",
            f"channels       : {self.n_channels}",
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
        colors = _channel_colors(len(stats.channel_names))
        for c, (name, color) in enumerate(zip(stats.channel_names, colors)):
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


@dataclass
class DatasetFingerprints:
    """One fingerprint per channel count. Index by the count: fps[3] for the
    colour images, fps[1] for the single-channel ones."""
    by_channels: dict[int, DatasetFingerprint]

    def __getitem__(self, n_channels) -> DatasetFingerprint:
        return self.by_channels[n_channels]

    def __iter__(self):
        return iter(self.by_channels.values())

    @property
    def single(self) -> DatasetFingerprint:
        """The one fingerprint, when the dataset has a single channel count."""
        if len(self.by_channels) != 1:
            raise ValueError(
                f"expected one channel group, got {sorted(self.by_channels)}")
        return next(iter(self.by_channels.values()))

    def describe(self) -> None:
        totals = {n: fp.n_images for n, fp in self.by_channels.items()}
        print(f"Channel groups (n_channels: images): {totals}\n")
        for fp in self.by_channels.values():
            fp.describe()
            print()

    def plot_summary(self):
        for fp in self.by_channels.values():
            fp.plot_summary()


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


@dataclass
class _Accum:
    """Running pixel samples for one channel-count group, before stats."""
    n_images: int = 0
    n_labelled: int = 0
    sizes: list = field(default_factory=list)
    voxels: list = field(default_factory=list)      # each (n_pixels, C)
    fg_voxels: list = field(default_factory=list)
    class_pixels: Counter = field(default_factory=Counter)   # unsampled

    def finalize(self, n_channels, label_map) -> DatasetFingerprint:
        all_stats = _stats(np.concatenate(self.voxels))
        # Reuse the all-pixel bin edges so the two histograms line up
        fg_stats = (_stats(np.concatenate(self.fg_voxels), all_stats.intensity_bins)
                    if self.fg_voxels else None)
        hw = np.array([s[:2] for s in self.sizes])   # first two dims = H, W
        return DatasetFingerprint(
            n_images=self.n_images,
            n_channels=n_channels,
            sizes=self.sizes,
            shape_min=hw.min(axis=0),
            shape_median=np.median(hw, axis=0).astype(int),
            shape_max=hw.max(axis=0),
            all=all_stats,
            foreground=fg_stats,
            n_labelled=self.n_labelled,
            label_map=label_map,
            class_pixels=dict(self.class_pixels))


def fingerprint(data_folder, label_folder=None,
                labels_file=None) -> DatasetFingerprints:
    label_map = read_label_map(labels_file) if labels_file else {}
    labels = label_index(label_folder) if label_folder else {}
    accums = defaultdict(_Accum)            # keyed by channel count
    for file in Path(data_folder).rglob("*"):
        if file.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            img_array = _normalize_format(itk.imread(str(file)))
        except Exception as err:            # skip unreadable/corrupt images
            print(f"skipping {file.name}: {err}")
            continue
        n_channels = _number_of_channels(img_array)
        if n_channels is None:              # unexpected dimensionality
            print(f"skipping {file.name}: cannot infer channels "
                  f"from shape {img_array.shape}")
            continue
        # each channel count fingerprinted alone
        acc = accums[n_channels]
        acc.sizes.append(list(img_array.shape))
        acc.voxels.append(_subsample(_pixels(img_array, n_channels)))
        acc.n_images += 1

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
        if seg.shape != img_array.shape[:2]:
            print(f"skipping label {label_file.name}: shape {tuple(seg.shape)} "
                  f"!= image {img_array.shape[:2]}")
            continue

        # Foreground = the values named in labels.txt, so anything unmapped
        # (background, stray values) is excluded rather than assumed foreground
        mask = np.isin(seg, list(label_map)) if label_map else seg > 0
        acc.n_labelled += 1
        for value, count in zip(*np.unique(seg[mask], return_counts=True)):
            acc.class_pixels[int(value)] += int(count)
        fg = _masked_pixels(img_array, mask, n_channels)   # (n_foreground, C)
        if len(fg):
            acc.fg_voxels.append(_subsample(fg))

    if not accums:
        raise ValueError(f"No readable images found under {data_folder}")

    # Most channels first (colour before grayscale) for a consistent report
    order = sorted(accums, key=lambda n: -n)
    return DatasetFingerprints({n: accums[n].finalize(n, label_map)
                                for n in order})


# %%
fp = fingerprint(DATA_FOLDER, LABEL_FOLDER, LABELS_FILE)
# %%
fp.describe()
fp.plot_summary()
# %%
