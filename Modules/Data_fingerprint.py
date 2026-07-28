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
DATA_FOLDER = DATASET / "Omniseq"
LABEL_FOLDER = Path(
    "/Users/gastoncrecikeinbaum/Library/CloudStorage/SynologyDrive-XDMD/Annotations/2025 Xyall/Omniseq")
# "<value>: <name>" per line, 0 = background
LABELS_FILE = DATASET / "labels.txt"
data_path = Path(DATA_FOLDER)

# Pixels sampled per image for intensity stats
SAMPLE_PER_IMAGE = 50_000
_rng = np.random.default_rng(0)

# A dataset can mix chromatic and neutral images (e.g. H&E slides alongside
# unstained scans). Neutral images have R == G == B at every pixel, so
# mean(max - min) is 0. Pooling the two makes every statistic a blend that
# describes no real image, so each population is fingerprinted on its own.
# The tolerance is slack for lossy compression, which perturbs exact equality.
CHROMA_TOL = 1.0

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


def _chromaticity(vx) -> float:
    """Mean per-pixel (max channel - min channel). 0 means perfectly neutral."""
    wide = vx.astype(np.int16)              # uint8 would wrap on subtraction
    return float((wide.max(axis=1) - wide.min(axis=1)).mean())


def modality_name(chromatic: bool) -> str:
    """Display name for the chromaticity flag the dataset is split on."""
    return "chromatic" if chromatic else "neutral"


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
    # {label value: stats}, so a mode in the foreground can be traced to a class
    per_class: dict[int, IntensityStats] = field(default_factory=dict)
    # True when this population carries colour, False when it is neutral
    # (R == G == B). None when the dataset was not split.
    chromatic: bool | None = None
    chroma: list[float] = field(default_factory=list)   # per image

    @property
    def modality(self) -> str:
        return "" if self.chromatic is None else modality_name(self.chromatic)

    def describe(self) -> None:
        """Print the full report: dataset meta + per-channel intensity tables."""
        def shp(a): return list(map(int, a))
        chroma = (f"{np.median(self.chroma):.2f} median, "
                  f"{min(self.chroma):.2f}-{max(self.chroma):.2f} range"
                  if self.chroma else "n/a")
        lines = [
            f"Dataset fingerprint{f' - {self.modality}' if self.modality else ''}",
            "=" * 42,
            f"images         : {self.n_images}",
            f"labelled       : {self.n_labelled}",
            f"channel counts : {self.channel_counts}",
            f"chromaticity   : {chroma}",
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
                lines += ["",
                          "Per-class foreground (mean +/- std per channel)",
                          "-" * 66,
                          f"{'':>3} {'class':<10}{'pixels':>12}{'share':>8}"
                          + "".join(f"{c:>14}" for c in CHANNELS)]
                for value, count in sorted(self.class_pixels.items(),
                                           key=lambda kv: -kv[1]):
                    name = self.label_map.get(value, "?")
                    row = (f"{value:>3} {name:<10}{count:>12,}"
                           f"{count / total:>8.1%}")
                    stats = self.per_class.get(value)
                    if stats is not None:
                        row += "".join(f"{stats.mean[c]:>8.1f}+/-{stats.std[c]:<3.0f}"
                                       for c in range(3))
                    lines.append(row)
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

        if self.modality:
            fig.suptitle(f"{self.modality}  ({self.n_images} images)")
        fig.tight_layout()
        plt.show()


@dataclass
class DatasetFingerprints:
    """One fingerprint per chromaticity class. Index by the flag: fps[True] for
    the chromatic images, fps[False] for the neutral (greyscale) ones."""
    by_modality: dict[bool, DatasetFingerprint]

    def __getitem__(self, chromatic) -> DatasetFingerprint:
        return self.by_modality[chromatic]

    @property
    def single(self) -> DatasetFingerprint:
        """The one fingerprint, when the dataset was not split by chromaticity."""
        if len(self.by_modality) != 1:
            raise ValueError(
                f"expected one fingerprint, got {len(self.by_modality)}: "
                f"{[fp.modality for fp in self.by_modality.values()]}")
        return next(iter(self.by_modality.values()))

    def __iter__(self):
        return iter(self.by_modality.values())

    def class_matrix(self) -> dict[int, dict[str, int]]:
        """{label value: {modality: n_pixels}} over every declared class, so a
        class that is absent from a modality - or from the dataset - shows up
        as a zero rather than as a missing row."""
        declared = {}
        for fp in self.by_modality.values():
            declared.update(fp.label_map)
        observed = {v for fp in self.by_modality.values()
                    for v in fp.class_pixels}
        return {value: {m: fp.class_pixels.get(value, 0)
                        for m, fp in self.by_modality.items()}
                for value in sorted(set(declared) | observed)}

    def describe(self) -> None:
        if set(self.by_modality) == {None}:          # split turned off
            print("Chromaticity split: off (all images pooled)\n")
        else:
            totals = {fp.modality: fp.n_images
                      for fp in self.by_modality.values()}
            print(f"Modalities: {totals}  (split on chromaticity, "
                  f"tol {CHROMA_TOL})\n")
        for fp in self.by_modality.values():
            fp.describe()
            print()

        names = {}
        for fp in self.by_modality.values():
            names.update(fp.label_map)
        matrix = self.class_matrix()
        if not matrix:
            return
        modalities = [fp.modality for fp in self.by_modality.values()]
        print("Class presence by modality (pixels)")
        print("-" * (15 + 16 * len(modalities) + 8))
        print(f"{'':>3} {'class':<10}" + "".join(f"{m:>16}" for m in modalities)
              + f"{'':>3}")
        for value, counts in matrix.items():
            row = f"{value:>3} {names.get(value, '?'):<10}"
            row += "".join(f"{c:>16,}" if c else f"{'-':>16}"
                           for c in counts.values())
            present = [m for m, c in counts.items() if c]
            if len(present) > 1:            # blended if pooled
                row += "  *"
            elif not present:               # declared but never annotated
                row += "  (absent)"
            print(row)
        if any(sum(1 for c in counts.values() if c) > 1
               for counts in matrix.values()):
            print("\n* spans modalities - compare per modality, never pooled")

    def plot_summary(self):
        for fp in self.by_modality.values():
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
    """Running pixel samples for one modality, before stats are computed."""
    n_images: int = 0
    n_labelled: int = 0
    sizes: list = field(default_factory=list)
    channels: list = field(default_factory=list)
    chroma: list = field(default_factory=list)
    # each entry is (n_pixels, 3)
    voxels: list = field(default_factory=list)
    fg_voxels: list = field(default_factory=list)
    class_pixels: Counter = field(default_factory=Counter)   # unsampled
    class_voxels: defaultdict = field(
        default_factory=lambda: defaultdict(list))

    def finalize(self, label_map, chromatic) -> DatasetFingerprint:
        all_stats = _stats(np.concatenate(self.voxels))
        # Reuse the all-pixel bin edges so every histogram lines up
        fg_stats = (_stats(np.concatenate(self.fg_voxels), all_stats.intensity_bins)
                    if self.fg_voxels else None)
        per_class = {value: _stats(np.concatenate(vx), all_stats.intensity_bins)
                     for value, vx in sorted(self.class_voxels.items())}
        # Spatial size stats (first two dims = H, W)
        hw = np.array([s[:2] for s in self.sizes])
        return DatasetFingerprint(n_images=self.n_images,
                                  sizes=self.sizes,
                                  channel_counts=dict(Counter(self.channels)),
                                  shape_min=hw.min(axis=0),
                                  shape_median=np.median(
                                      hw, axis=0).astype(int),
                                  shape_max=hw.max(axis=0),
                                  all=all_stats,
                                  foreground=fg_stats,
                                  n_labelled=self.n_labelled,
                                  label_map=label_map,
                                  class_pixels=dict(self.class_pixels),
                                  per_class=per_class,
                                  chromatic=chromatic,
                                  chroma=self.chroma)


def fingerprint(data_folder, label_folder=None, labels_file=None,
                split_chromaticity=True) -> DatasetFingerprints:
    """Fingerprint every image under data_folder.

    split_chromaticity separates chromatic from neutral images into their own
    fingerprints. Turn it off when the dataset is known to be uniform (every
    image colour, or every image greyscale) to get a single pooled fingerprint;
    per-image chromaticity is still reported either way.
    """
    label_map = read_label_map(labels_file) if labels_file else {}
    labels = label_index(label_folder) if label_folder else {}
    accums = defaultdict(_Accum)
    for file in Path(data_folder).rglob("*"):
        if file.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            img_array = _normalize_format(itk.imread(str(file)))
        except Exception as err:            # skip unreadable/corrupt images
            print(f"skipping {file.name}: {err}")
            continue
        rgb = _to_rgb(img_array)
        vx = _subsample(rgb.reshape(-1, 3))
        # Classify on the sample, not the full image: the sample is what the
        # statistics are built from, and neutrality is a whole-image property
        chroma = _chromaticity(vx)
        # None keys the single pooled population when the split is off
        acc = accums[(chroma > CHROMA_TOL) if split_chromaticity else None]
        acc.sizes.append(list(img_array.shape))
        acc.channels.append(img_array.shape[-1] if img_array.ndim == 3 else 1)
        acc.chroma.append(chroma)
        acc.voxels.append(vx)
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
        if seg.shape != rgb.shape[:2]:
            print(f"skipping label {label_file.name}: shape {tuple(seg.shape)} "
                  f"!= image {rgb.shape[:2]}")
            continue

        # Foreground = the values named in labels.txt, so anything unmapped
        # (background, stray values) is excluded rather than assumed foreground
        mask = np.isin(seg, list(label_map)) if label_map else seg > 0
        acc.n_labelled += 1
        for value, count in zip(*np.unique(seg[mask], return_counts=True)):
            value = int(value)
            acc.class_pixels[value] += int(count)
            # Sample each class separately: a rare class would otherwise be
            # swamped in the pooled foreground sample and get no usable stats
            acc.class_voxels[value].append(_subsample(rgb[seg == value]))
        fg = rgb[mask]                      # (n_foreground, 3)
        if len(fg):
            acc.fg_voxels.append(_subsample(fg))

    if not accums:
        raise ValueError(f"No readable images found under {data_folder}")

    # Chromatic first when both are present, so the report reads consistently
    order = [m for m in (True, False, None) if m in accums]
    return DatasetFingerprints({m: accums[m].finalize(label_map, m)
                                for m in order})


# # %%
fp = fingerprint(DATA_FOLDER, LABEL_FOLDER, LABELS_FILE)
# # %%
fp.describe()
fp.plot_summary()
# %%
