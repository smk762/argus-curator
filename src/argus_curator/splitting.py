"""Train/held-out splitting: one generic split, by group key, that images and speech both use.

:func:`split_held_out` knows nothing about what an item is; a caller passes one
group key per item (images: ``selection.SPLIT_GROUP_KEYS``; speech: a speaker
group, #187). Kept apart from :mod:`argus_curator.selection` so
:mod:`argus_curator.speech_select` can use it without importing the image
selection that itself routes speech scans to ``speech_select``.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Sequence

from argus_curator.models import SplitConfig, SplitPart


class SplitError(ValueError):
    """A requested split cannot be made from the kept images (too few, or one group)."""


def _allot(k: int, sizes: list[int]) -> list[int]:
    """*k* held-out groups shared across strata of *sizes* groups, in proportion (largest remainder).

    A stratum never gives all its groups: one with a single group gives none,
    so each stratum keeps a group to train on.
    """
    total = sum(sizes)
    caps = [max(0, s - 1) for s in sizes]
    k = min(k, sum(caps))
    exact = [k * s / total for s in sizes]
    share = [min(int(e), c) for e, c in zip(exact, caps, strict=True)]
    by_remainder = sorted(range(len(sizes)), key=lambda i: (-(exact[i] - int(exact[i])), i))
    while sum(share) < k:
        for i in by_remainder:
            if sum(share) < k and share[i] < caps[i]:
                share[i] += 1
    return share


def _interleave(lists: list[list[int]]) -> list[int]:
    """The items of *lists* merged so each list's items are spread evenly through the result."""
    keyed = [((j + 0.5) / len(items), s, g) for s, items in enumerate(lists) for j, g in enumerate(items)]
    return [g for _, _, g in sorted(keyed)]


def split_held_out(
    keys: Sequence[Hashable | None], cfg: SplitConfig, strata: Sequence[Hashable] | None = None
) -> list[SplitPart]:
    """Each ranked item's part of a train/held-out split, from its group key.

    *keys* holds one group key per item, best item first; items sharing a key
    always land in the same part, and a ``None`` key is a group of its own.
    Groups are ordered by their best item. ``cfg.method`` "spread" takes
    held-out groups evenly through that order (with one item per group,
    exactly argus-mint's ``int((j + 0.5) * n / k)`` picks); "random" shuffles
    the groups with ``random.Random(cfg.seed)``. Groups are then taken in that
    order, skipping any that would push the held-out part past its target.

    With *strata* (one per item; a group takes its best item's) the held-out
    groups are shared across strata in proportion to how many groups each
    has, a share never taking a stratum's last group, and picked within each
    stratum as above (speech stratifies by accent and sex, #187). Only when
    the shares fall short of the target (groups too big to fit) are the
    remaining groups taken, from any stratum.

    Pure and deterministic. Raises :class:`SplitError` when either part would
    be empty.
    """
    n = len(keys)
    target = cfg.held_out if cfg.held_out is not None else max(1, round((cfg.held_out_fraction or 0.0) * n))
    if target >= n:
        raise SplitError(f"{n} kept item(s): holding out {target} leaves none to train")
    members: dict[Hashable, list[int]] = {}
    for i, key in enumerate(keys):
        members.setdefault(key if key is not None else ("ungrouped", i), []).append(i)
    groups = list(members.values())
    m = len(groups)
    if m < 2:
        raise SplitError(f"all {n} kept item(s) share one {cfg.group_by}; a split needs at least two groups")
    by_stratum: dict[Hashable, list[int]] = {}
    for g, items in enumerate(groups):
        by_stratum.setdefault(strata[items[0]] if strata is not None else None, []).append(g)
    layers = list(by_stratum.values())
    k = min(m - 1, max(1, round(target * m / n)))
    shares = _allot(k, [len(layer) for layer in layers]) if len(layers) > 1 else [k]
    rng = random.Random(cfg.seed)
    picks: list[list[int]] = []
    rests: list[list[int]] = []
    for layer, share in zip(layers, shares, strict=True):
        size = len(layer)
        if cfg.method == "random":
            local = list(range(size))
            rng.shuffle(local)
            chosen = local[:share] if len(layers) > 1 else local
        else:
            chosen = list(dict.fromkeys(int((j + 0.5) * size / share) for j in range(share))) if share else []
        taken = set(chosen)
        picks.append([layer[j] for j in chosen])
        rests.append([layer[j] for j in range(size) if j not in taken])
    order = _interleave([p for p in picks if p]) + _interleave([r for r in rests if r])
    held: set[int] = set()
    count = 0
    for g in order:
        if count >= target:
            break
        if count + len(groups[g]) <= target:
            held.update(groups[g])
            count += len(groups[g])
    if not held:
        smallest = min(len(g) for g in groups)
        raise SplitError(f"no {cfg.group_by} group fits in {target} held-out item(s); the smallest holds {smallest}")
    return ["held_out" if i in held else "train" for i in range(n)]
