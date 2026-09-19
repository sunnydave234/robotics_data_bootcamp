"""
tensor_utils.py -- Week 3 Day 1 shared helper
================================================
One function, one job: turn whatever Ray Data hands you for a fixed-width
tensor column (like `action`, `observation.state`) into a genuine
(num_rows, dim) 2D float array, no matter which internal representation
Ray chose for that particular column.

WHY THIS EXISTS -- the real bug story:

Ray's own docs say fixed-shape tensor columns come back as regular
(num_rows, dim) ndarrays in "numpy" batch format. For aloha_mobile_cabinet's
`action` column, that did NOT happen -- confirmed by adding a debug print
and actually looking:

    batch["action"].shape == (256,)
    batch["action"].dtype == object

Instead of a (256, 14) array, Ray handed back a 1-D array of length 256
where each of the 256 "elements" is itself a 14-length array -- the exact
fallback Ray's docs describe for tensor columns it does NOT recognize as
uniformly fixed-shape ("If your tensors vary in shape, Ray Data represents
batches as arrays of object dtype"). Every row here genuinely has 14
values, so it isn't really ragged -- but whatever Ray inferred from
LeRobot's Parquet schema for this column made it fall back to the ragged
representation anyway.

The consequence: `np.linalg.norm(batch["action"], axis=-1)` on a
shape-(256,) object array reduces over axis 0 (the ONLY axis that shape
has) instead of over the 14 joint dimensions -- numpy doesn't raise an
error, it silently squares each of the 256 objects elementwise (each stays
a 14-vector), sums those 256 14-vectors position-by-position, and takes
the sqrt of THAT. Result: a 14-length array that means nothing, instead of
a 256-length array of real per-row magnitudes. That's where the
"expected length 256 but got length 14" error came from -- confirmed by
reproducing the exact mechanism with plain numpy, no Ray needed.

Lesson worth keeping, in the same spirit as Week 1-2's "verify the field,
don't trust the docs/name": a column's documented *typical* batch
representation is not a guarantee for *this* dataset's actual on-disk
schema. Check the real shape before you trust it.
"""
import numpy as np


def to_2d_float_array(batch: dict, column: str, expected_dim: int) -> np.ndarray:
    """
    Return batch[column] as a genuine (num_rows, expected_dim) float32
    ndarray, handling the case where Ray Data represented it as a 1-D
    object array of per-row arrays instead.

    Raises ValueError loudly on any shape it doesn't recognize, rather
    than silently computing something wrong -- exactly the failure mode
    this function exists to prevent.
    """
    arr = np.asarray(batch[column])
    n_rows = len(batch["episode_index"])  # any always-present scalar column works as the row-count reference

    if arr.ndim == 1 and arr.dtype == object:
        # The fallback case that actually happens on this dataset: stack
        # the n_rows separate (expected_dim,) arrays into one 2D array.
        arr = np.stack(arr)
    elif arr.ndim == 1:
        raise ValueError(
            f"'{column}' came back 1-D with dtype={arr.dtype}, not object -- "
            f"this is a genuinely new shape this helper doesn't handle. "
            f"Inspect batch['{column}'] directly before trusting anything downstream."
        )

    if arr.shape == (expected_dim, n_rows):
        # Defensive: handle a transposed layout if it ever shows up.
        arr = arr.T

    if arr.shape != (n_rows, expected_dim):
        raise ValueError(
            f"'{column}' has shape {arr.shape} after normalization; "
            f"expected ({n_rows}, {expected_dim})."
        )

    return arr.astype(np.float32)