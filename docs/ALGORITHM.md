# The Projection Algorithm

## 1. Geometry

A plane is given as an isosceles **trapezoid** `(a1, a2, b, c)` with equal legs `a1→b` and
`a2→c`. A **triangle** `(a, b, c)` is the special case `a1 = a2 = a`, so both cases share one
algorithm.

```
        a1 ---- a2            a1 = a2
       /          \             /\
      /   rows y   \           /  \        each row y is the segment a1(y) -> a2(y)
     /              \         /    \       sampled at unit steps x1 = 0..floor(d12)
    b -------------- c       b ------ c
```

Notation:

* `Dab = |b − a1|` (leg length, equal to `|c − a2|` up to EPS)
* `Dbc = |c − b|` (base length)
* `v_ab = (b − a1) / Dab`, `v_ac = (c − a2) / |c − a2|` (unit leg directions)

## 2. The algorithm (as implemented)

```
Input : volume M (sx × sy × sz), valid plane (a1, a2, b, c)
Output: E with H = floor(Dab) rows and W = floor(Dbc) + 1 columns, dtype of M

E ← zeros(H, W)
for y = 0 .. H−1:
    p1  ← a1 + (y+1) · v_ab                 # point on the left leg  (step, then sample)
    p2  ← a2 + (y+1) · v_ac                 # point on the right leg
    d12 ← |p2 − p1|
    if d12 = 0: continue                    # degenerate row (cannot happen for a valid plane)
    dp  ← (p2 − p1) / d12                   # unit step along the row
    off ← floor((Dbc − d12) / 2)            # center the row in the image
    for x1 = 0 .. floor(d12):
        x ← off + x1
        if x ≥ W: break                     # floating-point safety
        p ← p1 + x1 · dp
        (i, j, k) ← (round(p.x), round(p.y), round(p.z))
        if 0 ≤ i < sx and 0 ≤ j < sy and 0 ≤ k < sz:
            E[y, x] ← M[i, j, k]            # otherwise E[y, x] stays 0
return E
```

This is `ReferenceProjector` in [projection/projector.py](../projection/projector.py), almost
line for line.

## 3. Differences from the assignment's pseudo-code, and why

| Assignment | Here | Why |
|---|---|---|
| `a1 = a1 + v(ab)` accumulated every row | `p1 = a1 + (y+1)·v_ab` computed directly | Same value mathematically. Direct computation avoids accumulating floating-point error over hundreds of rows, and makes rows **independent**, which is what allows vectorization. |
| `p = p + dp` accumulated along the row | `p = p1 + x1·dp` | Same reasons. |
| `E[x][y] = v` | `E[y, x] = v` | Row-major image layout (row = y), as expected by image and video libraries. |
| `x = (Dbc/2) − (d12/2) + x1` (real) | `x = floor((Dbc − d12)/2) + x1` | An array index must be an integer. |
| Output size `Dab × Dbc` (real) | `floor(Dab) × (floor(Dbc) + 1)` | Integer sizes; `x1` runs to `floor(d12) ≤ floor(Dbc)` inclusive, which needs `floor(Dbc)+1` columns. |
| `M[p]` with real `p` | `M[round(p)]` | Nearest-neighbour sampling (decision D1 in the SRS). |

**Note: the first row.** The assignment steps `a1`/`a2` *before* sampling, so row 0 is one
unit below the apex and the apex itself is never sampled, while the last row reaches `b`/`c`
when `Dab` is an integer. We kept this behavior for faithfulness. It is a one-line change
(`y+1` → `y`) if the opposite convention is preferred.

## 4. Fast implementation (`FastProjector`)

A pure Python loop costs about 1 µs per pixel, which gives ~2–5 FPS for a 500×500 image.
`FastProjector` computes the same result with whole-array NumPy operations:

1. **All rows at once:** `P1 = a1 + s·v_ab` and `P2 = a2 + s·v_ac` for `s = 1..H` (H×3 arrays),
   then `d12`, `dp` and `off` per row.
2. **Gather instead of scatter:** instead of computing where each sample lands, compute for
   every output pixel `(y, x)` which sample lands there: `x1 = x − off[y]`. A pixel is valid if
   `0 ≤ x1 ≤ floor(d12[y])`. Every pixel is then computed independently, with no write
   conflicts.
3. **One axis at a time:** the coordinate of axis `k` is `P1[y,k] + x1·dp[y,k]`. It is
   computed into one reused H×W float64 buffer, instead of an H×W×3 array, which reduces
   memory traffic about 3×. It uses float64 and the same operations as the reference
   (`a1 + x1·dp`, round half to even), so the two give identical images. A float32 buffer was
   about 15% faster, but on long planes (thousands of voxels) its rounding error picked a
   neighbouring voxel for about one pixel per plane.
4. **Flat index:** `flat = (i·sy + j)·sz + k` in int32 (safe because `sx·sy·sz < 2^30`). The
   volume is stored C-contiguous once, so `M.ravel()` is a view and one `take(flat)` reads all
   voxels.
5. **Masking:** out-of-volume or out-of-row pixels get index 0 (always valid) and are
   multiplied by 0 afterwards. There is no boolean-indexing copy.

The two implementations are compared pixel by pixel on random planes in the test suite
(see [TESTING.md](TESTING.md)).

## 5. Complexity

* Time: **O(H·W) = O(Dab·Dbc)** per frame. It does not depend on the volume size; only the
  number of output pixels matters. In practice, memory access patterns into a large volume
  also matter (see [EXPERIMENT.md](EXPERIMENT.md)).
* Extra memory: O(H·W) per frame. The volume is never copied per frame.
