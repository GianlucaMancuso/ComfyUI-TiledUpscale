# ComfyUI-TiledUpscale

Tiled refine/upscale for reference-latent edit models (FLUX.2 [klein] and similar): split an image into overlapping tiles, regenerate each one at full tile resolution, and blend them back into a single image so you can add detail beyond the resolution a model can handle in one pass.

|  Before  |  After (Tiled Upscale & Refine)  |
|:---:|:---:|
| ![before](examples/before.png) | ![after](examples/after.png) |

## Nodes

### Tiled Upscale & Refine
The all-in-one node. Give it a model, plain text conditioning, a VAE and an image; it does the whole pipeline internally and returns the finished image. No `ReferenceLatent` / `EmptyLatent` / `KSampler` wiring needed because the per-tile reference latent is attached inside.

**Inputs**
- `upscale_by`: resize the image before tiling (Lanczos). `1.0` = refine at current size.
- `tile_size`: resolution each tile is regenerated at. Keep it at what the model is happiest with (1024 for klein).
- `overlap` (px): how much neighbouring tiles share. Wider = smoother joins, more tiles, slower.
- `feather` (0–1): fraction of the actual overlap used for the blend fade. `1.0` (default) fades across the whole shared region.
- `seed` / `steps` / `cfg` / `sampler_name` / `scheduler` / `denoise`: as in a normal KSampler.
- `sequential_context` (default on): each tile is cropped from the canvas of already-generated tiles, so it continues real neighbouring pixels instead of inventing that region blind. This is what keeps detail lining up across a join. Turn it off to regenerate every tile independently from the source.
- `color_match` / `color_match_strength`: per-tile colour matching back to the source crop (`mkl` by default), which stops tiles from drifting apart in exposure or tint.
- `final_color_match`: one more pass over the finished image against the source.

**Outputs**
- `image`: the refined result, back at the input's resolution (times `upscale_by`).
- `info`: the grid that was used and each tile's position and real per-side overlap.

### Tile Split / Tile Merge
The manual building blocks, if you want your own nodes between splitting and merging (a different sampler, extra processing per tile, etc.).

`Tile Split` takes `image`, `tile_size`, `overlap` and returns `Tiles Batch` (all tiles as one batch), `Tiles List` (a ComfyUI list: anything you connect downstream runs once per tile automatically), `Tile Info` (metadata `Tile Merge` needs) and `info`.

`Tile Merge` takes the processed `images` (batch or list), the `tile_info`, and a `feather` fraction, and blends everything back into one image.

Round-trip is lossless: splitting and merging without touching the tiles returns the original image (verified to ~1e-7).

### Tile Grid Advisor
Given an image, a desired grid (e.g. 2×2) and an overlap, it suggests the `tile_size` that produces it, and reports the grid you'd *actually* get including a note when the requested grid isn't reachable with a single tile size at that aspect ratio.

## Notes

- **Tiles are laid out evenly, not by fixed stride.** Spacing tiles evenly across the image makes every gap between neighbours identical. The obvious alternative, step by a fixed stride and clamp the last tile to the edge, leaves one abnormally large final overlap, and when the leftover is smaller than a step it can even stack several tiles at the same position.
- **The blend fade is derived from the real overlap, not the requested one.** If the fade is narrower than the region two tiles actually share, the middle of that region ends up with both tiles at full weight, meaning a flat 50/50 average of two independent generations. That reads as a hard band across the image, visible even on flat backgrounds where there's no detail to hide it.
- **Grids that cut through a subject are riskier than grids that don't.** Each tile is a separate generation; where a boundary crosses a face, a hand or a continuous line, the two halves have to agree on something they generated separately. `sequential_context` helps a lot here, a wider `overlap` helps further.