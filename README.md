# ComfyUI-TiledUpscale

Tiled refine/upscale for reference-latent edit models (FLUX.2 [klein], Krea 2 and similar): split an image into overlapping tiles, regenerate each one at full tile resolution, and blend them back into a single image so you can add detail beyond the resolution a model can handle in one pass.

The tile geometry and the latent the sampler starts from are read from the loaded model, so the node is not tied to one architecture: FLUX.2 (128-channel latent, 1/16 VAE) and Krea 2 (16-channel video-shaped latent, 1/8 VAE) both work.

### Krea 2 setup

Stock Krea 2 (Raw or Turbo) is a **text-to-image** model with no reference path at all, which ComfyUI records as `default_ref_method = None`. Handed a reference it either ignores it or produces noise, so tiled refine needs one of the reference-capable LoRAs on top:

| | LoRA | `reference_mode` |
|---|---|---|
| Detail/identity-preserving refine | [`krea2_identity_edit`](https://huggingface.co/conradlocke/krea2-identity-edit) | `in_context` (what `auto` picks) |
| Style reference | [`krea2_style_reference`](https://huggingface.co/Comfy-Org/Krea-2) | `conditioning` |

Wire `UNETLoader → LoraLoaderModelOnly → this node`, and use the rest of the Krea 2 stack: CLIP loaded with `CLIPLoader` type `krea2` (Qwen3-VL 4B) and **`qwen_image_vae.safetensors`**, not the Wan 2.1 VAE. Both share the same latent format, so the wrong one raises no error and just degrades the output into artefacts.

With the identity-edit LoRA, also connect the `clip` input and put your instruction in `prompt`. These LoRAs are trained with the source image in *both* paths, as latent tokens and inside the text encoder's user turn, and with the text half missing the model largely ignores the reference and generates from the prompt alone. Connecting `clip` re-encodes the prompt per tile with that tile as the encoder's vision input, and overrides `positive`. Grounding the whole image once instead (for instance with an external grounded-encode node) describes the entire scene to every tile, and each tile then redraws the whole composition.

The identity-edit LoRA's own release notes recommend generating at ~1–1.5MP and upscaling afterwards, which is exactly what this node is for.

|  Before  |  After (Tiled Upscale & Refine)  |
|:---:|:---:|
| ![before](examples/before.png) | ![after](examples/after.png) |

**Large-scale example:** 1920x1088 (2.1MP) source upscaled to 7968x4512 (36MP) using `upscale_by=4.151`, `tile_size=2048`, `overlap=192`, for 15 tiles.

|  Before (1920x1088)  |  After (7968x4512, 36MP, 15 tiles)  |
|:---:|:---:|
| ![base](examples/large-scale-base.jpg) | ![36mp result](examples/large-scale-36mp.jpg) |

## Example workflow

Drag [`workflows/ComfyUI-TiledUpscale.json`](workflows/ComfyUI-TiledUpscale.json) into ComfyUI for a working setup. Apart from this node pack it only uses core nodes, so there is nothing else to install.

## Nodes

### Tiled Upscale & Refine
The all-in-one node. Give it a model, plain text conditioning, a VAE and an image; it does the whole pipeline internally and returns the finished image. No `ReferenceLatent` / `EmptyLatent` / `KSampler` wiring needed because the per-tile reference latent is attached inside.

**Inputs**
- `upscale_by`: resize the image before tiling (Lanczos). `1.0` = refine at current size.
- `tile_size`: resolution each tile is regenerated at. `0` (default) picks one from the output size: about four tiles along the longest side, kept between 1024 and 1536. Set it by hand for fewer seams if you have the VRAM, bearing in mind that attention cost grows with the square of the tile while the tile count only falls linearly, so the total cost grows too. Tested working at 2048 on a 36MP image.
- `overlap` (px): how much neighbouring tiles share. `0` (default) picks 30% of the tile size. A tile drifts from its neighbours over roughly the outer 15% of its width, and an overlap band is the one place two tile edges meet, so the band has to be wide enough that its middle falls outside both edge zones. It follows the tile, not `upscale_by`.
- `feather` (0–1): fraction of the actual overlap used for the blend fade. `1.0` (default) fades across the whole shared region.
- `seed` / `steps` / `cfg` / `sampler_name` / `scheduler` / `denoise`: as in a normal KSampler. Each tile is sampled from its own latent, so `denoise` is the usual img2img strength: `1.0` regenerates the tile outright, lower values renoise it and keep more of what was there. Lower it when a subject sits across a seam: four tiles independently regenerating the same face at `1.0` average into a ghost, while at `0.7` to `0.85` they stay put.
- `sequential_context` (default on): each tile is cropped from the canvas of already-generated tiles, so it continues real neighbouring pixels instead of inventing that region blind. This is what keeps detail lining up across a join. Turn it off to regenerate every tile independently from the source.
- `reference_mode` (default `auto`): how each tile is handed to the model as its own reference. `conditioning` attaches it as a reference latent, the path FLUX.2, Qwen and the Krea 2 style-reference LoRA use. `in_context` prepends the tile as a block of clean source tokens inside the model's forward (`[text | source(frame=1) | target(frame=0)]`), which is what the Krea 2 edit LoRAs are trained on and what ComfyUI's generic reference path does *not* reproduce. `auto` picks `in_context` for Krea 2 and `conditioning` for everything else.
- `clip` / `prompt` / `grounding_px`: Krea 2 in-context mode only. See [Krea 2 setup](#krea-2-setup). `clip` turns on per-tile prompt grounding and overrides `positive`; `grounding_px` (default 768) caps the longest side of the tile fed to the text encoder, lower favouring edit adherence and higher favouring likeness.
- `redo_first_tile` (default on): the top-left tile is the only one generated with no already-refined neighbour to continue from, so it can sharpen a blurry background or hallucinate detail the rest of the image never gets. With this on, it is generated once more at the end, now surrounded by finished tiles, so it gets the same context every other tile had. Costs one extra tile of sampling; needs `sequential_context`.
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
- **Some models need to be told to read their reference latents.** Most edit models carry a default reference method (FLUX.2 `index`, FLUX.1 `offset`); when one doesn't, its reference latents are *silently ignored* and each tile is regenerated from the prompt alone, and the giveaway is an output with nothing to do with the input image. In `conditioning` mode the node fills in `index_timestep_zero` for models in that state and says so in `info`. To force a different method, put the core `Edit Model Reference Method` node on the positive conditioning before this node; a method set upstream is left alone.
- **The in-context layout is not ComfyUI's reference layout.** ComfyUI builds `[text | target | refs]` and gives the reference tokens their own timestep-zero modulation; the Krea 2 edit LoRAs were trained on `[text | source(frame=1) | target(frame=0)]` with a single shared timestep, the source told apart from the noisy target only by its RoPE frame index. That is why `in_context` exists as a separate mode rather than a flag. The layout follows [ComfyUI-Krea2Edit](https://github.com/lbouaraba/comfyui-krea2edit) (Apache-2.0), which mirrors ai-toolkit's `predict_velocity_edit`.
- **A tile's input fades back to the source steeply, and only at the rim.** With `sequential_context` a tile's input is part already-generated neighbour, part untouched source, and both ways of getting that wrong are visible. *Switching* between them wherever coverage is non-zero puts a hard step in the model's input at the rim of every overlap band. The stored neighbour content is weighted, so normalising it makes even a 0.1%-weight sliver read as full-strength output, sitting right beside untouched source. The model redraws that step: measured at up to 25 luminance levels on a flat night sky, against 0.34 for ordinary pixel-to-pixel noise. *Fading* across the whole band removes the step but leaves the neighbour at a fraction of its strength mid-band, which is too weak to anchor a subject: the model repositions it, and the output blend averages the two positions into a ghost. So the fade is confined to the outer quarter of the band, leaving the neighbour at full strength everywhere else.
- **Exposure and detail need opposite blends, so they get one each.** Tiles disagree in two ways at once, and a single weighted average cannot serve both. Exposure and haze drift over hundreds of pixels and need the *widest* possible transition to hide, which argues for a generous overlap. Contours drawn a few pixels apart need no transition at all: averaging keeps both copies, which is what a doubled edge is, and a wider overlap only spreads the doubling further. The arithmetic makes the conflict concrete: at 44% overlap only 5% of the image comes from a single tile, so almost every contour is an average of two drawings of it. Each tile is therefore split into a low band, blended across the full ramp, and a detail band, taken from whichever tile is dominant at that pixel and switched over a narrow strip. Where the picture is smooth the detail band is near zero and the switch is invisible; where it has texture, the texture hides it. This does not make disagreeing tiles agree; it stops the disagreement from being printed twice.
- **Grids that cut through a subject are riskier than grids that don't.** Each tile is a separate generation; where a boundary crosses a face, a hand or a continuous line, the two halves have to agree on something they generated separately. `sequential_context` helps a lot here, a wider `overlap` helps further.