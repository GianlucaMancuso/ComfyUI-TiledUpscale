import torch.nn.functional as F
import torch
import math
import logging

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management
import node_helpers

# FLUX2's latent is height//16, width//16 (see EmptyFlux2LatentImage in comfy_extras/nodes_flux.py),
# Tile dimensions are snapped to this so a tile maps to a whole number of latent rows/columns.
LATENT_SCALE = 16

# One-shot flags so optional-dependency warnings don't repeat once per tile.
_warned = {"color_matcher": False}


def _plan_axis(size, tile, overlap):
    """
    Lay out tiles along one axis.
    Returns a list of (position, length). Tiles are distributed *evenly* rather than by a fixed
    stride with the last one clamped to the edge: even spacing makes every gap between neighbours
    identical, which keeps the blend weights well behaved.
    """
    tile = min(tile, size)
    if size <= tile:
        return [(0, size)]

    stride = max(1, tile - max(0, overlap))
    count = math.ceil((size - tile) / stride) + 1
    span = size - tile
    return [(round(i * span / (count - 1)), tile) for i in range(count)]


def _axis_fades(positions):
    """
    Per-tile (fade_start, fade_end) widths: the actual overlap with each neighbour.
    Deriving these from real positions rather than from the requested `overlap` value is what
    keeps the blend correct.
    """
    fades = []
    for i, (pos, length) in enumerate(positions):
        start = (positions[i - 1][0] + positions[i - 1][1]) - pos if i > 0 else 0
        end = (pos + length) - positions[i + 1][0] if i < len(positions) - 1 else 0
        fades.append((max(0, start), max(0, end)))
    return fades


def _axis_ramp(size, fade_start, fade_end, device, dtype):
    """1-D weight ramp: 1.0 in the tile's core, fading to 0 across each overlapping edge."""
    ramp = torch.ones(size, device=device, dtype=dtype)

    # Opposite fades must not eat into each other, or the weights stop summing to 1.
    if fade_start + fade_end > size:
        scale = size / (fade_start + fade_end)
        fade_start = int(fade_start * scale)
        fade_end = int(fade_end * scale)

    if fade_start > 0:
        ramp[:fade_start] *= torch.linspace(0.0, 1.0, fade_start, device=device, dtype=dtype)
    if fade_end > 0:
        ramp[-fade_end:] *= torch.linspace(1.0, 0.0, fade_end, device=device, dtype=dtype)
    return ramp


def _tile_weight(info, feather, device, dtype):
    """2-D weight map for one tile, as the outer product of its two axis ramps."""
    ramp_y = _axis_ramp(info["height"], round(info["fade_top"] * feather), round(info["fade_bottom"] * feather), device, dtype)
    ramp_x = _axis_ramp(info["width"], round(info["fade_left"] * feather), round(info["fade_right"] * feather), device, dtype)
    return (ramp_y.unsqueeze(1) * ramp_x.unsqueeze(0)).unsqueeze(0).unsqueeze(-1)


def _plan_tiles(img_h, img_w, tile_size, overlap):
    """Full tile grid over an image, with per-side overlap widths already resolved."""
    rows = _plan_axis(img_h, tile_size, overlap)
    cols = _plan_axis(img_w, tile_size, overlap)
    row_fades = _axis_fades(rows)
    col_fades = _axis_fades(cols)

    tiles = []
    tile_id = 0
    for r, (y, h) in enumerate(rows):
        for c, (x, w) in enumerate(cols):
            tiles.append({
                "tile_id": tile_id,
                "row": r,
                "col": c,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "fade_top": row_fades[r][0],
                "fade_bottom": row_fades[r][1],
                "fade_left": col_fades[c][0],
                "fade_right": col_fades[c][1],
                "is_edge_top": r == 0,
                "is_edge_bottom": r == len(rows) - 1,
                "is_edge_left": c == 0,
                "is_edge_right": c == len(cols) - 1,
            })
            tile_id += 1
    return tiles, len(cols), len(rows)


def _describe_tiles(tiles, num_x, num_y, img_w, img_h, tile_size):
    lines = [f"Tile grid: {num_x}x{num_y} ({len(tiles)} tiles), tile size {tile_size}px, image {img_w}x{img_h}."]
    for info in tiles:
        lines.append(f"  tile {info['tile_id']}: row={info['row']} col={info['col']} "
                     f"pos=({info['x']},{info['y']}) size={info['width']}x{info['height']} "
                     f"overlap(t,b,l,r)=({info['fade_top']},{info['fade_bottom']},{info['fade_left']},{info['fade_right']})")
    return lines


def _color_match(target, reference, method, strength):
    """Match `target`'s colour distribution to `reference` (both BHWC tensors, batch of 1)."""
    if method == "none" or strength <= 0.0:
        return target
    try:
        from color_matcher import ColorMatcher
    except ImportError:
        # Say so once instead of silently skipping: without colour matching the tiles drift
        # apart in exposure, and the resulting seams look like a bug in the blending.
        if not _warned["color_matcher"]:
            _warned["color_matcher"] = True
            logging.warning("[TiledUpscale] colour matching is disabled: the 'color-matcher' package "
                            "is not installed. Install this node pack's requirements.txt, or set "
                            "color_match to 'none' to silence this.")
        return target

    cm = ColorMatcher()
    src = target[0].detach().cpu().float().numpy()
    ref = reference[0].detach().cpu().float().numpy()
    try:
        result = cm.transfer(src=src, ref=ref, method=method)
    except Exception:
        logging.exception("[TiledUpscale] colour matching failed for one tile; using it uncorrected.")
        return target
    if strength != 1.0:
        result = src + strength * (result - src)
    out = torch.from_numpy(result).to(device=target.device, dtype=target.dtype).unsqueeze(0)
    return out.clamp(0.0, 1.0)


class TileSplit:
    CATEGORY = "Tiled Upscale"
    RETURN_TYPES = ("IMAGE", "IMAGE", "TILE_INFO", "STRING")
    RETURN_NAMES = ("Tiles Batch", "Tiles List", "Tile Info", "info")
    OUTPUT_IS_LIST = (False, True, False, False)
    FUNCTION = "split"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 4096, "step": 16}),
            }
        }

    def split(self, image, tile_size, overlap):
        _, img_h, img_w, _ = image.shape
        if overlap >= tile_size:
            overlap = tile_size - 16

        tiles_info, num_x, num_y = _plan_tiles(img_h, img_w, tile_size, overlap)

        crops = []
        for info in tiles_info:
            x, y, w, h = info["x"], info["y"], info["width"], info["height"]
            crops.append(image[:, y:y + h, x:x + w, :])
            info["original_width"] = img_w
            info["original_height"] = img_h

        # Tiles clipped at the image edge can be smaller, so a single stacked batch is only
        # possible when every tile came out the same size.
        same_size = all(c.shape[1:] == crops[0].shape[1:] for c in crops)
        tiles_batch = torch.cat(crops, dim=0) if same_size else crops[0]

        info_lines = _describe_tiles(tiles_info, num_x, num_y, img_w, img_h, tile_size)
        if not same_size:
            info_lines.append("Note: tiles have differing sizes, so 'Tiles Batch' holds only the first tile — use 'Tiles List'.")

        return (tiles_batch, crops, tiles_info, "\n".join(info_lines))


class TileMerge:
    CATEGORY = "Tiled Upscale"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Merged Image",)
    FUNCTION = "merge"
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tile_info": ("TILE_INFO",),
                "feather": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    def merge(self, images, tile_info, feather):
        tile_info = tile_info[0]
        feather = feather[0]

        if len(images) == 1 and images[0].shape[0] == len(tile_info):
            tiles = [images[0][i:i + 1] for i in range(images[0].shape[0])]
        else:
            tiles = images

        if len(tiles) != len(tile_info):
            raise ValueError(f"TileMerge: got {len(tiles)} tiles but {len(tile_info)} tile_info entries.")

        out_w = tile_info[0]["original_width"]
        out_h = tile_info[0]["original_height"]

        first = tiles[0]
        B, _, _, C = first.shape
        device, dtype = first.device, first.dtype

        canvas = torch.zeros(B, out_h, out_w, C, device=device, dtype=dtype)
        weight_sum = torch.zeros(B, out_h, out_w, 1, device=device, dtype=dtype)

        for tile, info in zip(tiles, tile_info):
            x, y, w, h = info["x"], info["y"], info["width"], info["height"]
            if tile.shape[1] != h or tile.shape[2] != w:
                tile = F.interpolate(tile.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

            weight = _tile_weight(info, feather, device, dtype)
            canvas[:, y:y + h, x:x + w, :] += tile * weight
            weight_sum[:, y:y + h, x:x + w, :] += weight

        return (canvas / weight_sum.clamp(min=1e-8),)


class TileGridAdvisor:
    CATEGORY = "Tiled Upscale"
    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("suggested_tile_size", "info")
    FUNCTION = "advise"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tiles_x": ("INT", {"default": 2, "min": 1, "max": 32}),
                "tiles_y": ("INT", {"default": 2, "min": 1, "max": 32}),
                "overlap": ("INT", {"default": 128, "min": 0, "max": 4096, "step": 16}),
            }
        }

    def advise(self, image, tiles_x, tiles_y, overlap):
        _, img_h, img_w, _ = image.shape

        def needed(size, count):
            if count <= 1:
                return size
            return math.ceil((size + overlap * (count - 1)) / count)

        suggested = max(needed(img_w, tiles_x), needed(img_h, tiles_y))
        suggested = max(64, (suggested // 16) * 16)

        tiles, actual_x, actual_y = _plan_tiles(img_h, img_w, suggested, overlap)
        info = (f"Image {img_w}x{img_h}. For a {tiles_x}x{tiles_y} grid with overlap={overlap}px, "
                f"set tile_size={suggested}.\nActual resulting grid: {actual_x}x{actual_y} ({len(tiles)} tiles).")
        if actual_x != tiles_x or actual_y != tiles_y:
            info += ("\nNote: that grid isn't exactly reachable with one shared tile_size at this aspect "
                     "ratio — this is the closest fit (the longer axis sets the size).")
        return (suggested, info)


class TiledUpscaleRefine:
    CATEGORY = "Tiled Upscale"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "refine"
    DESCRIPTION = ("All-in-one tiled refine/upscale for reference-latent edit models (FLUX.2 klein and "
                   "similar). Upscales, splits into overlapping tiles, regenerates each tile at full "
                   "tile resolution, colour-matches it back to the source, and blends everything into "
                   "one image. Feed it plain CLIPTextEncode conditioning — the per-tile reference latent "
                   "is attached internally, so no ReferenceLatent/EmptyLatent/KSampler nodes are needed.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "upscale_by": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 8.0, "step": 0.05}),
                "tile_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "overlap": ("INT", {"default": 192, "min": 0, "max": 2048, "step": 16}),
                "feather": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sequential_context": ("BOOLEAN", {"default": True}),
                "color_match": (["mkl", "hm", "reinhard", "mvgd", "hm-mvgd-hm", "hm-mkl-hm", "none"],),
                "color_match_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "final_color_match": ("BOOLEAN", {"default": True}),
            }
        }

    def _upscale(self, image, factor):
        if factor == 1.0:
            return image
        _, h, w, _ = image.shape
        new_h = max(16, round(h * factor / LATENT_SCALE) * LATENT_SCALE)
        new_w = max(16, round(w * factor / LATENT_SCALE) * LATENT_SCALE)
        bchw = image.permute(0, 3, 1, 2)
        bchw = comfy.utils.common_upscale(bchw, new_w, new_h, "lanczos", "disabled")
        return bchw.permute(0, 2, 3, 1)

    def _sample_tile(self, model, positive, negative, vae, tile_pixels, seed, steps, cfg,
                     sampler_name, scheduler, denoise):
        """Regenerate one tile: encode it, attach it as the reference latent, sample, decode."""
        tile_latent = vae.encode(tile_pixels[:, :, :, :3])

        # Same thing the ReferenceLatent node does — the tile guides its own regeneration.
        positive_tile = node_helpers.conditioning_set_values(
            positive, {"reference_latents": [tile_latent]}, append=True)

        _, h, w, _ = tile_pixels.shape
        latent = torch.zeros([1, 128, h // LATENT_SCALE, w // LATENT_SCALE],
                             device=comfy.model_management.intermediate_device())

        noise = comfy.sample.prepare_noise(latent, seed, None)
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler,
                                      positive_tile, negative, latent, denoise=denoise,
                                      disable_pbar=True, seed=seed)

        decoded = vae.decode(samples)
        if len(decoded.shape) == 5:
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        return decoded

    def refine(self, model, positive, negative, vae, image, upscale_by, tile_size, overlap, feather,
               seed, steps, cfg, sampler_name, scheduler, denoise, sequential_context,
               color_match, color_match_strength, final_color_match):
        source = self._upscale(image, upscale_by)
        _, img_h, img_w, _ = source.shape

        if overlap >= tile_size:
            overlap = tile_size - 16

        # Tiles must land on whole latent cells, so pad the image up to a multiple of the latent
        # scale and give every tile a size the VAE can round-trip exactly.
        pad_h = (-img_h) % LATENT_SCALE
        pad_w = (-img_w) % LATENT_SCALE
        if pad_h or pad_w:
            bchw = source.permute(0, 3, 1, 2)
            bchw = F.pad(bchw, (0, pad_w, 0, pad_h), mode="replicate")
            source = bchw.permute(0, 2, 3, 1)
        work_h, work_w = img_h + pad_h, img_w + pad_w

        tile_size = max(LATENT_SCALE, (tile_size // LATENT_SCALE) * LATENT_SCALE)
        tiles, num_x, num_y = _plan_tiles(work_h, work_w, tile_size, overlap)

        device, dtype = source.device, source.dtype
        canvas = torch.zeros(1, work_h, work_w, source.shape[-1], device=device, dtype=dtype)
        weight_sum = torch.zeros(1, work_h, work_w, 1, device=device, dtype=dtype)

        pbar = comfy.utils.ProgressBar(len(tiles))
        info_lines = _describe_tiles(tiles, num_x, num_y, img_w, img_h, tile_size)
        info_lines.insert(1, f"Working size after upscale/pad: {work_w}x{work_h}.")

        for info in tiles:
            x, y, w, h = info["x"], info["y"], info["width"], info["height"]
            source_crop = source[:, y:y + h, x:x + w, :]

            if sequential_context:
                # Build this tile's input from what has already been generated where it overlaps
                # earlier tiles, falling back to the untouched source elsewhere. That way the model
                # continues real neighbouring pixels instead of inventing that region from scratch,
                # which is what makes independently generated tiles disagree along their shared edge.
                done = weight_sum[:, y:y + h, x:x + w, :]
                blended = canvas[:, y:y + h, x:x + w, :] / done.clamp(min=1e-8)
                tile_input = torch.where(done > 1e-6, blended, source_crop)
            else:
                tile_input = source_crop

            generated = self._sample_tile(model, positive, negative, vae, tile_input, seed, steps,
                                          cfg, sampler_name, scheduler, denoise)

            if generated.shape[1] != h or generated.shape[2] != w:
                generated = F.interpolate(generated.permute(0, 3, 1, 2), size=(h, w),
                                          mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

            generated = generated.to(device=device, dtype=dtype)
            generated = _color_match(generated, source_crop, color_match, color_match_strength)

            weight = _tile_weight(info, feather, device, dtype)
            canvas[:, y:y + h, x:x + w, :] += generated * weight
            weight_sum[:, y:y + h, x:x + w, :] += weight
            pbar.update(1)

        merged = canvas / weight_sum.clamp(min=1e-8)
        merged = merged[:, :img_h, :img_w, :]

        if final_color_match:
            reference = self._upscale(image, upscale_by)[:, :img_h, :img_w, :]
            merged = _color_match(merged, reference, color_match, color_match_strength)

        return (merged.clamp(0.0, 1.0), "\n".join(info_lines))
