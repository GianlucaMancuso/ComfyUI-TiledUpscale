import torch.nn.functional as F
import torch
import math
import logging

from einops import rearrange

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management
import comfy.patcher_extension
import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding
import node_helpers

# Tile dimensions are snapped to a multiple of the model's pixels-per-token so a tile maps to a
# whole number of tokens. It is 16 for FLUX.2 (16x VAE, patch 1) and for Krea 2 (8x VAE, patch 2),
# and is read from the loaded model by _pixel_alignment(); this is only the fallback.
LATENT_SCALE = 16

# One-shot flags so optional-dependency warnings don't repeat once per tile.
_warned = {"color_matcher": False}

# Fraction of an overlap band over which a tile's *input* fades from the already-generated
# neighbour back to the plain source. It is deliberately small: the neighbour's pixels have to
# reach the model at full strength across most of the band, or the model stops treating them as
# something to continue. It redraws the subject in its own position instead, and the output blend
# averages the two into a ghost. The fade only exists so the input has no step at the rim.
_CONTEXT_EDGE = 0.25

# How far a tile drifts from its neighbours near its own edges, as a fraction of the tile. Measured
# at ~0.15 on flat content: an overlap band is the one place in the image where two tile edges meet,
# and that is exactly where it shows. Overlapping by twice this keeps the middle of every band out
# of both tiles' edge zones.
_EDGE_FALLOFF = 0.15

# Bounds for an automatically chosen tile. Below 1024 a tile stops carrying enough context to
# refine from; above ~1536 these models are past the resolution they were trained at, and since
# attention cost grows with the square of the token count while the tile count only falls
# linearly with area, the total cost grows quadratically with the tile.
_AUTO_TILE_MIN, _AUTO_TILE_MAX = 1024, 1536

# How much narrower the detail band's transition is than the blend ramp. Exposure and haze differ
# between tiles over hundreds of pixels and need the full ramp to hide; contours a few pixels apart
# must not be averaged at all, or both survive as a doubled edge.
_DETAIL_SHARPNESS = 20.0

# Frequency split between the two, as a fraction of the overlap.
_DETAIL_RADIUS = 0.25


def _pixel_alignment(model):
    """
    How many pixels one model token covers: the VAE's spatial compression times the DiT's patch
    size. Reading it from the loaded model is what makes this work beyond FLUX.2. Krea 2 encodes at
    1/8 with patch 2 and FLUX.2 at 1/16 with patch 1, so both land on 16, while other
    reference-latent models differ.
    """
    try:
        downscale = int(model.get_model_object("latent_format").spacial_downscale_ratio)
    except Exception:
        logging.warning("[TiledUpscale] could not read the model's latent format; assuming %dpx tokens.", LATENT_SCALE)
        return LATENT_SCALE

    patch = 1
    try:
        diffusion_model = model.get_model_object("diffusion_model")
        for attr in ("patch_size", "patch"):
            value = getattr(diffusion_model, attr, None)
            if isinstance(value, (tuple, list)):
                value = max(value) if value else None
            if isinstance(value, int) and value > 0:
                patch = value
                break
    except Exception:
        pass

    return max(1, downscale * patch)


def _reference_method(model, conditioning):
    """
    The method name to attach to the conditioning so the model actually reads its reference
    latents, or None to leave the conditioning as it is.

    Most edit models carry a default (FLUX.2 "index", Qwen "index", FLUX.1 "offset"), but Krea 2
    ships `default_ref_method = None` and *silently drops* every reference latent unless the
    conditioning names one. The tile is then regenerated from the prompt alone and comes back as an
    image with nothing to do with the input. Its own reference workflow uses
    "index_timestep_zero", so that is what we fill in.
    """
    if any("reference_latents_method" in entry[1] for entry in conditioning):
        return None  # set upstream, e.g. by the Edit Model Reference Method node.
    try:
        diffusion_model = model.get_model_object("diffusion_model")
    except Exception:
        return None
    # Only models that declare the attribute *and* leave it unset need us to fill it in.
    if getattr(diffusion_model, "default_ref_method", "unset") is None:
        return "index_timestep_zero"
    return None


# Submodules a Krea 2 SingleStreamDiT exposes; the in-context forward below is built out of them.
_KREA2_PARTS = ("patch", "channels", "tdim", "first", "last", "blocks", "tmlp", "tproj",
                "txtfusion", "txtmlp", "pe_embedder", "_unpack_context")


def _is_krea2(diffusion_model):
    return all(hasattr(diffusion_model, part) for part in _KREA2_PARTS)


def _img_ids(batch, frame, h, w, device):
    """3-axis RoPE ids for one image block: axis 0 is the frame index, axes 1-2 the grid."""
    ids = torch.zeros(h, w, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h * w, 3).repeat(batch, 1, 1)


def _to_4d(latent):
    """(B,C,T,H,W) -> (B*T,C,H,W); 4D passes through. Images are T=1."""
    if latent.ndim == 5:
        b, c, t, h, w = latent.shape
        return latent.reshape(b * t, c, h, w)
    return latent


def _krea2_in_context_forward(dm, x, timesteps, context, source, transformer_options):
    """
    Krea 2's forward with the source latent prepended as a block of clean tokens.

    ComfyUI's native Krea 2 forward is text-to-image: `[text | target]`, with reference latents
    appended *after* the target and given their own timestep-zero modulation. The Krea 2 edit
    LoRAs are trained the other way round, on `[text | source(frame=1) | target(frame=0)]` with one
    shared timestep, the source told apart from the noisy target only by its RoPE frame index, so
    the native path produces noise with them. This rebuilds the sequence the way they expect,
    out of the model's own submodules.

    Layout follows ComfyUI-Krea2Edit (Apache-2.0), which in turn mirrors ai-toolkit's
    predict_velocity_edit.
    """
    patch = dm.patch

    temporal = x.ndim == 5
    if temporal:
        b5, _, t5, _, _ = x.shape
    x = _to_4d(x)
    batch, _, h_orig, w_orig = x.shape

    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    height, width = x.shape[-2:]
    h_, w_ = height // patch, width // patch

    # The target latent arrives already scaled by process_latent_in; the source is scaled by the
    # caller, and only has to be resized onto the target's grid.
    src = _to_4d(source).to(device=x.device, dtype=x.dtype)
    if src.shape[0] != batch:
        src = src[:1].expand(batch, *src.shape[1:])
    if src.shape[-2:] != (height, width):
        src = F.interpolate(src.float(), size=(height, width), mode="bilinear").to(x.dtype)
    src = comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch))

    to_tokens = lambda t: dm.first(rearrange(t, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                                             ph=patch, pw=patch))
    target_tokens = to_tokens(x)
    source_tokens = to_tokens(src)

    t = dm.tmlp(timestep_embedding(timesteps, dm.tdim).unsqueeze(1).to(target_tokens.dtype))
    tvec = dm.tproj(t)

    context = dm.txtfusion(dm._unpack_context(context), mask=None,
                           transformer_options=transformer_options)
    context = dm.txtmlp(context)

    txt_len, src_len, tgt_len = context.shape[1], source_tokens.shape[1], target_tokens.shape[1]
    combined = torch.cat([context, source_tokens, target_tokens], dim=1)

    device = combined.device
    pos = torch.cat([torch.zeros(batch, txt_len, 3, device=device, dtype=torch.float32),
                     _img_ids(batch, 1, h_, w_, device),
                     _img_ids(batch, 0, h_, w_, device)], dim=1)
    freqs = dm.pe_embedder(pos)

    transformer_options = dict(transformer_options)
    transformer_options["total_blocks"] = len(dm.blocks)
    transformer_options["block_type"] = "single"
    transformer_options["img_slice"] = [txt_len + src_len, combined.shape[1]]
    for index, block in enumerate(dm.blocks):
        transformer_options["block_index"] = index
        combined = block(combined, tvec, freqs, None, transformer_options=transformer_options)

    final = dm.last(combined, t)
    out = final[:, txt_len + src_len:txt_len + src_len + tgt_len, :]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_, w=w_, ph=patch, pw=patch, c=dm.channels)
    out = out[:, :, :h_orig, :w_orig]
    if temporal:
        out = out.reshape(b5, t5, dm.channels, h_orig, w_orig).movedim(1, 2)
    return out


def _in_context_model(model, source_holder):
    """
    Clone the model with a forward that reads the current tile out of `source_holder`.

    Cloned once rather than per tile: a fresh ModelPatcher would make ComfyUI re-apply every
    LoRA patch for each tile, and these models are always LoRA-patched.
    """
    patched = model.clone()

    def wrapper(executor, x, timesteps, context, *args, **kwargs):
        source = source_holder["latent"]
        if source is None:
            return executor(x, timesteps, context, *args, **kwargs)
        # Everything after `context` is passed positionally by some ComfyUI versions
        # (attention_mask, ref_latents, transformer_options) and by keyword in others.
        options = kwargs.get("transformer_options")
        if options is None:
            options = next((arg for arg in reversed(args) if isinstance(arg, dict)), {})
        # executor.class_obj is the diffusion model as patched for this step, LoRAs included.
        return _krea2_in_context_forward(executor.class_obj, x, timesteps, context, source, options)

    options = patched.model_options.setdefault("transformer_options", {})
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "tiled_upscale_in_context",
        wrapper, options)
    return patched


# Qwen3-VL user turn with the source image in it. Identical to the stock Krea 2 conditioning
# template (comfy/text_encoders/krea2.py) apart from the vision block, which is the half the
# edit LoRAs were trained with.
_GROUNDING_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "{}<|im_end|>\n<|im_start|>assistant\n"
)


def _grounded_conditioning(clip, prompt, image, grounding_px):
    """
    Encode the prompt with one tile as the text encoder's vision input.

    The Krea 2 edit LoRAs see the source twice during training: as latent tokens (appearance) and
    inside the text encoder's user turn (semantics). Grounding on the whole picture instead would
    describe the entire scene to every tile, and each tile then redraws the whole composition;
    grounding per tile keeps both paths talking about the same pixels.
    """
    samples = image.movedim(-1, 1)
    h, w = samples.shape[2], samples.shape[3]
    if grounding_px and max(h, w) > grounding_px:
        scale = grounding_px / max(h, w)
        samples = comfy.utils.common_upscale(samples, round(w * scale), round(h * scale),
                                             "area", "disabled")
    grounded = samples.movedim(1, -1)[:, :, :, :3]

    tokens = clip.tokenize(prompt, images=[grounded], llama_template=_GROUNDING_TEMPLATE)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    # Park on CPU: one of these per tile would otherwise sit in VRAM beside the model. The
    # sampler moves conditioning onto the device per batch anyway.
    return [[entry[0].to("cpu"),
             {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in entry[1].items()}]
            for entry in conditioning]


def _resolve_reference_mode(model, requested):
    """How the tile is handed to the model: as a conditioning value, or as in-context tokens."""
    if requested != "auto":
        return requested
    try:
        diffusion_model = model.get_model_object("diffusion_model")
    except Exception:
        return "conditioning"
    # A Krea 2 model with no default reference method is the edit-LoRA case.
    if _is_krea2(diffusion_model) and getattr(diffusion_model, "default_ref_method", "unset") is None:
        return "in_context"
    return "conditioning"


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


def _split_frequencies(image, radius):
    """Split a BHWC image into (low, high) about `radius` pixels, so high = image - low."""
    radius = max(1, int(radius))
    bchw = image.permute(0, 3, 1, 2)
    h, w = bchw.shape[-2:]
    small = F.interpolate(bchw, size=(max(1, h // radius), max(1, w // radius)), mode="area")
    low = F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False)
    low = low.permute(0, 2, 3, 1)
    return low, image - low


def _detail_weight(weight, weight_max):
    """
    Turn a blend ramp into a near-exclusive one for the detail band.

    Two tiles drawing the same edge a few pixels apart both survive an average, which is what a
    doubled contour is. Detail therefore comes from whichever tile is dominant at that pixel,
    switching over a narrow strip instead of being mixed across the whole overlap. Sharpening the
    ramp relative to the local maximum keeps the dominant tile at full strength even at a corner
    where four tiles meet and no single weight is anywhere near 1.
    """
    relative = weight / weight_max.clamp(min=1e-8)
    return ((relative - 1.0) * _DETAIL_SHARPNESS + 1.0).clamp(0.0, 1.0)


def _tile_weight(info, feather, device, dtype):
    """2-D weight map for one tile, as the outer product of its two axis ramps."""
    ramp_y = _axis_ramp(info["height"], round(info["fade_top"] * feather), round(info["fade_bottom"] * feather), device, dtype)
    ramp_x = _axis_ramp(info["width"], round(info["fade_left"] * feather), round(info["fade_right"] * feather), device, dtype)
    return (ramp_y.unsqueeze(1) * ramp_x.unsqueeze(0)).unsqueeze(0).unsqueeze(-1)


def _resolve_geometry(width, height, tile_size, overlap, alignment):
    """
    Fill in tile_size and overlap when they are left at 0, and snap them to the token grid.

    Tile size follows the output, aiming for roughly four tiles along the longest side and held
    inside [_AUTO_TILE_MIN, _AUTO_TILE_MAX]. Overlap follows the *tile*, not `upscale_by`: the edge drift
    it has to cover is a property of the tile measured in the tile's own pixels, so the same
    fraction is right whether the image was upscaled 1.5x or 4x.
    """
    auto_tile, auto_overlap = tile_size <= 0, overlap <= 0

    if auto_tile:
        tile_size = min(_AUTO_TILE_MAX, max(_AUTO_TILE_MIN, max(width, height) // 4))
    tile_size = max(alignment, round(tile_size / alignment) * alignment)

    if auto_overlap:
        overlap = round(tile_size * 2 * _EDGE_FALLOFF / alignment) * alignment
    overlap = max(0, min(overlap, tile_size - alignment))
    return tile_size, overlap, auto_tile, auto_overlap


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
            info_lines.append("Note: tiles have differing sizes, so 'Tiles Batch' holds only the first tile. Use 'Tiles List'.")

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
                     "ratio, so this is the closest fit (the longer axis sets the size).")
        return (suggested, info)


class TiledUpscaleRefine:
    CATEGORY = "Tiled Upscale"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "refine"
    DESCRIPTION = ("All-in-one tiled refine/upscale for reference-latent edit models (FLUX.2 klein, "
                   "Krea 2 and similar). Upscales, splits into overlapping tiles, regenerates each tile at full "
                   "tile resolution, colour-matches it back to the source, and blends everything into "
                   "one image. Feed it plain CLIPTextEncode conditioning; the per-tile reference latent "
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
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16, "tooltip":
                    "Resolution each tile is regenerated at. 0 picks one from the output size: "
                    "about four tiles along the longest side, kept between 1024 and 1536. Raise it "
                    "by hand for fewer seams if you have the VRAM, keeping in mind that cost grows "
                    "with the square of the tile."}),
                "overlap": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 16, "tooltip":
                    "How much neighbouring tiles share. 0 picks 30% of the tile size: a tile drifts "
                    "from its neighbours over roughly the outer 15% of its width, and an overlap "
                    "band is where two tile edges meet, so it has to be wide enough that the middle "
                    "of the band is outside both edge zones. Depends on the tile, not on "
                    "upscale_by."}),
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
            },
            "optional": {
                "reference_mode": (["auto", "conditioning", "in_context"], {"default": "auto", "tooltip":
                    "How each tile is handed to the model as its own reference. 'conditioning' is "
                    "the reference-latent path FLUX.2, Qwen and the Krea 2 style-reference LoRA "
                    "use. 'in_context' prepends the tile as clean source tokens inside the model's "
                    "forward, which is what the Krea 2 edit LoRAs are trained on. 'auto' picks "
                    "in_context for Krea 2 and conditioning for everything else."}),
                "clip": ("CLIP", {"tooltip":
                    "Krea 2 only, and only in the in_context mode. Connect the Krea 2 CLIP to "
                    "re-encode the prompt per tile with that tile as the text encoder's vision "
                    "input, the semantic half the edit LoRAs are trained with. Overrides the "
                    "'positive' input. Without it the model gets no image grounding and tends to "
                    "ignore the reference entirely."}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip":
                    "The instruction to encode per tile. Only used when 'clip' is connected."}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 64, "tooltip":
                    "Longest side of the tile fed to the text encoder; 0 keeps the tile's own "
                    "resolution. Lower favours edit adherence, higher favours likeness."}),
                "redo_first_tile": ("BOOLEAN", {"default": True, "tooltip":
                    "Regenerate the top-left tile once more at the end, when the tiles around it "
                    "are finished. The first tile is otherwise the only one generated without any "
                    "refined neighbour to continue from, which is what makes it sharpen backgrounds "
                    "or hallucinate detail the other tiles don't. Costs one extra tile of sampling "
                    "time; needs sequential_context."}),
            },
        }

    def _upscale(self, image, factor, alignment=LATENT_SCALE):
        if factor == 1.0:
            return image
        _, h, w, _ = image.shape
        new_h = max(alignment, round(h * factor / alignment) * alignment)
        new_w = max(alignment, round(w * factor / alignment) * alignment)
        bchw = image.permute(0, 3, 1, 2)
        bchw = comfy.utils.common_upscale(bchw, new_w, new_h, "lanczos", "disabled")
        return bchw.permute(0, 2, 3, 1)

    def _sample_tile(self, model, sampling_model, positive, negative, vae, tile_pixels, seed, steps,
                     cfg, sampler_name, scheduler, denoise, reference_method=None,
                     source_holder=None):
        """Regenerate one tile: encode it, hand it to the model as reference, sample, decode."""
        tile_latent = vae.encode(tile_pixels[:, :, :, :3])

        if source_holder is not None:
            # In-context path: the tile enters inside the model's forward as clean source tokens,
            # scaled into the same latent space the sampler works in.
            source_holder["latent"] = model.model.process_latent_in(tile_latent)
            positive_tile = positive
        else:
            # Same thing the ReferenceLatent node does: the tile guides its own regeneration.
            positive_tile = node_helpers.conditioning_set_values(
                positive, {"reference_latents": [tile_latent]}, append=True)
            if reference_method is not None:
                positive_tile = node_helpers.conditioning_set_values(
                    positive_tile, {"reference_latents_method": reference_method})

        # Sampling starts from the tile's own latent, which also gives the sampler exactly the
        # layout this model works in: channel count, spatial size, and the extra temporal axis that
        # video-style VAEs (Krea 2 uses one) add, instead of FLUX.2's 128 channels hardcoded.
        #
        # Starting from an empty latent instead would make `denoise` meaningless: these are flow
        # models, so the sampler forms `sigma * noise + (1 - sigma) * latent`, and an empty latent
        # turns denoise=0.8 into "80% noise over black" rather than "renoise the tile by 80%".
        # At denoise=1.0, sigma is 1 and the tile scales out completely, so the default is
        # unaffected.
        latent = tile_latent.to(device=comfy.model_management.intermediate_device(),
                                dtype=comfy.model_management.intermediate_dtype())
        latent = comfy.sample.fix_empty_latent_channels(model, latent)

        noise = comfy.sample.prepare_noise(latent, seed, None)
        samples = comfy.sample.sample(sampling_model, noise, steps, cfg, sampler_name, scheduler,
                                      positive_tile, negative, latent, denoise=denoise,
                                      disable_pbar=True, seed=seed)

        decoded = vae.decode(samples)
        if len(decoded.shape) == 5:
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        return decoded

    def refine(self, model, positive, negative, vae, image, upscale_by, tile_size, overlap, feather,
               seed, steps, cfg, sampler_name, scheduler, denoise, sequential_context,
               color_match, color_match_strength, final_color_match, redo_first_tile=True,
               reference_mode="auto", clip=None, prompt="", grounding_px=768):
        alignment = _pixel_alignment(model)
        mode = _resolve_reference_mode(model, reference_mode)
        if mode == "in_context":
            source_holder = {"latent": None}
            sampling_model = _in_context_model(model, source_holder)
            reference_method = None
        else:
            source_holder = None
            sampling_model = model
            reference_method = _reference_method(model, positive)
        source = self._upscale(image, upscale_by, alignment)
        _, img_h, img_w, _ = source.shape

        # Tiles must land on whole latent cells, so pad the image up to a multiple of the token
        # size and give every tile a size the VAE can round-trip exactly.
        pad_h = (-img_h) % alignment
        pad_w = (-img_w) % alignment
        if pad_h or pad_w:
            bchw = source.permute(0, 3, 1, 2)
            bchw = F.pad(bchw, (0, pad_w, 0, pad_h), mode="replicate")
            source = bchw.permute(0, 2, 3, 1)
        work_h, work_w = img_h + pad_h, img_w + pad_w

        tile_size, overlap, auto_tile, auto_overlap = _resolve_geometry(
            work_w, work_h, tile_size, overlap, alignment)
        tiles, num_x, num_y = _plan_tiles(work_h, work_w, tile_size, overlap)

        device, dtype = source.device, source.dtype
        # Two blends run side by side: the broad one carries exposure and haze, which differ between
        # tiles over hundreds of pixels and need the full ramp to hide, and the near-exclusive one
        # carries detail, which must come from a single tile or contours a few pixels apart survive
        # twice. `weight_max` is what lets the detail blend know which tile is dominant; the weights
        # only depend on the grid, so it is built before anything is generated.
        canvas = torch.zeros(1, work_h, work_w, source.shape[-1], device=device, dtype=dtype)
        weight_sum = torch.zeros(1, work_h, work_w, 1, device=device, dtype=dtype)
        detail = torch.zeros_like(canvas)
        detail_sum = torch.zeros_like(weight_sum)
        weight_max = torch.zeros_like(weight_sum)
        # `weight_max` covers tiles that do not exist yet, so a freshly placed tile contributes no
        # detail wherever a later tile will dominate. That is right for the finished image but not
        # for what the next tile is shown: read mid-run, those regions would be low band only, and
        # a tile handed a blurred neighbour paints a blurred band. Context therefore accumulates
        # separately, as the plain weighted average of everything generated so far.
        context = torch.zeros_like(canvas)
        for info in tiles:
            region = (slice(None), slice(info["y"], info["y"] + info["height"]),
                      slice(info["x"], info["x"] + info["width"]), slice(None))
            weight_max[region] = torch.maximum(weight_max[region],
                                               _tile_weight(info, feather, device, dtype))
        detail_radius = max(1, round(overlap * _DETAIL_RADIUS))

        def merged_image():
            return (canvas / weight_sum.clamp(min=1e-8)) + (detail / detail_sum.clamp(min=1e-8))

        grounded = clip is not None and mode == "in_context"
        if clip is not None and not grounded:
            logging.warning("[TiledUpscale] 'clip' only applies to the in_context reference mode; "
                            "ignoring it and using the 'positive' input as-is.")

        tile_positive = tile_negative = None
        if grounded:
            # Encoded up front, in one pass, so the text encoder is loaded once instead of being
            # swapped in against the diffusion model on every tile. Grounding uses the untouched
            # source crop rather than the sequential-context input: the semantics of a tile are a
            # property of the original picture, while the appearance path still gets the blended
            # pixels that make neighbouring tiles line up.
            tile_positive, tile_negative = [], ([] if cfg > 1.0 else None)
            for info in tiles:
                crop = source[:, info["y"]:info["y"] + info["height"],
                              info["x"]:info["x"] + info["width"], :]
                tile_positive.append(_grounded_conditioning(clip, prompt, crop, grounding_px))
                if tile_negative is not None:
                    # Training's unconditional is the same image with an empty instruction.
                    tile_negative.append(_grounded_conditioning(clip, "", crop, grounding_px))

        redo_first = redo_first_tile and sequential_context and len(tiles) > 1
        pbar = comfy.utils.ProgressBar(len(tiles) + (1 if redo_first else 0))
        info_lines = _describe_tiles(tiles, num_x, num_y, img_w, img_h, tile_size)
        info_lines.insert(1, f"Working size after upscale/pad: {work_w}x{work_h}, "
                             f"{alignment}px per model token."
                             + (f" Auto tile_size={tile_size}." if auto_tile else "")
                             + (f" Auto overlap={overlap}." if auto_overlap else ""))
        info_lines.insert(2, f"Reference mode: {mode}"
                             + (" (auto)" if reference_mode == "auto" else "")
                             + (f", prompt grounded per tile at {grounding_px}px" if grounded else ""))
        if reference_method is not None:
            info_lines.insert(3, f"This model has no default reference method, using "
                                 f"'{reference_method}' so the tile is actually used as reference.")

        def place_tile(info, previous=None):
            """Regenerate one tile and blend it into the canvas. Returns the generated tile."""
            x, y, w, h = info["x"], info["y"], info["width"], info["height"]
            region = (slice(None), slice(y, y + h), slice(x, x + w), slice(None))
            weight = _tile_weight(info, feather, device, dtype)
            detail_weight = _detail_weight(weight, weight_max[region])

            if previous is not None:
                # Take the earlier attempt back out, so the context this tile now sees is its
                # neighbours' output rather than a blend that still contains itself.
                low, high = _split_frequencies(previous, detail_radius)
                canvas[region] -= low * weight
                weight_sum[region] -= weight
                detail[region] -= high * detail_weight
                detail_sum[region] -= detail_weight
                context[region] -= previous * weight

            source_crop = source[region]
            if sequential_context:
                # Build this tile's input from what has already been generated where it overlaps
                # earlier tiles, fading into the untouched source elsewhere. That way the model
                # continues real neighbouring pixels instead of inventing that region from scratch,
                # which is what makes independently generated tiles disagree along their shared edge.
                #
                # Coverage drives a crossfade rather than a switch, but a steep one. Switching on
                # any non-zero coverage put a hard edge in the model's input at the outer rim of
                # the overlap. Coverage there is ~0, yet `blended` is normalised, so it is already
                # the neighbour's full-strength output sitting right beside untouched source, and
                # the model redrew that edge as a visible line. Fading across the whole band instead
                # fixes the line but leaves the neighbour at a fraction of its strength through the
                # middle of the band, too weak to anchor a subject, so the model repositions it and
                # the output blend turns the disagreement into a ghost.
                done = weight_sum[region]
                blended = context[region] / done.clamp(min=1e-8)
                alpha = (done / _CONTEXT_EDGE).clamp(0.0, 1.0)
                tile_input = blended * alpha + source_crop * (1.0 - alpha)
            else:
                tile_input = source_crop

            tile_id = info["tile_id"]
            generated = self._sample_tile(
                model, sampling_model,
                tile_positive[tile_id] if tile_positive else positive,
                tile_negative[tile_id] if tile_negative else negative,
                vae, tile_input, seed, steps, cfg, sampler_name, scheduler, denoise,
                reference_method, source_holder)

            if generated.shape[1] != h or generated.shape[2] != w:
                generated = F.interpolate(generated.permute(0, 3, 1, 2), size=(h, w),
                                          mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

            generated = generated.to(device=device, dtype=dtype)
            generated = _color_match(generated, source_crop, color_match, color_match_strength)

            low, high = _split_frequencies(generated, detail_radius)
            canvas[region] += low * weight
            weight_sum[region] += weight
            detail[region] += high * detail_weight
            detail_sum[region] += detail_weight
            context[region] += generated * weight
            pbar.update(1)
            return generated

        first_tile = None
        for index, info in enumerate(tiles):
            generated = place_tile(info)
            if index == 0 and redo_first:
                first_tile = generated

        if redo_first:
            # The first tile is the only one generated with no already-refined neighbour to
            # continue from, so it is free to sharpen or invent detail the rest of the image never
            # gets. Running it once more at the end, now surrounded by finished tiles, gives it
            # the same context every other tile had.
            place_tile(tiles[0], previous=first_tile)
            info_lines.append("Tile 0 was regenerated at the end with its neighbours as context.")

        merged = merged_image()[:, :img_h, :img_w, :]

        if final_color_match:
            reference = self._upscale(image, upscale_by, alignment)[:, :img_h, :img_w, :]
            merged = _color_match(merged, reference, color_match, color_match_strength)

        return (merged.clamp(0.0, 1.0), "\n".join(info_lines))
