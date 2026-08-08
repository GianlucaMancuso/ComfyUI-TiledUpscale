from .nodes import (
    TileSplit,
    TileMerge,
    TileGridAdvisor,
    TiledUpscaleRefine,
)

NODE_CLASS_MAPPINGS = {
    "TileSplit": TileSplit,
    "TileMerge": TileMerge,
    "TileGridAdvisor": TileGridAdvisor,
    "TiledUpscaleRefine": TiledUpscaleRefine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TileSplit": "Tile Split",
    "TileMerge": "Tile Merge",
    "TileGridAdvisor": "Tile Grid Advisor",
    "TiledUpscaleRefine": "Tiled Upscale & Refine",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
