from pathlib import Path
import rasterio
from rasterio.enums import Resampling

input_folder = Path("data/BN Imagerial Data/hwh-ortho/2025/Ortho/1/8/beelden_RGB_jpeg_tegels")
output_folder = Path("data_downsampled/BN Imagerial Data/hwh-ortho/2025/Ortho/1/8/beelden_RGB_jpeg_tegels")
output_folder.mkdir(parents=True, exist_ok=True)

max_size = 500  # target max pixels on longest side

for tif_path in input_folder.glob("*.tif"):
    with rasterio.open(tif_path) as src:
        scale = max(src.width / max_size, src.height / max_size, 1.0)
        out_width = max(1, int(src.width / scale))
        out_height = max(1, int(src.height / scale))
        img = src.read(
            out_shape=(src.count, out_height, out_width),
            resampling=Resampling.nearest
        )
        profile = src.profile
        profile.update({
            "height": out_height,
            "width": out_width,
            "transform": src.transform * src.transform.scale(
                (src.width / out_width),
                (src.height / out_height)
            )
        })
        # Force JPEG compression for YCBCR photometric
        if profile.get("photometric") == "YCBCR":
            profile.update({
                "compress": "JPEG",
                "photometric": "YCBCR"
            })
        out_path = output_folder / tif_path.name
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(img)
    print(f"Saved downsampled: {out_path}")