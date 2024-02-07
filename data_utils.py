import os
import numpy as np
from common.geoimage.raster_dataset import RasterDataset, RasterDatasetPerBlockTraverser
from common.logger import logger

np.random.seed(1000)
class patch_generator:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def generate_nonoverlap_patch(in_dir, out_dir, x, y, unique_id):
        img_out_dir = os.path.join(out_dir, "patch", unique_id, "image")
        os.makedirs(img_out_dir, exist_ok=True)
        logger.info(f"Successfully generated folder for image")
        label_out_dir = os.path.join(out_dir, "patch", unique_id, "label")
        os.makedirs(label_out_dir, exist_ok=True)
        logger.info(f"Successfully generated folder for label")
        image = RasterDataset.from_file(in_dir)
        label = RasterDataset.from_file(in_dir.replace("img", "label"))
        image_block = RasterDatasetPerBlockTraverser(image, x, y, 112, 112).traverse()
        label_block = list(RasterDatasetPerBlockTraverser(label, x, y, 112, 112).traverse())
        for idx, (img, _, _) in enumerate(image_block):
            if np.mean(img.data!=0) < 0.5:
                continue
            if np.mean(label_block[idx][0].data == 1) == 0:
                continue
            img.to_geotiff(os.path.join(img_out_dir, f"{unique_id}_{idx}.tif"))
            label_block[idx][0].to_geotiff(os.path.join(label_out_dir, f"{unique_id}_{idx}.tif"))



        

    def generate_overlap_patch():
        pass

if __name__ == "__main__":
    regions = [f"region{i}" for i in range(1, 7)]
    for region in regions:
        patch_generator.generate_nonoverlap_patch(
            in_dir=f"/NAS6/Members/linchenxi/projects/morocco/data/img/{region}.tif",
            out_dir="/NAS6/Members/linchenxi/projects/morocco/data",
            x=224,
            y=224,
            unique_id=region
            )

