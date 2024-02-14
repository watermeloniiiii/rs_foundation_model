import os
import numpy as np
import numpy.random as random 
from common.geoimage.raster_dataset import RasterDataset, RasterDatasetPerBlockTraverser
from common.logger import logger
from common.img_utils.img_color import color_stretch_normal
from common.img_utils.img_geom import rotate, flip


np.random.seed(1000)
class patch_generator:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def generate_nonoverlap_patch(in_dir, out_dir, x, y, unique_id, mode="training", augment=True):
        img_out_dir = os.path.join(out_dir, "patch", unique_id, "image")
        os.makedirs(img_out_dir, exist_ok=True)
        logger.info(f"Successfully generated folder for image at {img_out_dir}")
        label_out_dir = os.path.join(out_dir, "patch", unique_id, "label")
        os.makedirs(label_out_dir, exist_ok=True)
        logger.info(f"Successfully generated folder for label at {label_out_dir}")
        image = RasterDataset.from_file(in_dir)
        label = RasterDataset.from_file(in_dir.replace("img", "label"))
        image_block = RasterDatasetPerBlockTraverser(image, x, y, 56, 56, boundary_treatment="shift").traverse()
        label_block = list(RasterDatasetPerBlockTraverser(label, x, y, 56, 56, boundary_treatment="shift").traverse())
        for idx, (img, _, _) in enumerate(image_block):
            if img.data.dtype == 'uint8':
                if mode == "training" and (img.data == 255).any():
                    continue
                img.data[img.data == 255] = 0
                stretched = color_stretch_normal(img.data, dest_min=0, dest_max=255)
                img.data = stretched
            if img.data.dtype == 'uint16':
                if mode == "training" and (img.data == 65535).any():
                    continue
                img.data[img.data == 65535] = 0
                stretched = color_stretch_normal(img.data, dest_min=0, dest_max=255)
                img.data = stretched
            if np.mean(img.data!=0) < 0.5:
                continue
            lbl = label_block[idx][0]
            lbl.data[lbl.data > 1] = 0
            if mode == "training":
                if np.mean(lbl.data == 1) == 0 and random.choice(2, 1, [0.3, 0.7]) == 0:
                    continue
            if not os.path.exists(os.path.join(img_out_dir, f"{unique_id}_{idx}.tif")):
                img.to_geotiff(os.path.join(img_out_dir, f"{unique_id}_{idx}.tif"))
            if not os.path.exists(os.path.join(label_out_dir, f"{unique_id}_{idx}.tif")):
                lbl.to_geotiff(os.path.join(label_out_dir, f"{unique_id}_{idx}.tif"))
            if augment:
                random.seed(1001)
                rotate_angle = random.choice([90, 180, 270], 1, [1/3] * 3)[0]
                directions = {0: "horizon", 1: "vertical", 2: "horizon_vertical"}
                flip_direction = directions[random.choice(3, 1, [1/3] * 3)[0]]
                rotated_img = rotate(img.data, rotate_angle)
                flipped_img = flip(img.data, 
                                   horizon=("horizon" in flip_direction),
                                   vertical=("vertical" in flip_direction))
                rotated_lbl = rotate(lbl.data, rotate_angle)
                flipped_lbl = flip(lbl.data, 
                                   horizon=("horizon" in flip_direction),
                                   vertical=("vertical" in flip_direction))
                img.data = rotated_img
                lbl.data = rotated_lbl
                img.to_geotiff(os.path.join(img_out_dir, f"{unique_id}_{idx}_{rotate_angle}.tif"))
                lbl.to_geotiff(os.path.join(label_out_dir, f"{unique_id}_{idx}_{rotate_angle}.tif"))
                img.data = flipped_img
                lbl.data = flipped_lbl
                img.to_geotiff(os.path.join(img_out_dir, f"{unique_id}_{idx}_{flip_direction}.tif"))
                lbl.to_geotiff(os.path.join(label_out_dir, f"{unique_id}_{idx}_{flip_direction}.tif"))

            

    def generate_overlap_patch():
        pass

if __name__ == "__main__":
    regions = [f"region{i}" for i in range(1, 5)]
    for region in regions:
        patch_generator.generate_nonoverlap_patch(
            in_dir=f"/NAS6/Members/linchenxi/projects/morocco/data/img/{region}.tif",
            out_dir="/NAS6/Members/linchenxi/projects/morocco/data",
            x=112,
            y=112,
            unique_id=region
            )

