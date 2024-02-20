import os
import numpy as np
import numpy.random as random
from common.geoimage.raster_dataset import RasterDataset, RasterDatasetPerBlockTraverser
from common.logger import logger
from common.img_utils.img_color import color_stretch_normal
from common.img_utils.img_geom import rotate, flip

np.random.seed(1000)

NODATA = {"uint8": 255, "uint16": 65535}
DIRECTION = {0: "horizon", 1: "vertical", 2: "horizon_vertical"}


def _export_patch(img, dir, filename):
    if not os.path.exists(os.path.join(dir, filename)):
        img.to_geotiff(os.path.join(dir, filename))
    else:
        logger.info(f"{os.path.join(dir, filename)} already existed. Will skip.")


def _placeholder(length):
    for _ in range(length):
        yield (1, 1, 1)


class patch_generator:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _makedirs_for_patch(dir, unique_id, dtype="image"):
        out_dir = os.path.join(dir, "patch", unique_id, dtype)
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"Successfully generated folder for image at {out_dir}")
        return out_dir

    def _patch_augment(img: RasterDataset, label: RasterDataset, **kwargs) -> None:
        random.seed(1001)

        img_out_dir = kwargs["img_out_dir"]
        lb_out_dir = kwargs["lb_out_dir"]
        unique_id = kwargs["unique_id"]
        idx = kwargs["idx"]
        all_mode = kwargs["all_mode"]

        if not all_mode:
            rotate_angles = random.choice([90, 180, 270], 1, [1 / 3] * 3)
            flip_directions = DIRECTION[random.choice(3, 1, [1 / 3] * 3)[0]]
        else:
            rotate_angles = [90, 180, 270]
            flip_directions = ["horizon", "vertical", "horizon_vertical"]

        for rotate_angle, flip_direction in zip(rotate_angles, flip_directions):
            rotated_img = rotate(img.data, rotate_angle)
            flipped_img = flip(
                img.data,
                horizon=("horizon" in flip_direction),
                vertical=("vertical" in flip_direction),
            )
            rotated_lbl = rotate(label.data, rotate_angle)
            flipped_lbl = flip(
                label.data,
                horizon=("horizon" in flip_direction),
                vertical=("vertical" in flip_direction),
            )
            img.data = rotated_img
            label.data = rotated_lbl
            _export_patch(img, img_out_dir, f"{unique_id}_{idx}_{rotate_angle}.tif")
            _export_patch(label, lb_out_dir, f"{unique_id}_{idx}_{rotate_angle}.tif")
            img.data = flipped_img
            label.data = flipped_lbl
            _export_patch(img, img_out_dir, f"{unique_id}_{idx}_{flip_direction}.tif")
            _export_patch(label, lb_out_dir, f"{unique_id}_{idx}_{flip_direction}.tif")

    @staticmethod
    def generate_nonoverlap_patch(
        in_dir,
        out_dir,
        x,
        y,
        unique_id,
        mode="training",
        augment=True,
        generate_label=True,
    ):
        img_out_dir = patch_generator._makedirs_for_patch(out_dir, unique_id)
        label = None
        if generate_label:
            lb_out_dir = patch_generator._makedirs_for_patch(
                out_dir, unique_id, "label"
            )
            label = RasterDataset.from_file(in_dir.replace("img", "label"))
        image = RasterDataset.from_file(in_dir)
        image_block = RasterDatasetPerBlockTraverser(
            image, x, y, 56, 56, boundary_treatment="shift"
        )
        label_block = (
            RasterDatasetPerBlockTraverser(
                label, x, y, 56, 56, boundary_treatment="shift"
            ).traverse()
            if label
            else _placeholder(image_block.n_blocks)
        )
        for idx, ((img, _, _), (lb, _, _)) in enumerate(
            zip(image_block.traverse(), label_block)
        ):
            # if there is no label image, store all image patches for testing purpose
            if not isinstance(lb, RasterDataset):
                _export_patch(img, img_out_dir, f"{unique_id}_{idx}.tif")
            else:
                # if we are working on training dataset, two requirements are needed:
                # 1. all data is valid
                # 2. only 30% of patches with no pos samples will be included
                if mode == "training":
                    if (img.data == NODATA[str(img.data.dtype)]).any() or (
                        np.mean(lb.data == 1) == 0
                        and random.choice(2, 1, [0.3, 0.7]) == 0
                    ):
                        continue
                img.data[img.data == NODATA[str(img.data.dtype)]] = 0
                stretched = color_stretch_normal(img.data, dest_min=0, dest_max=254)
                img.data = stretched
                lb.data[lb.data > 1] = 0
                _export_patch(img, img_out_dir, f"{unique_id}_{idx}.tif")
                _export_patch(lb, lb_out_dir, f"{unique_id}_{idx}.tif")
                if augment:
                    kwargs = {
                        "img_out_dir": img_out_dir,
                        "lb_out_dir": lb_out_dir,
                        "unique_id": unique_id,
                        "idx": idx,
                        "all_mode": False,
                    }
                    patch_generator._patch_augment(img, lb, **kwargs)

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
            unique_id=region,
            generate_label=True,
        )
