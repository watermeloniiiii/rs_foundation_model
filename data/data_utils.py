import os
import numpy as np
import numpy.random as random
import cv2
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours
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
        out_dir = os.path.join(dir, unique_id, dtype)
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
    def generate_patch_by_row_column(
        in_dir: str,
        out_dir: str,
        x: int = 112,
        y: int = 112,
        x_delta: int = 56,
        y_delta: int = 56,
        unique_id: str = "",
        mode="training",
        augment=True,
        generate_label=True,
    ):
        if mode == "testing":
            augment = False
        img_out_dir = patch_generator._makedirs_for_patch(out_dir, unique_id)
        label = None
        if generate_label:
            lb_out_dir = patch_generator._makedirs_for_patch(
                out_dir, unique_id, "label"
            )
            label = RasterDataset.from_file(in_dir.replace("img", "label"))
        image = RasterDataset.from_file(in_dir)
        image_block = RasterDatasetPerBlockTraverser(
            image, x, y, x_delta, y_delta, boundary_treatment="shift"
        )
        label_block = (
            RasterDatasetPerBlockTraverser(
                label, x, y, x_delta, y_delta, boundary_treatment="shift"
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
                        "all_mode": True,
                    }
                    patch_generator._patch_augment(img, lb, **kwargs)

    def generate_patch_random(
        in_dir: str,
        out_dir: str,
        x: int = 112,
        y: int = 112,
        x_delta: int = 56,
        y_delta: int = 56,
        unique_id: str = "",
        num_patch=500,
    ):
        img_out_dir = patch_generator._makedirs_for_patch(out_dir, unique_id)
        lb_out_dir = patch_generator._makedirs_for_patch(out_dir, unique_id, "label")
        label = RasterDataset.from_file(in_dir.replace("img", "label"))
        image = RasterDataset.from_file(in_dir)
        image_block = RasterDatasetPerBlockTraverser(
            image, x, y, x_delta, y_delta, boundary_treatment="shift"
        )
        label_block = (
            RasterDatasetPerBlockTraverser(
                label, x, y, x_delta, y_delta, boundary_treatment="shift"
            ).traverse()
            if label
            else _placeholder(image_block.n_blocks)
        )
        count = 0
        for idx, ((img, _, _), (lb, _, _)) in enumerate(
            zip(image_block.traverse(), label_block)
        ):
            # if there is no label image, store all image patches for testing purpose
            if not isinstance(lb, RasterDataset):
                _export_patch(img, img_out_dir, f"{unique_id}_{idx}.tif")
            else:
                img.data[img.data == NODATA[str(img.data.dtype)]] = 0
                stretched = color_stretch_normal(img.data, dest_min=0, dest_max=254)
                img.data = stretched
                lb.data[lb.data > 1] = 0
                if np.mean(lb.data == 1) < 0.1:
                    continue
                _export_patch(img, img_out_dir, f"{unique_id}_{idx}.tif")
                _export_patch(lb, lb_out_dir, f"{unique_id}_{idx}.tif")
                count += 1
                if count > num_patch:
                    break


class bbox_generator:
    def __init__(self) -> None:
        pass

    """ Creating a directory """

    def create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    """ Convert a mask to border image """

    def mask_to_border(self, mask):
        h, w = mask.shape
        border = np.zeros((h, w))

        contours = find_contours(mask, 128)
        for contour in contours:
            for c in contour:
                x = int(c[0])
                y = int(c[1])
                border[x][y] = 255

        return border

    """ Mask to bounding boxes """

    def mask_to_bbox(self, mask):
        bboxes = []

        mask = self.mask_to_border(mask)
        lbl = label(mask)
        props = regionprops(lbl)
        for prop in props:
            x1 = prop.bbox[1]
            y1 = prop.bbox[0]

            x2 = prop.bbox[3]
            y2 = prop.bbox[2]

            bboxes.append([x1, y1, x2, y2])

        return bboxes

    def parse_mask(self, mask):
        mask = np.expand_dims(mask, axis=-1)
        mask = np.concatenate([mask, mask, mask], axis=-1)
        return mask

    def main(self):
        """Load the dataset"""
        images = sorted(glob(os.path.join("data", "image", "*")))
        masks = sorted(glob(os.path.join("data", "mask", "*")))

        """ Create folder to save images """
        self.create_dir("results")

        """ Loop over the dataset """
        for x, y in tqdm(zip(images, masks), total=len(images)):
            """Extract the name"""
            name = x.split("/")[-1].split(".")[0]

            """ Read image and mask """
            x = cv2.imread(x, cv2.IMREAD_COLOR)
            y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

            """ Detecting bounding boxes """
            bboxes = self.mask_to_bbox(y)

            """ marking bounding box on image """
            for bbox in bboxes:
                x = cv2.rectangle(
                    x, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2
                )

            """ Saving the image """
            cat_image = np.concatenate([x, self.parse_mask(y)], axis=1)
            cv2.imwrite(f"results/{name}.png", cat_image)


if __name__ == "__main__":
    regions = [f"region{i}" for i in range(5, 6)]
    generator_config = {
        "type": "row_col",
        "x_size": 224,
        "y_size": 224,
        "x_delta": 0,
        "y_delta": 0,
        "mode": "testing",
    }
    for region in regions:
        if generator_config["type"] == "row_col":
            patch_generator.generate_patch_by_row_column(
                in_dir=f"/NAS6/Members/linchenxi/projects/morocco/data/img/{region}.tif",
                out_dir="/NAS6/Members/linchenxi/projects/morocco/data/patch",
                x=generator_config["x_size"],
                y=generator_config["y_size"],
                x_delta=generator_config["x_delta"],
                y_delta=generator_config["y_delta"],
                unique_id=region,
                mode=generator_config["mode"],
                generate_label=False,
            )
        if generator_config["type"] == "random":
            patch_generator.generate_patch_random(
                in_dir=f"/NAS6/Members/linchenxi/projects/morocco/data/img/{region}.tif",
                out_dir="/NAS6/Members/linchenxi/projects/morocco/data/balanced_test",
                x=generator_config["x_size"],
                y=generator_config["y_size"],
                x_delta=generator_config["x_delta"],
                y_delta=generator_config["y_delta"],
                unique_id=region,
                num_patch=500,
            )
