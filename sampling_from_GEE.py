import datetime
import numpy as np
import os

# from common.geoimage.raster_dataset import RasterDataset, RasterDatasetPerBlockTraverser
# from catalog.place.place_api import PlaceAPI, PlaceRecord
# from catalog.models.sentinel2_l2a import Sentinel2L2AScene
# from catalog.raster.raster import retrieve_by_item
import ee
from ee.batch import Export, Task
import time

from common.paii_gee import gee_utils, gcp_utils
from common.paii_gee.gee_utils import composite_by_month
from common.logger import logger


S2_CHANNEL_LIST = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
]


def generate_patch():
    aoi = PlaceRecord.query_many_items(
        target_name="Ramsey County", target_parent_name="Minnesota"
    )[0]
    mgrs_set = PlaceAPI.get_mgrs_set_with_place_name(
        target_name="Ramsey County", target_parent_name="Minnesota", region_level=3
    )
    target_mgrs = list(mgrs_set)[0]
    s2_items = Sentinel2L2AScene.query_many_items(
        mgrs_tile=target_mgrs,
        temporal=(datetime.datetime(2021, 7, 1), datetime.datetime(2021, 7, 6)),
        cloud_cover=[0, 10],
    )
    date = datetime.datetime.strftime(s2_items[0].end_datetime, "%Y%m%d")
    rds = retrieve_by_item(item=s2_items[0], asset_names=S2_CHANNEL_LIST)
    image_block = RasterDatasetPerBlockTraverser(
        rds, 512, 512, 0, 0, boundary_treatment="shift"
    )
    count = 0
    os.makedirs("./assets/satellite-imagery/sentinel2", exist_ok=True)
    for idx, (blocks, _, _) in enumerate(image_block.traverse()):
        if np.random.choice(2, 1, [0.1, 0.9]) == 0 and np.mean(blocks.data == 0) < 0.1:
            blocks.to_geotiff(
                f"./assets/satellite-imagery/sentinel2/{target_mgrs}_{date}_{idx}.tif"
            )
            count += 1
            if count > 500:
                break


def upload_to_gee():
    # gcp_utils.delete_blob(bucket_name)
    # for img in os.listdir(server_folder_upload):
    #     gcp_utils.upload_blob(
    #         "paii_gee", os.path.join(server_folder_upload, img), f"upload/{img}"
    #     )
    # when all files have been uploaded tp Google Cloud Platform
    file_to_upload = gcp_utils.list_blobs(bucket_name)
    for idx, item in enumerate(file_to_upload):
        if "upload" in item and item.endswith(".tif"):
            properties = {"name": item.split(".")[0]}
            if not gee_utils.AssetManager.asset_exists(gee_asset_name):
                ee.data.createAsset({"type": "ImageCollection"}, gee_asset_name)
            gcp_utils.gee_upload_image(
                os.path.join(gee_asset_name, str(idx)),
                f"gs://paii_gee/" + item,
                properties,
            )


def downlowd_landcover():
    server_folder_download = (
        "/NAS6/Members/linchenxi/projects/RS_foundation_model/data"
    )
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
    sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    reference = ee.ImageCollection(
        "users/lin00370/paii/foundation_model/benchmark_ramsey"
    )
    landcover = reference.map(
        lambda img: worldcover.clip(img.geometry().bounds()).copyProperties(
            img, ["name"]
        )
    )
    landcover_lst = landcover.toList(1000)
    for i in range(landcover.size().getInfo()):
        export_img = ee.Image(landcover_lst.get(i))
        filename = ee.String(export_img.get("name")).getInfo()
        task = Export.image.toCloudStorage(
            image=export_img,
            description=f"exporting {i}th result",
            bucket="paii_gee",
            fileNamePrefix=os.path.join("download", filename),
            region=export_img.geometry(),
            maxPixels=1e13,
            scale=10,
        )
        task.start()

    active_task_list = gee_utils.TaskManager.get_active_task()
    active_task_list = [task.state for task in active_task_list]
    while len(active_task_list) != 0:
        logger.info(f"{len(active_task_list)} are still runing. Please wait.")
        time.sleep(300)
        active_task_list = gee_utils.TaskManager.get_active_task()
        active_task_list = [task.state for task in active_task_list]
    gcp_utils.download_blob(
        "paii_gee",
        server_folder_download,
    )


def downlowd_patch_from_points():
    server_folder_download = (
        "/NAS6/Members/linchenxi/RemoteCLIP/assets/satellite-imagery/landcover"
    )
    samples = ee.FeatureCollection(
        "users/lin00370/paii/foundation_model/samples_for_text_generation"
    )
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
    sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(
        "2020-12-01", "2022-12-01"
    )
    for i in range(2):
        l = ee.Feature(samples.toList(1000).get(i))
        square = l.buffer(ee.Number(1280), 1).bounds()
        filtered_s2 = sentinel2.filterDate("2020-12-01", "2022-12-01").map(
            lambda img: img.clip(square)
        )
        composited_s2 = composite_by_month(
            imgCl=filtered_s2,
            region=square.geometry(),
            num_year=3,
            reducer=ee.Reducer.mean(),
        )

        landcover = worldcover.clip(square)
        filename = f"{i}_landcover.tif"
        task_landcover = Export.image.toCloudStorage(
            image=landcover,
            description=f"exporting {i}th land cover",
            bucket="paii_gee",
            fileNamePrefix=os.path.join("download", filename),
            region=landcover.geometry(),
            maxPixels=1e13,
            scale=10,
        )
        task_landcover.start()
        task_sentinel2 = Export.image.toCloudStorage(
            image=composited_s2,
            description=f"exporting {i}th sentinel2",
            bucket="paii_gee",
            fileNamePrefix=os.path.join("download", filename),
            region=landcover.geometry(),
            maxPixels=1e13,
            scale=10,
        )
        task_sentinel2.start()

    # active_task_list = gee_utils.TaskManager.get_active_task()
    # active_task_list = [task.state for task in active_task_list]
    # while len(active_task_list) != 0:
    #     logger.info(f"{len(active_task_list)} are still runing. Please wait.")
    #     time.sleep(300)
    #     active_task_list = gee_utils.TaskManager.get_active_task()
    #     active_task_list = [task.state for task in active_task_list]
    # gcp_utils.download_blob(
    #     "paii_gee",
    #     server_folder_download,
    # )


if __name__ == "__main__":
    bucket_name = "paii_gee"
    gee_asset_name = "users/lin00370/paii/foundation_model/benchmark_ramsey"
    server_folder_upload = (
        "/NAS6/Members/linchenxi/RemoteCLIP/assets/satellite-imagery/sentinel2"
    )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
        "/NAS6/Members/linchenxi/Earth Engine default project-d372e5cfe621.json"
    )

    try:
        ee.Initialize()
    except Exception as e:
        ee.Authenticate()
        ee.Initialize()
    # upload_to_gee()
    downlowd_patch_from_points()
