import numpy as np
import os
from osgeo import gdal
import osgeo.ogr as ogr
import osgeo.osr as osr



def reproject(
        in_dir: str,
        out_dir: str,
        CRS: str = "EPSG:4326"):
    """Given input directory and output directory, reprojecting all imagery into the target projection

    Parameters
    in_dir, str
    out_dir, str
    CRS, str, by default "EPSG:4326"
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for item in os.listdir(in_dir):
        if item in os.listdir(out_dir):
            print (f"the file {item} has been reprojected and will skip")
            continue
        if item.endswith('tif'):
            ds = gdal.Open(os.path.join(in_dir, item))
            ds_repro = gdal.Warp(os.path.join(out_dir, item), ds, dstSRS=CRS)
            ds = None
            ds_repro = None

def resample(
        in_dir: str,
        out_dir: str,
        target_res: int,
        target_crs: str):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for item in os.listdir(in_dir):
        if item in os.listdir(out_dir):
            print('existed!')
            return
        try:
            ds = gdal.Open(os.path.join(in_dir, item))
            ds_Res = gdal.Warp(os.path.join(out_dir, item), ds, xRes=target_res, yRes=target_res)
            ds = None
            ds_Res = None
        except:
            pass

if __name__ == "__main__":
    resample()
