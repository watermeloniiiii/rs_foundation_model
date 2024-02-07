import numpy as np
from osgeo import gdal
import os
import pandas as pd
from skimage import io
import osgeo.ogr as ogr
import osgeo.osr as osr

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
    src = (src / 255) ** invGamma * 255
    return src.astype(np.uint8)

def scale_percentile_n(matrix):
    d, w, h = matrix.shape
    for i in range(d):
        mins = np.percentile(matrix[i][matrix[i] != 0], 1)
        maxs = np.percentile(matrix[i], 99)
        matrix[i] = matrix[i].clip(mins, maxs)
        matrix[i] = ((matrix[i] - mins) / (maxs - mins) * 255).astype(np.uint8)
        matrix[i] = gammaCorrection(matrix[i], 1.3)
        print(np.max(matrix[i]), np.min(matrix[i]))
    return matrix

def non_overlap_segmentation(img, folder, x_range, y_range, target=False, require_proj=False, continue_count=False):
    if require_proj:
        img_geotrans = img.GetGeoTransform()
        img_proj = img.GetProjection()
        top_left_x = img_geotrans[0]
        w_e_pixel_resolution = img_geotrans[1]
        top_left_y = img_geotrans[3]
        n_s_pixel_resolution = img_geotrans[5]

    x_num = img.RasterXSize // x_range
    y_num = img.RasterYSize // y_range
    x_size, y_size, x_off, y_off = img.RasterXSize, img.RasterYSize, 0, 0
    img_array = img.ReadAsArray(x_off, y_off, x_size, y_size)

    if continue_count:
        original_count = len(os.listdir(os.path.join(folder, 'input')))
    else:
        original_count = 0

    for i in range(0, x_num):
        for j in range(0, y_num):
            x_off_patch = i * x_range
            y_off_patch = j * y_range
            if not target:
                patch = img_array[:, y_off_patch:y_off_patch + y_range, x_off_patch:x_off_patch + x_range]
                swap_patch = patch.copy()
                swap_patch[0] = patch[2]
                swap_patch[2] = patch[0]
                valid_ratio = (swap_patch != 0).mean()
                # patch *= 2
                # patch = patch.clip(0, 255)
                ## determine if the patch has enough valid pixel

            patch_name = os.path.join(folder, str(i * y_num + j + original_count) + '_0.tif')

            if require_proj:
                new_top_left_x = top_left_x + x_off_patch * np.abs(w_e_pixel_resolution)
                new_top_left_y = top_left_y - y_off_patch * np.abs(n_s_pixel_resolution)
                dst_transform = (
                    new_top_left_x, img_geotrans[1], img_geotrans[2], new_top_left_y, img_geotrans[4], img_geotrans[5])
                writeTif(swap_patch, patch_name, require_proj, dst_transform, img_proj)
            else:
                writeTif(swap_patch, patch_name)

def writeTif(bands, path, require_proj=False, transform=None, proj=None):
    if bands is None or bands.__len__() == 0:
        return
    else:
        # 认为各波段大小相等，所以以第一波段信息作为保存
        band1 = bands[0]
        # 设置影像保存大小、波段数
        img_width = band1.shape[1]
        img_height = band1.shape[0]
        num_bands = bands.__len__()

        # 设置保存影像的数据类型
        if 'int8' in band1.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in band1.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, img_width, img_height, num_bands, datatype)
        if dataset is not None:
            if require_proj:
                dataset.SetGeoTransform(transform)  # 写入仿射变换参数
                dataset.SetProjection(proj)  # 写入投影
            for i in range(bands.__len__()):
                dataset.GetRasterBand(i + 1).WriteArray(bands[i])
        print("save image success.")

def random_segmentation(img, folder, x_range, y_range, target=False, require_proj=False, continue_count=False):
    if require_proj:
        img_geotrans = img.GetGeoTransform()
        img_proj = img.GetProjection()
        top_left_x = img_geotrans[0]
        w_e_pixel_resolution = img_geotrans[1]
        top_left_y = img_geotrans[3]
        n_s_pixel_resolution = img_geotrans[5]

    x_num = img.RasterXSize // x_range
    y_num = img.RasterYSize // y_range
    x_size, y_size, x_off, y_off = img.RasterXSize, img.RasterYSize, 0, 0
    img_array = img.ReadAsArray(x_off, y_off, x_size, y_size)
    # for i in range(3):
    #     print (np.max(img_array[i]))
    # writeTif(img_array, os.path.join(folder, 'test.tif'))

    ## generate pixels coordinates
    np.random.seed(1015)
    n = 500
    radius = x_range // 2
    x_coor = np.random.choice(range(radius+1, x_size - radius - 1), n)
    y_coor = np.random.choice(range(radius+1, y_size - radius - 1), n)

    for i in range(0, n):
        if not target:
            patch = img_array[:, y_coor[i] - radius:y_coor[i] + radius,
                    x_coor[i] - radius:x_coor[i] + radius]
            ## determine if the patch has enough valid pixel
            valid_ratio = (patch != 0).mean()
            # if valid_ratio < 0.8:
            #     continue
        if target:
            patch = img_array[y_coor[i] - radius:y_coor[i] + radius,
                    x_coor[i] - radius:x_coor[i] + radius][np.newaxis, :, :]
            ## determine if the patch has enough valid pixel
        new_top_left_x = top_left_x + (x_coor[i] - radius) * np.abs(w_e_pixel_resolution)
        new_top_left_y = top_left_y - (y_coor[i] - radius) * np.abs(n_s_pixel_resolution)

        dst_transform = (
            new_top_left_x, img_geotrans[1], img_geotrans[2], new_top_left_y, img_geotrans[4],
            img_geotrans[5])
        patch_name = os.path.join(folder, str(x_coor[i]) + '_' + str(y_coor[i]) + '_0.tif')

        writeTif(patch, patch_name, True, transform=dst_transform, proj=img_proj)

def readImagePath(img_path, x_range, y_range, folder_name='', target=False, require_proj=True, continue_count=True):
    img = gdal.Open(img_path)
    if not target:
        folder = os.path.join(r"F:\DigitalAG\morocco\unet\training\img", folder_name)
    if target:
        folder = os.path.join(r"F:\DigitalAG\morocco\unet\training\target", folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(folder + ' has been created')
    if not continue_count:
        [os.remove(os.path.join(folder, file)) for file in os.listdir(folder)]
    random_segmentation(img, folder, x_range, y_range, target, require_proj, continue_count=False)

def check_bands(dir):
    for item in os.listdir(dir):
        if item.endswith('tif'):
            ds = gdal.Open(os.path.join(dir, item))
            img_geotrans = ds.GetGeoTransform()  # crs transform information
            img_proj = ds.GetProjection()  # projection
            top_left_x = img_geotrans[0]  # x coordinate of upper lefe corner
            w_e_pixel_resolution = img_geotrans[1]  # horizontal resolution
            top_left_y = img_geotrans[3]  # y coordinate of upper lefe corner
            n_s_pixel_resolution = img_geotrans[5]  # vertical resolution
            print (w_e_pixel_resolution, n_s_pixel_resolution)

def generate_shapefile():
    root_dir = r'Z:\Morocco\rescaled'
    save_dir = r'Z:\Morocco\shp'
    for file in os.listdir(root_dir):
        if not file.endswith('tif'):
            continue
        reference_path = os.path.join(root_dir, file)
        shape_path = os.path.join(save_dir, file.replace('tif', 'shp'))
        wkt = buffer_bbox(gdal.Open(reference_path))
        driver = ogr.GetDriverByName("ESRI Shapefile")
        data_source = driver.CreateDataSource(shape_path)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32630)
        layer = data_source.CreateLayer(file.replace('tif', 'shp'), srs, ogr.wkbPolygon)
        # field_name = ogr.FieldDefn("Name", ogr.OFTString)
        # field_name.SetWidth(14)
        # layer.CreateField(field_name)
        # field_name = ogr.FieldDefn("data", ogr.OFTString)
        # field_name.SetWidth(14)
        # layer.CreateField(field_name)
        feature = ogr.Feature(layer.GetLayerDefn())
        # feature.SetField("Name", "test")
        # feature.SetField("data", "1.2")
        polygon = ogr.CreateGeometryFromWkt(wkt)
        feature.SetGeometry(polygon)
        layer.CreateFeature(feature)

        feature = None
        data_source = None

def buffer_bbox(img):
    """
    Buffers the geom by buff and then calculates the bounding box.
    Returns a Geometry of the bounding box
    """
    img_geotrans = img.GetGeoTransform()
    lon1 = img_geotrans[0]
    w_e_pixel_resolution = img_geotrans[1]
    lat1 = img_geotrans[3]
    n_s_pixel_resolution = img_geotrans[5]
    lon2 = lon1 + img.RasterXSize
    lat2 = lat1 - img.RasterYSize
    wkt = """POLYGON((
        %s %s,
        %s %s,
        %s %s,
        %s %s,
        %s %s
    ))""" % (lon1, lat1, lon1, lat2, lon2, lat2, lon2, lat1, lon1, lat1)
    wkt = wkt.replace('\n', '')
    return wkt

if __name__ == '__main__':

    # readImagePath(r"F:\DigitalAG\morocco\unet\data\label\region2_label.tif", 180, 180, '', target=True)
    root_dir = r'Z:\Morocco\second_data\fused'
    # resample(root_dir, root_dir.replace("fused", "resample"))
    # candidate = pd.read_csv(os.path.join(r'G:\My Drive\Digital_Agriculture\Morocco\entire_region', 'candidate_tiles.csv'))['name']
    # candidate = ['18DEC12113125-M1BS-505246646030_01_P004']
    # for c in candidate:
    #     if c+'.tif' in os.listdir(r'Z:\Morocco\resampled'):
    #         continue
    #     save_dir = os.path.join(r"F:\DigitalAG\morocco\ViT\data", c)
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     if not os.path.exists(os.path.join(root_dir, c + '.tif')):
    #         continue
    #     readImagePath(os.path.join(root_dir, c + '.tif'), 90, 90, '')