import os
import numpy as np
import pandas as pd
from osgeo import gdal

def confusion_matrix_export(cm, out_dir, label=None):
    """ export confusion matrix as excel file

        Parameters
        ----------
        cm: ndarray
            confusion matrix
        out_dir: str
            the output directory
        label: List[str]
            labels used for the rows and columns, if None, will automatically be numbers from 0
    """
    if not label:
        label = np.arange(1, cm.shape[0] + 1)
    index = pd.MultiIndex.from_product([['Ground truth'], label])
    column = pd.MultiIndex.from_product([['Predicted label'], label])
    df = pd.DataFrame(data=cm, columns=column, index=index)
    df['User accuracy'] = 0
    df = pd.concat([df, pd.DataFrame(data=np.zeros([1, len(label)]), \
                           columns=pd.MultiIndex.from_product([['Predicted label'], label]), \
                           index=[['Producer accuracy'], ['']])], axis=0)
    value_arr = df.values
    for i in range(0, len(label)):
        tp = df['Predicted label'][label[i]].loc['Ground truth'][label[i]]
        fp_plus_tp = sum(df['Predicted label'].loc['Ground truth'].loc[label[i]])
        tn_plus_tp = sum(df.loc['Ground truth']['Predicted label'][label[i]])
        value_arr[i, -1] = tp / fp_plus_tp
        value_arr[-1, i] = tp / tn_plus_tp
        # df['User accuracy'].loc['Ground truth'].loc[label[i]] = tp / fp_plus_tp
        # df.loc['Producer accuracy'].loc['']['Predicted label'][label[i]] = tp / tn_plus_tp
    OA = np.diagonal(cm).sum()/cm.sum()
    value_arr[-1, -1] = OA
    df = pd.DataFrame(data=value_arr)
    # df.loc['Producer accuracy'].loc['']['User accuracy'] = OA
    df.to_csv(out_dir)

def writeTif(bands, path, require_proj=False, transform=None, proj=None):
    if bands is None or bands.__len__() == 0:
        return
    else:
        band1 = bands[0]
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
        dataset = driver.Create(path, img_width, img_height, num_bands, datatype, options=['COMPRESS=LZW', 'BIGTIFF=YES', 'CHECK_DISK_FREE_SPACE=FALSE'])
        if dataset is not None:
            if require_proj:
                dataset.SetGeoTransform(transform)  # 写入仿射变换参数
                dataset.SetProjection(proj)  # 写入投影
            for i in range(bands.__len__()):
                dataset.GetRasterBand(i + 1).WriteArray(bands[i])
        print("save image success.")