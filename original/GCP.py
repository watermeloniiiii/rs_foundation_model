import ee
from google.cloud import storage
import datetime
import os

## please use your own credential
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "G:\My Drive\Digital_Agriculture\potato\Earth Engine default project-d372e5cfe621.json"

def gee_upload_image(gee_id, gcs_filename, properties, bands=None, pyramidingPolicy=None):
    # gee_id -- path to asset (/users/username/folder/asset_name)
    # gcs_filename -- path to file on GCS (gs://bucket_name/path_to_blob)

    params = {}
    params['id'] = gee_id

    sources = [{'primaryPath': gcs_filename}]
    params['tilesets'] = [{'sources': sources}]

    if pyramidingPolicy is not None:
        params['pyramidingPolicy'] = pyramidingPolicy  # 'MODE'

    if bands is not None:
        params['bands'] = bands  # [{'id':l['band_name']} for l in md['sublayers']]

    params['properties'] = properties

    newId = ee.data.newTaskId()[0]
    ret = ee.data.startIngestion(newId, params)
    return ret

def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"
    name_list = []
    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)
    for blob in blobs:
        name_list.append(blob.name)
    return name_list

def change_name(bucket_name, blob_name, new_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    new_blob = bucket.rename_blob(blob, new_name)

def delete_asset(id):
    ee.data.deleteAsset(id)



if __name__ == '__main__':
    ## maybe the first time you need to authenticate
    # ee.Authenticate()
    ee.Initialize()
    bucket_name = 'benchmark_zhang2021'
    name_list = list_blobs(bucket_name)
    properties = {}
    # task_list = ee.batch.Task.list()
    # for item in task_list:
    #     ee.batch.Task.cancel(item)
    for item in name_list:
        tile = item.split('/')[-1].split(".")[0]
        year = item.split('/')[1]
        name = f"{tile}_{year}"
        # properties["date"] = f"{year}-01-01"
        properties = {}
        gee_upload_image('users/lin00370/paii/HLJ_crop/2020/' + item.replace('gcp/',"").replace('.tif', ""),
                         f'gs://benchmark_zhang2021/' + item, properties)
        # try:
        #     test = ee.data.getAsset('projects/jin-digitalag-lab/PepsiCo/CCI_soil_moisture/'+band_name)
        #     delete_asset('projects/jin-digitalag-lab/PepsiCo/CCI_soil_moisture/'+band_name)
        # except:
        #     continue
        ## add a word 'ready' at the end of the name to avoid repeating uploading
        # change_name('chenxilin', item, item.replace('.tif', '_ready.tif'))