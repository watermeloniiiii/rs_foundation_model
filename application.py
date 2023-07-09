import os
import torch
import numpy as np
import pandas as pd
from skimage import io
from osgeo import gdal
import torch.nn as nn
import torch.utils as utils
from utilities import writeTif
import matplotlib.pyplot as plt
from model_shallow import shallowNet_224
from model_pretrain import pretrainedViT
from sklearn.cluster import KMeans
from torchvision import models
from dataio import UNetDataset, shallowDataset
from torch.autograd import Variable



def reconstruct(test_dir, model):
    net = shallowNet()
    net.load_state_dict(torch.load(model))
    test_data = BuildingDataset(dir=test_dir, transform=None, target=False)
    test_loader = utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=8)
    for l, sample in enumerate(test_loader):
        image = Variable(sample['image'], requires_grad=False).type(torch.FloatTensor)
        if cuda:
            image = image.cuda()
        vector, pred = net(image)
        result_folder = os.path.join(root_dir, r"visualization")
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        plt.figure(1, figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(image.cpu().data.numpy().squeeze().transpose())
        plt.subplot(1, 2, 2)
        plt.imshow(pred.cpu().data.numpy().squeeze().transpose())
        plt.show()

def clustering(test_dir, model):
    import shutil
    if os.path.exists(os.path.join(root_dir, 'clustering')):
        shutil.rmtree(os.path.join(root_dir, 'clustering'))
    os.makedirs(os.path.join(root_dir, 'clustering'))
    net = torch.load(model)
    clusterCenter = net.clusterCenter.cpu().data.numpy().squeeze()
    clusterNumber = clusterCenter.shape[0]
    for i in range(clusterNumber):
        os.makedirs(os.path.join(root_dir, 'clustering', str(i)))
    kmeans = KMeans(n_clusters=4, init=clusterCenter)
    test_data = BuildingDataset(dir=test_dir, transform=None, target=None)
    test_loader = utils.data.DataLoader(test_data, batch_size=16, shuffle=True, num_workers=8)
    for l, sample in enumerate(test_loader):
        image = Variable(sample['image'], requires_grad=False).type(torch.FloatTensor)
        if cuda:
            image = image.cuda()
        vector, pred = net(image)
        label = kmeans.fit_predict(vector.cpu().data.numpy())
        for j in range(len(label)):
            io.imsave(os.path.join(root_dir, 'clustering', str(label[j]), sample['name'][j]),
                      image[j].cpu().data.numpy().squeeze().transpose()[:,:,:3])
            # shutil.copyfile(os.path.join(test_dir, sample['name'][j]),
            #                     os.path.join(root_dir, 'clustering', str(label[j]), sample['name'][j]))
            # plt.figure(1, figsize=(5, 5))
            # plt.imshow(image[j].cpu().data.numpy().squeeze().transpose())
            # plt.savefig(os.path.join(root_dir, 'clustering', str(label[j]), sample['name'][j]))

def assessment_vit(test_dir, out_dir, model, mode='sub'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_dir.replace('pred', 'prob')):
        os.makedirs(out_dir.replace('pred', 'prob'))
    # net = pretrainedViT().cuda()
    net = torch.load(model)
    test_data = shallowDataset(img_dir=test_dir)
    test_loader = utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=8)
    pred_arr = np.zeros([224, 224])
    prob_arr = np.zeros([224, 224])
    label_list = []
    pred_list = []
    try:
        for l, sample in enumerate(test_loader):
            image = Variable(sample['image'], requires_grad=False).type(torch.FloatTensor)
            if cuda:
                image = image.cuda()
            m = nn.Sigmoid()
            _, patches, last_layer = net(image)[:3]

            if mode == 'all':
                # prob = m(torch.mean(patches, axis=1).flatten())
                prob = torch.mean(m(last_layer), dim=1)
                pred = torch.gt(prob, 0.5).type(torch.cuda.FloatTensor).cpu().detach().data.numpy()
                prob = prob.cpu().detach().data.numpy()
                for i in range(49):
                    row_min = (i // 7) * 32
                    row_max = ((i // 7) + 1) * 32
                    col_min = (i % 7) * 32
                    col_max = ((i % 7) + 1) * 32
                    pred_arr[row_min:row_max, col_min:col_max] = pred
                    prob_arr[row_min:row_max, col_min:col_max] = prob
            elif mode == 'sub':
                prob = m(last_layer).flatten()
                # pred = torch.gt(prob, 0.5).type(torch.cuda.FloatTensor).cpu().detach().data.numpy()
                prob = prob.cpu().data.numpy()
                for i in range(49):
                    row_min = (i // 7) * 32
                    row_max = ((i // 7) + 1) * 32
                    col_min = (i % 7) * 32
                    col_max = ((i % 7) + 1) * 32
                    # pred_arr[row_min:row_max, col_min:col_max] = pred[i]
                    prob_arr[row_min:row_max, col_min:col_max] = prob[i]
            # io.imsave(os.path.join(out_dir, sample['name'][0]+'_0.tif'), pred_arr)
            io.imsave(os.path.join(out_dir.replace('pred', 'prob'), sample['name'][0]+'_0.tif'), prob_arr)
    except:
        pass

def assessment_unet(test_dir, out_dir, model):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_dir.replace('pred', 'prob')):
        os.makedirs(out_dir.replace('pred', 'prob'))
    net = torch.load(model)
    test_data = UNetDataset(dir=test_dir, transform=None, target=False)
    test_loader = utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=8)
    for l, sample in enumerate(test_loader):
        image = Variable(sample['image'], requires_grad=False).type(torch.FloatTensor)
        if cuda:
            image = image.cuda()
        prob = torch.sigmoid(net(image))
        # pred = torch.ge(torch.sigmoid(net(image)), 0.5).type(torch.cuda.FloatTensor).squeeze()
        # pred = pred.cpu().detach().data.numpy()
        prob = prob.cpu().detach().data.numpy()
        # io.imsave(os.path.join(out_dir, sample['name'][0]), pred)
        io.imsave(os.path.join(out_dir.replace('pred', 'prob'), sample['name'][0]), prob)

def assessment_shallow(test_dir, out_dir, model, size=128):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # net = shallowNet_224().cuda()
    net = torch.load(model)
    # net = models.resnet18(pretrained=True)
    for param in net.parameters():
        test1 = param
    # model = models.densenet121(pretrained=True)
    # num_ftrs = net.fc.in_features
    # net.fc = nn.Linear(int(num_ftrs), 1)
    # net.load_state_dict(torch.load(model))
    for param in net.parameters():
        test2 = param
    net = net.cuda()
    test_data = shallowDataset(img_dir=test_dir)
    test_loader = utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=8)
    pred_list = []
    for l, sample in enumerate(test_loader):
        image = Variable(sample['image'], requires_grad=False).type(torch.FloatTensor)
        if cuda:
            image = image.cuda()
        m = nn.Sigmoid()
        out = net(image)
        prob = m(out)
        prob = prob.cpu().detach().data.numpy()
        prob_arr = np.ones([size, size]) * prob
        io.imsave(os.path.join(out_dir, sample['name'][0] + '_0.tif'), prob_arr)


def college_vit(ref_dir, pred_dir, prob_dir, pred_out, prob_out, remove_folder=False, size=128):
    if not os.path.exists(prob_dir.replace("prob", 'mosaic')):
        os.makedirs(prob_dir.replace("prob", 'mosaic'))
    ref_img = gdal.Open(ref_dir)
    img_geotrans = ref_img.GetGeoTransform()
    img_proj = ref_img.GetProjection()

    x_num = ref_img.RasterXSize // size
    y_num = ref_img.RasterYSize // size
    x_range = size
    y_range = size

    pred_collaged = np.zeros([1, ref_img.RasterYSize, ref_img.RasterXSize])
    prob_collaged = np.zeros([1, ref_img.RasterYSize, ref_img.RasterXSize])

    for i in range(0, x_num):
        for j in range(0, y_num):
            try:
                # pred = io.imread(os.path.join(pred_dir, str(str(i * y_num + j) + '_0.tif')))
                prob = io.imread(os.path.join(prob_dir, str(str(i * y_num + j) + '_0.tif')))
                print(i * y_num + j)
            except:
                # pred_collaged[:, j * y_range: (j + 1) * y_range, i * x_range: (i + 1) * x_range] = 0
                prob_collaged[:, j * y_range: (j + 1) * y_range, i * x_range: (i + 1) * x_range] = 0
            else:
                # pred_collaged[:, j * y_range: (j + 1) * y_range, i * x_range: (i + 1) * x_range] = pred
                prob_collaged[:, j * y_range: (j + 1) * y_range, i * x_range: (i + 1) * x_range] = prob

    # writeTif(pred_collaged, pred_out, require_proj=True, transform=img_geotrans, proj=img_proj)
    writeTif(prob_collaged, prob_out, require_proj=True, transform=img_geotrans, proj=img_proj)
    if remove_folder:
        import shutil
        shutil.rmtree(prob_dir)

def college_unet(ref_dir, pred_dir, pred_out, remove_folder=False):
    if not os.path.exists(pred_dir.replace("prob", 'mosaic')):
        os.makedirs(pred_dir.replace("prob", 'mosaic'))
    ref_img = gdal.Open(ref_dir)
    img_geotrans = ref_img.GetGeoTransform()
    img_proj = ref_img.GetProjection()

    x_num = ref_img.RasterXSize // 224
    y_num = ref_img.RasterYSize // 224
    x_range = 224
    y_range = 224

    pred_collaged = np.zeros([1, ref_img.RasterYSize, ref_img.RasterXSize])

    for i in range(0, x_num):
        for j in range(0, y_num):
            try:
                pred = io.imread(os.path.join(pred_dir, str(str(i * y_num + j) + '_0.tif')))
                # prob = io.imread(os.path.join(prob_dir, str(str(i * y_num + j) + '_0.tif')))
            except:
                pred_collaged[:, j * y_range: (j + 1) * y_range, i * x_range: (i + 1) * x_range] = 0
                # prob_collaged[:, j * y_range: (j + 1) * y_range, i * x_range: (i + 1) * x_range] = 0
            else:
                pred_collaged[:, j * y_range: (j + 1) * y_range, i * x_range: (i + 1) * x_range] = pred
                # prob_collaged[:, j * y_range: (j + 1) * y_range, i * x_range: (i + 1) * x_range] = prob

    writeTif(pred_collaged, pred_out, require_proj=True, transform=img_geotrans, proj=img_proj)
    if remove_folder:
        import shutil
        shutil.rmtree(pred_dir)

def application():
    for model_folder in os.listdir(model_dir):
        if model_folder.split('_')[-1] in ['notselected']:
            continue
        model_folder_path = os.path.join(model_dir, model_folder)
        if not os.path.isdir(model_folder_path):
            continue
        model = os.path.join(model_folder_path, str(model_folder) + '.pkl')
        clustering(test_dir, model)

def scale_percentile_n(matrix):
    matrix = matrix.transpose(2, 0, 1).astype(np.float)
    d, w, h = matrix.shape
    for i in range(d):
        mins = np.percentile(matrix[i][matrix[i] != 0], 1)
        maxs = np.percentile(matrix[i], 99)
        matrix[i] = matrix[i].clip(mins, maxs)
        matrix[i] = ((matrix[i] - mins) / (maxs - mins))
    return matrix

def clean_noise(in_dir, thres=0.5, img_range=32):
    img = gdal.Open(in_dir)
    img_geotrans = img.GetGeoTransform()
    img_proj = img.GetProjection()
    rows = img.RasterYSize
    cols = img.RasterXSize
    img_prob = img.ReadAsArray(0, 0, cols, rows)
    img_pred = (img_prob > thres).astype(np.int8)
    assert len(img_prob.shape) == 2, 'please make sure the input is 2D binary mask'
    col_num = cols // img_range
    row_num = rows // img_range
    for row in range(1, row_num - 1):
        for col in range(1, col_num - 1):
            candi_prob = img_prob[row * img_range: (row + 1) * img_range, col * img_range: (col + 1) * img_range].mean()
            candi_pred = img_pred[row * img_range: (row + 1) * img_range, col * img_range: (col + 1) * img_range].mean()
            top_prob, bottom_prob, left_prob, right_prob = img_prob[(row - 1) * img_range: row * img_range, col * img_range: (col + 1) * img_range].mean(), \
                                                           img_prob[(row + 1) * img_range: (row + 2) * img_range, col * img_range: (col + 1) * img_range].mean(), \
                                                           img_prob[row * img_range: (row + 1) * img_range, (col - 1) * img_range: col * img_range].mean(), \
                                                           img_prob[row * img_range: (row + 1) * img_range, (col + 1) * img_range: (col + 2) * img_range].mean()
            top_pred, bottom_pred, left_pred, right_pred = img_pred[(row - 1) * img_range: row * img_range,
                                                           col * img_range: (col + 1) * img_range].mean(), \
                                                           img_pred[(row + 1) * img_range: (row + 2) * img_range,
                                                           col * img_range: (col + 1) * img_range].mean(), \
                                                           img_pred[row * img_range: (row + 1) * img_range,
                                                           (col - 1) * img_range: col * img_range].mean(), \
                                                           img_pred[row * img_range: (row + 1) * img_range,
                                                           (col + 1) * img_range: (col + 2) * img_range].mean()
            flag = int(top_pred == candi_pred) + \
                   int(bottom_pred == candi_pred) + \
                   int(left_pred == candi_pred) + \
                   int(right_pred == candi_pred)
            if flag == 0:
                img_prob[row * img_range: (row + 1) * img_range,
                col * img_range: (col + 1) * img_range] = (top_prob+bottom_prob+left_prob+right_prob)/4
    writeTif(img_prob[np.newaxis,:,:], in_dir.replace('_prob', '_prob_post'),
                     require_proj=True, transform=img_geotrans, proj=img_proj)


if __name__ == '__main__':
    cuda = True
    type = "vit"
    all = True
    root_dir = r"F:\DigitalAG\morocco\unet"
    model_dir = r"F:\DigitalAG\morocco\unet\model"
    model = "vit_2.pkl"
    if not all:
        ###################
        ##### run inference for single imagery
        ###################
        dataset = r'baseline\32\testing\region5'
        test_dir = fr"F:\DigitalAG\morocco\unet\{dataset}\img"
        if type == "vit":
            for mode in ['sub']:
                assessment_vit(test_dir,
                               out_dir=fr"F:\DigitalAG\morocco\unet\{dataset}\{type}\pred",
                               model=fr"F:\DigitalAG\morocco\unet\model\selected\{model}")
                college_vit(ref_dir=fr"Z:\Morocco\second_data\georeferenced\\19FEB13111500-M1BS-505246646070_01_P002_gr.tif",
                             pred_dir=fr"F:\DigitalAG\morocco\unet\{dataset}\{type}\pred",
                             prob_dir=fr"F:\DigitalAG\morocco\unet\{dataset}\{type}\prob",
                             pred_out=fr"F:\DigitalAG\morocco\unet\{dataset}\{type}\mosaic\{model.split('.')[0]}_{mode}.tif",
                             prob_out=fr"F:\DigitalAG\morocco\unet\{dataset}\{type}\mosaic\{model.split('.')[0]}_{mode}_prob.tif",
                             remove_folder=True,
                             size=224
                             )
                clean_noise(fr"F:\DigitalAG\morocco\unet\{dataset}\vit\mosaic\{model.split('.')[0]}_{mode}_prob.tif")
        if type == "unet":
            assessment_unet(test_dir,
                            out_dir=fr"F:\DigitalAG\morocco\unet\{dataset}\unet\pred",
                            model=fr"F:\DigitalAG\morocco\unet\model\{model}")
            college_unet(ref_dir=fr"F:\DigitalAG\morocco\unet\data\img\region7_gr.tif",
                         pred_dir=fr"F:\DigitalAG\morocco\unet\{dataset}\{type}\prob",
                         pred_out=fr"F:\DigitalAG\morocco\unet\{dataset}\{type}\mosaic\{model.split('.')[0]}_prob.tif")
        if type == "shallow":
            assessment_shallow(test_dir,
                            out_dir=fr"F:\DigitalAG\morocco\unet\{dataset}\{type}\prob",
                            model=fr"F:\DigitalAG\morocco\unet\model\selected\{model}",
                            size=32)
            college_vit(ref_dir=fr"Z:\Morocco\georeference_task\qualified\\19FEB13111500-M1BS-505246646070_01_P002_gr.tif",
                        pred_dir=None,
                        prob_dir=fr"F:\DigitalAG\morocco\unet\{dataset}\{type}\prob",
                        pred_out=None,
                        prob_out=fr"F:\DigitalAG\morocco\unet\{dataset}\{type}\mosaic\{model.split('.')[0]}_prob.tif",
                        remove_folder=True,
                        size=32
                        )
    if all:
        ##################
        #### run inference for all imagery
        ##################
        for dataset in os.listdir(r"Z:\Morocco\patch"):
            test_dir = os.path.join(r"Z:\Morocco\patch", dataset)
            if os.path.exists(fr"Z:\Morocco\results\{type}\gcp\{dataset}.tif"):
                continue
            if type == 'vit':
                    print (fr"now processing tile {dataset}")
                    assessment_vit(test_dir,
                                   out_dir=fr"Z:\Morocco\results\vit\{dataset}\pred",
                                   model=fr"F:\DigitalAG\morocco\unet\model\selected\{model}")
                    college_vit(ref_dir=fr"Z:\Morocco\second_data/georeferenced\{dataset}.tif",
                                pred_dir=fr"Z:\Morocco\results\vit\{dataset}\\pred",
                                prob_dir=fr"Z:\Morocco\results\vit\{dataset}\prob",
                                pred_out=fr"Z:\Morocco\results\vit\{dataset}\mosaic\{model.split('.')[0]}.tif",
                                prob_out=fr"Z:\Morocco\results\vit\{dataset}\mosaic\{model.split('.')[0]}_prob.tif",
                                remove_folder=True,
                                size=224
                                )
                    clean_noise(fr"Z:\Morocco\results\vit\{dataset}\mosaic\{model.split('.')[0]}_prob.tif")
            if type == "unet":
                assessment_unet(test_dir,
                               out_dir=fr"Z:\Morocco\results\{type}\{dataset}\pred",
                               model=fr"F:\DigitalAG\morocco\unet\model\selected\{model}")
                college_unet(ref_dir=fr"Z:\Morocco\second_data\georeferenced\{dataset}.tif",
                            pred_dir=fr"Z:\Morocco\results\{type}\{dataset}\\prob",
                            pred_out=fr"Z:\Morocco\results\{type}\{dataset}\mosaic\{model.split('.')[0]}.tif",
                            remove_folder=True
                            )


    # application()
    # generate_visualization2(r"F:\DigitalAG\morocco\unet\training\img")