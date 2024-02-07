# See PyCharm help at https://www.jetbrains.com/help/pycharm/
from osgeo import gdal
import numpy as np
import pandas as pd
import os,cv2
import math
from sklearn.linear_model import LinearRegression

def fusion(
        in_dir: str,
        out_dir: str,
        ms_tag: str,
        pan_tag: str):
        """
        Conduct image fusion between coarse multi-spectral bands and fine pan-chromatic band

        parameters
        in_dir, str
        out_dir, str
        ms_tag, str
                the unique tag to denote the multi-spectral imagery, e.g., "M2AS"
        pan_tag, str
                the unique tag to denote the pan-chromatic imagery, e.g., "P2AS"
        """

        dir_list = os.listdir(in_dir)
        outlist = os.listdir(out_dir)
        for file in dir_list:
                if not file.endswith('tif') or file in outlist:
                        print ('existed!')
                        continue
                if ms_tag in file:
                        ms_dir = os.path.join(in_dir, file)
                        pan_dir = ms_dir.replace(ms_tag, pan_tag)
                        if not os.path.exists(pan_dir):
                                continue
                else:
                        continue

                coarse_ds = gdal.Open(ms_dir)
                coarse_gt = coarse_ds.GetGeoTransform()
                nsc, nlc, nbc = coarse_ds.RasterXSize, coarse_ds.RasterYSize, coarse_ds.RasterCount
                coarseImg = coarse_ds.ReadAsArray(0, 0, nsc, nlc).transpose()
                coarseImg[coarseImg == 65535] = 0
                leftCx = coarse_gt[0]
                leftCy = coarse_gt[3]
                ps0 = coarse_gt[1]

                if nbc == 4:
                        bandInd = [0,1,2]
                else:
                        bandInd = [1,2,4] #bgrn
                coarseImg = coarseImg[:, :, bandInd]
                nbc = 3

                fine_ds = gdal.Open(pan_dir)
                fine_gt = fine_ds.GetGeoTransform()
                nsf, nlf, nbf = fine_ds.RasterXSize, fine_ds.RasterYSize, fine_ds.RasterCount
                fineImg = fine_ds.ReadAsArray(0, 0, nsf, nlf).transpose()
                fineImg[fineImg == 65535] = 0
                leftFx = fine_gt[0]
                leftFy = fine_gt[3]
                psF0 = fine_gt[1]

                ratio = round(ps0 / psF0)

                #block size
                step = 1000.0
                ss = math.ceil(nsc/step)
                ls = math.ceil(nlc / step)
                inF = 0.0

                Prediction = np.zeros([nsc*ratio, nlc*ratio, nbc], dtype=np.uint16)
                driver = gdal.GetDriverByName("GTiff")  # OutPath+r"/"+file
                new_X = nsc * ratio
                new_Y = nlc * ratio
                outdata = driver.Create(os.path.join(OutPath, file), new_X, new_Y, nbc, gdal.GDT_UInt16, options=['COMPRESS=LZW', 'BIGTIFF=YES'])  # cols, rows
                proj = coarse_ds.GetProjection()
                geoTransform = coarse_ds.GetGeoTransform()
                geoTransform = np.array(geoTransform)
                geoTransform[1] = ps0 / ratio
                geoTransform[5] = -1 * ps0 / ratio
                outdata.SetGeoTransform(geoTransform)
                outdata.SetProjection(proj)


                for i in range(ss):
                        for j in range(ls):
                                step = math.ceil(step)
                                nsMax = min([nsc,(i+1)*step])
                                nlMax = min([nlc,(j+1)*step])
                                coarseBlock = coarseImg[i*step:nsMax,j*step:nlMax, :]

                                mask = (coarseBlock[:, :, 0]>0) & (coarseBlock[:, :, 0]<1e4)
                                mask = mask.astype(int)
                                pos2 = np.where(mask)
                                pos2 = np.array(pos2)
                                if pos2[0,:].size == 0:
                                        continue
                                coarseT = np.zeros([pos2[0,:].size,nbc])
                                fineUp = np.zeros([pos2[0,:].size,1])
                                FineBlock = np.zeros([(nsMax-i*step)*ratio, (nlMax-j*step)*ratio])
                                for k in range(pos2[0,:].size):
                                        ai = max([pos2[0, k]-ratio+1, 0])
                                        aj = min([pos2[0, k]+ratio-1, nsMax-i*step-1])
                                        bi = max([pos2[1, k]-ratio+1, 0])
                                        bj = min([pos2[1, k]+ratio-1, nlMax-j*step-1])
                                        rcWin = coarseBlock[ai:aj+1, bi:bj+1,0]
                                        indexRC = np.where((rcWin<=0)|(rcWin>1e4))
                                        indexRC = np.array(indexRC)
                                        if indexRC[0,:].size>0:
                                                mask[pos2[0, k], pos2[1, k]] = 0
                                                continue
                                        #preparation for fusion
                                        xT = leftCx+(pos2[0, k]+i*step)*ps0 #-0.5*ps0
                                        yT = leftCy-(pos2[1, k]+j*step)*ps0 #+0.5*ps0
                                        if ((xT<leftFx) | (xT+ps0>leftFx+psF0*nsf)) | ((yT>leftFy) | (yT+0<leftFy-psF0*nlf)):
                                                fineUp[k] = inF
                                                continue
                                        xf = round(max([(xT-leftFx)/psF0, 0]))
                                        yf = round(max([(leftFy-yT)/psF0, 0]))
                                        fineT = fineImg[xf:xf+ratio, yf:yf+ratio]
                                        indexT = np.where((fineT<=0) | (fineT>1e4))
                                        indexT = np.array(indexT)
                                        if indexT[0,:].size>0:
                                                fineUp[k] = inF
                                                continue
                                        fineUp[k] = np.mean(fineT)
                                        coarseT[k, :] = coarseBlock[pos2[0,k],pos2[1,k],:]
                                        FineBlock[pos2[0, k]*ratio:(pos2[0,k]+1)*ratio, pos2[1,k]*ratio:(pos2[1,k]+1)*ratio] = fineT

                                mask = cv2.resize(mask, (coarseBlock[:,:,0].shape[1]*ratio,coarseBlock[:,:,0].shape[0]*ratio), interpolation=cv2.INTER_NEAREST) #open cv: (width, height): (columns, rows)
                                coarseBlockRe = cv2.resize(coarseBlock,(coarseBlock[:,:,0].shape[1]*ratio, coarseBlock[:,:,0].shape[0]*ratio), interpolation=cv2.INTER_NEAREST)

                                indexN = np.where(fineUp != inF)
                                indexN = np.array(indexN)
                                indexN = indexN[0,:]
                                for m in range(nbc):
                                        try:
                                                reg = LinearRegression().fit(fineUp[indexN,:], coarseT[indexN, m])
                                        except:
                                                pass
                                        Residual = np.zeros([nsMax - i * step, nlMax - j * step])
                                        Residual[pos2[0],pos2[1]] = coarseT[:, m]-(reg.coef_*fineUp[:,0]+reg.intercept_)
                                        Residual = cv2.resize(Residual, (int((nlMax-j*step)/2), int((nsMax-i*step)/2)), interpolation=cv2.INTER_AREA) #scale up the residuals to reduce the impacts of misregistration
                                        Residual = cv2.resize(Residual, (coarseBlock[:,:,0].shape[1]*ratio,coarseBlock[:,:,0].shape[0]*ratio),interpolation=cv2.INTER_CUBIC)
                                        tmpA = np.round(reg.coef_*FineBlock+reg.intercept_+Residual) * mask
                                        #remove negative values in the results
                                        indexZ1 = np.where((tmpA<0)&(FineBlock>0))
                                        if indexZ1[0].size != 0:
                                                tmpA[indexZ1[0],indexZ1[1]] = coarseBlockRe[indexZ1[0],indexZ1[1],m]
                                        indexZ = np.where(tmpA < 0)
                                        if indexZ[0].size != 0:
                                                tmpA[indexZ] = 0
                                        Prediction[i*step*ratio:nsMax*ratio, j*step*ratio:nlMax*ratio, m] = (tmpA).astype(np.uint16)
                                pos2 = []
                                mask = []
                                coarseBlockRe = []
                                indexN = []
                                Residual = []

                # Prediction = scale_percentile_n(np.transpose(Prediction, (1, 0, 2)))
                # Prediction = (Prediction/256).astype(np.uint8)
                Prediction = Prediction.transpose(1,0,2)
                # np.save(os.path.join(OutPath, 'np.npy'), Prediction)
                for i in range(nbc):
                        outdata.GetRasterBand(i+1).WriteArray(Prediction[:,:,i])
                outdata.FlushCache()  ##saves to disk!!
                outdata = None
                drive = None

def scale_percentile_n(matrix):
    # matrix = matrix.transpose(1, 2, 0)
    w, h, d = matrix.shape
    result = []
    for i in range(d):
        sub_matrix = np.reshape(matrix[:,:,i], [w * h]).astype(np.float16)  # 百分位的值是在一个列表中排序 再取
        mins = np.percentile(sub_matrix, 0, axis=0)
        maxs = np.percentile(sub_matrix, 100, axis=0)
        sub_matrix = ((sub_matrix - mins) / (maxs - mins))
        sub_matrix = np.reshape(sub_matrix, [w, h]) * 255
        result.append(sub_matrix.astype(np.uint8))

    return np.array(result).transpose(1,2,0)

if __name__ == '__main__':
        fusion()