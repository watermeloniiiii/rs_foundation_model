# See PyCharm help at https://www.jetbrains.com/help/pycharm/
from osgeo import gdal
import numpy as np
import pandas as pd
import os, cv2
import math
from sklearn.linear_model import LinearRegression

def fusion():
        #input path
        InPath = r"X:\Morocco\second_data\reproject"
        #output path
        OutPath = r"X:\Morocco\second_data\fused"

        dir_list = os.listdir(InPath)
        outlist = os.listdir(OutPath)

        for file in dir_list:
                if not file.endswith('tif') or file in outlist:
                        continue
                # if not file in ['21MAR31105355-M1BS-505246644070_01_P008.tif']:
                #         continue
                if "M1BS" in file:
                        # fine resolution
                        CP = os.path.join(InPath, file)
                        # coarse resolution
                        FP = CP.replace("M1BS", 'P1BS')
                else:
                        continue

                # coarse_band fineHis= coarse_ds.GetRasterBand(1)
                coarse_ds = gdal.Open(CP)
                coarse_gt = coarse_ds.GetGeoTransform()
                nsc, nlc, nbc = coarse_ds.RasterXSize, coarse_ds.RasterYSize, coarse_ds.RasterCount
                coarseImg = coarse_ds.ReadAsArray(0, 0, nsc, nlc).transpose()
                leftCx = coarse_gt[0]
                leftCy = coarse_gt[3]
                ps0 = coarse_gt[1]

                if nbc == 4:
                        bandInd = [0,1,2]
                else:
                        bandInd = [1,2,4]
                coarseImg = coarseImg[:, :, bandInd]
                nbc = 3

                fine_ds = gdal.Open(FP)
                fine_gt = fine_ds.GetGeoTransform()
                nsf, nlf, nbf = fine_ds.RasterXSize, fine_ds.RasterYSize, fine_ds.RasterCount
                fineImg = fine_ds.ReadAsArray(0, 0, nsf, nlf).transpose()
                leftFx = fine_gt[0]
                leftFy = fine_gt[3]
                psF0 = fine_gt[1]
                projF = fine_ds.GetProjection()

                ratio = round(ps0 / psF0)

                #block size
                step = 1000.0
                ss = math.ceil(nsc / step)
                ls = math.ceil(nlc / step)
                inF = 0.0

                Prediction = np.zeros([nsc * ratio, nlc * ratio, nbc]).astype(np.uint8)

                driver = gdal.GetDriverByName("GTiff")
                outdata = driver.Create(OutPath + r"/"+file, nsc*ratio, nlc*ratio, nbc, gdal.GDT_Byte) #cols, rows
                proj = coarse_ds.GetProjection()
                geoTransform = coarse_ds.GetGeoTransform()
                geoTransform = np.array(geoTransform)
                geoTransform[1] = ps0/ratio
                geoTransform[5] = -1*ps0/ratio
                outdata.SetGeoTransform(geoTransform)
                outdata.SetProjection(proj)

                for i in range(ss):
                        for j in range(ls):
                                step = math.ceil(step)
                                nsMax = min([nsc, (i+1) * step])
                                nlMax = min([nlc, (j+1) * step])
                                coarseBlock = coarseImg[i*step:nsMax,j*step:nlMax, :]
                                FineBlock = np.zeros([(nsMax-i*step)*ratio, (nlMax-j*step)*ratio])
                                ct = coarseBlock[:,:,0]
                                ## find valid pixels (pixel value between 1 and 10000)
                                pos2 = np.where((ct > 0) & (ct < 1e4))
                                pos2 = np.array(pos2)
                                if pos2[0, :].size == 0:
                                        continue
                                coarseT = np.zeros([pos2[0,:].size, nbc])
                                fineUp = np.zeros([pos2[0,:].size, 1])
                                mask = (ct > 0) & (ct < 1e4)
                                mask = mask.astype(int)
                                for k in range(pos2[0,:].size):
                                        ai = max([pos2[0, k] - ratio + 1, 0])
                                        aj = min([pos2[0, k] + ratio - 1, nsMax - i * step - 1])
                                        bi = max([pos2[1, k] - ratio + 1, 0])
                                        bj = min([pos2[1, k] + ratio - 1, nlMax - j * step - 1])
                                        rcWin = ct[ai:aj+1, bi:bj+1]
                                        indexRC = np.where((rcWin <= 0) | (rcWin > 1e4))
                                        indexRC = np.array(indexRC)
                                        if indexRC[0,:].size>0:
                                                mask[pos2[0, k], pos2[1, k]] = 0
                                                continue
                                        #preparation for fusion
                                        xT = leftCx+(pos2[0, k]+i*step)*ps0 - 0.5*ps0
                                        yT = leftCy-(pos2[1, k]+j*step)*ps0 + 0.5*ps0
                                        if ((xT < leftFx) | (xT+ps0 > leftFx+psF0 * nsf)) | ((yT > leftFy) | (yT+0 < leftFy-psF0 * nlf)):
                                                fineUp[k] = inF
                                                continue

                                        xf = round(max([(xT - leftFx) / psF0, 0]))
                                        yf = round(max([(leftFy - yT) / psF0, 0]))
                                        fineT = fineImg[xf:xf+ratio, yf:yf+ratio]
                                        indexT = np.where((fineT <= 0) | (fineT > 1e4))
                                        indexT = np.array(indexT)
                                        if indexT[0,:].size>0:
                                                fineUp[k] = inF
                                                continue
                                        fineUp[k] = np.mean(fineT)
                                        coarseT[k, :] = coarseBlock[pos2[0, k], pos2[1, k], :]
                                        FineBlock[pos2[0, k]*ratio:(pos2[0,k]+1)*ratio, pos2[1,k]*ratio:(pos2[1,k]+1)*ratio] = fineT

                                mask = cv2.resize(mask, (ct.shape[1]*ratio,ct.shape[0]*ratio), interpolation=cv2.INTER_NEAREST) #open cv: (width, height): (columns, rows)
                                coarseBlockRe = cv2.resize(coarseBlock, (ct.shape[1]*ratio, ct.shape[0]*ratio), interpolation=cv2.INTER_NEAREST)

                                indexN = np.where(fineUp != inF)
                                indexN = np.array(indexN)
                                indexN = indexN[0,:]
                                RCCoarse = np.zeros([nsMax-i*step, nlMax-j*step])
                                for m in range(nbc):
                                        reg = LinearRegression().fit(fineUp[indexN,:], coarseT[indexN, m])
                                        RCCoarse[pos2[0],pos2[1]] = coarseT[:, m]-(reg.coef_*fineUp[:,0]+reg.intercept_)
                                        RCCoarse1 = cv2.resize(RCCoarse, (int((nlMax-j*step)/2), int((nsMax-i*step)/2)), interpolation=cv2.INTER_AREA) #scale up the residuals to reduce the impacts of misregistration
                                        RCFine = cv2.resize(RCCoarse1, (ct.shape[1]*ratio,ct.shape[0]*ratio),interpolation=cv2.INTER_CUBIC)
                                        tmpA = np.round(reg.coef_*FineBlock+reg.intercept_+RCFine) * mask
                                        #remove negative values in the results
                                        indexZ1 = np.where((tmpA < 0) & (FineBlock>0))
                                        if indexZ1[0].size != 0:
                                                tmpA[indexZ1[0],indexZ1[1]] = coarseBlockRe[indexZ1[0],indexZ1[1],m]

                                        indexZ = np.where(tmpA < 0)
                                        if indexZ[0].size != 0:
                                                tmpA[indexZ] = 0
                                        Prediction[i*step*ratio:nsMax*ratio, j*step*ratio:nlMax*ratio, m] = tmpA

                Prediction = np.transpose(Prediction, (1, 0, 2))
                for i in range(nbc):
                        outdata.GetRasterBand(i+1).WriteArray(Prediction[:,:,i])

                outdata.FlushCache()  ##saves to disk!!
                outdata = None
                drive = None


if __name__ == '__main__':
        fusion()