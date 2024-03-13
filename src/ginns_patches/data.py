#%% imports
import numpy as np
import gdal
from gdal import gdalconst
import ogr
import cv2
import os
import h5py
#%% comments

#%% function



def image2Patches(image, sizeTiles_x, sizeTiles_y):
    """ Clip an image/tensor to patches with specific dimention, using padding at the edges.
        The number of patches in x and y dimention and the width and length of the padded area is 
        saved in metaInfo.
        Input:
        param image        numpy array with dim [channels, sizeTiles_y, sizeImage_x]
        param sizeTiles_x  int value which specified the x dim of the patches 
        param sizeTiles_y  int value which specified the y dim of the patches 
        Output:
        return X_data      numpy array with dim [num_patches, sizeTiles_y, sizeTiles_x, channels]
        return metaInfo    tuple with [numPatches_x, numPatches_y, numPadding_x, numPadding_y]
        
        TODO: col-wise,row-wise and start 0 or 1
        TODO: OVERLAP
    """
    if(len(image.shape) == 2):
        image = np.expand_dims(image, 0)
    sizeImage_x = image.shape[2] 
    sizeImage_y = image.shape[1]
    sizeImage_channels = image.shape[0]
    
    tiles_cols = np.ceil(sizeImage_x/sizeTiles_x)
    tiles_rows = np.ceil(sizeImage_y/sizeTiles_y)
    
    patchIndex_cols = np.arange(0, tiles_cols * sizeTiles_x - 1, sizeTiles_x, int)
    patchIndex_cols = np.append(patchIndex_cols, sizeImage_x - 1)
    patchIndex_rows = np.arange(0, tiles_rows * sizeTiles_y - 1, sizeTiles_y, int)
    patchIndex_rows = np.append(patchIndex_rows, sizeImage_y - 1)
    
    size_samples = int(tiles_cols*tiles_rows)
    X_data= np.empty([size_samples, sizeTiles_y, sizeTiles_x, sizeImage_channels])
    
    tt = 0
    
    pad_left = 0
    pad_right = 0
    pad_bottom = 0
    pad_top = 0

    for col_i in range(0, patchIndex_cols.shape[0] - 1):
        for row_i in range(0, patchIndex_rows.shape[0] - 1):
            
            indX_from = patchIndex_cols[col_i]
            indX_to = patchIndex_cols[col_i + 1]
            indY_from = patchIndex_rows[row_i]
            indY_to = patchIndex_rows[row_i + 1]
            
            patch = image[:, indY_from : indY_to, indX_from : indX_to]
            padX = (0, 0)
            if col_i == patchIndex_cols.shape[0]-2:            
                 padX = (0, sizeTiles_x - (indX_to - indX_from))   
            padY = (0, 0)
            if row_i == patchIndex_rows.shape[0]-2:
                 padY = (0, sizeTiles_y - (indY_to - indY_from))   
                 
            patch = np.pad(patch, ((0,0), padY, padX), 'reflect')
            
            patch = np.moveaxis(patch, 0, 2)
            X_data[tt] = patch        
    
            if(padX[0] > 0 ):
                pad_left = padX[0]
            if(padX[1] > 0 ):
                pad_right = padX[1]
            if(padY[0] > 0 ):
                pad_top = padY[0]
            if(padY[1] > 0 ):
                pad_bottom = padY[1]
                
            tt +=1
    metaInfo = (tiles_rows, tiles_cols, pad_left, pad_right, pad_top, pad_bottom)
    
    return X_data, metaInfo

#TODO: comment
def images2Patches(foldername_src, prefix, sizeTiles_x, sizeTiles_y, overlap = 0):
    X_data = np.array([])
    metaInfo = list()
    for file in os.listdir(foldername_src):
        print(foldername_src + file)
        if file.endswith(prefix):
            dataGeo = gdal.Open(foldername_src + file, gdalconst.GA_ReadOnly)
            image = dataGeo.ReadAsArray()
            if overlap>0:
                X_data_i, metaInfo_i = image2Patches(image, sizeTiles_x, sizeTiles_y)
            else:
                X_data_i, metaInfo_i = image2PatchesOverlap(image, sizeTiles_x, sizeTiles_y, overlap)(image, sizeTiles_x, sizeTiles_y, overlap)
            X_data = np.concatenate((X_data, X_data_i), axis=0) if X_data.size else X_data_i
            metaInfo.append(metaInfo_i + ( X_data.shape[0], ))
    return X_data, metaInfo


def image2PatchesOverlap(myImageIn, sizeTiles_x, sizeTiles_y, overlap):
    """ Clip an image/tensor to patches with specific dimention, using padding at the edges.
        Temporarily image in blown up to have the dimension [ceil(H/sX)*sX + 2*ov, ceil(H/sX)*sX + 2*ov]
        The number of patches in x and y dimention and the width and length of the padded area is 
        saved in metaInfo.
        Input:
        param image        numpy array with dim [channels, sizeTiles_y, sizeImage_x]
        param sizeTiles_x  int value which specified the x dim of the patches 
        param sizeTiles_y  int value which specified the y dim of the patches 
        overlap            double the overlap (between zero and something)
        Output:
        return X_data      numpy array with dim [num_patches, sizeTiles_y, sizeTiles_x, channels]
        return metaInfo    tuple with [tiles_row, tiles_col, sizeTiles_x, sizeTiles_y, rc, cc, overlap, rr0_all, cc0_all]
        
    """
    option = np.inf
    
    if myImageIn.ndim == 2:
        myImageIn = myImageIn[:, :, np.newaxis]
    else:
        myImageIn = np.moveaxis(myImageIn, 0, 2)
        


    r, c, f = myImageIn.shape
    # copy image size
    rc = r
    cc = c

    tiles_row = (np.ceil(r / sizeTiles_y)).astype(int)
    tiles_col = (np.ceil(c / sizeTiles_x)).astype(int)
    size_samples = int(tiles_col*tiles_row)
    # margin  options:

    rows2extend_r = np.flipud(myImageIn[2 * r - tiles_row * sizeTiles_y-overlap:r,:, :])
    rows2extend_l = np.flipud(myImageIn[0:overlap,:, :])
    myImageIn = np.append(myImageIn, rows2extend_r, 0)
    myImageIn = np.append(rows2extend_l, myImageIn, 0)
    columns2extend_d = np.fliplr(myImageIn[:, 2 * c - tiles_col * sizeTiles_x-overlap:c,:])
    columns2extend_u = np.fliplr(myImageIn[:, 0:overlap, :])
    myImageIn = np.append(myImageIn, columns2extend_d, 1)
    myImageIn = np.append(columns2extend_u, myImageIn, 1)
    
 

    if  ~np.isinf(option):
        myImageIn[r:-1, :, :] = option
        myImageIn[:, c:-1, :] = option

    r, c, f = myImageIn.shape
    
    count = 0
    
    X_data= np.empty([size_samples, sizeTiles_y+ 2*overlap, sizeTiles_x+ 2*overlap, f], myImageIn.dtype)
    rr0_all = np.empty([size_samples], 'int')
    cc0_all = np.empty([size_samples], 'int')


    for i in range(0, tiles_row):
        for j in range(0, tiles_col):
            rr0 = i * sizeTiles_y
            cc0 = j * sizeTiles_x
            rrp = min(rr0 + sizeTiles_y + 2*overlap, r)
            ccp = min(cc0 + sizeTiles_x + 2*overlap, c)
            # test Image
            imageIn = myImageIn[rr0:rrp, cc0:ccp, :] #???
            X_data[count, :, :, :] = imageIn 
            rr0_all[count] = rr0
            cc0_all[count] = cc0
            count +=1
            
    
    metaInfo = (tiles_row, tiles_col, sizeTiles_x, sizeTiles_y, rc, cc, overlap, rr0_all, cc0_all)
    return X_data, metaInfo

def patches2ImageOverlap(X_data, metaInfo):
    """ Undo the method image2Patches, merging patches to an image/tensor.
        
        Input:
        param X_data    numpy array with dim [num_patches, sizeTiles_y, sizeTiles_x, channels]
        metaInfo        list with important meta-information: 
                            number of tiles (x, y) 
                            size of tiles (x, y)
                            size of original image
                            overlap
                            coordinates of left upper corners of all patches
        
        Output:
        return image      numpy array with dim [num_patches, sizeTiles_y, sizeTiles_x, channels]

        TODO: OVERLAP
        TODO if (np.shape(indArray)[0] <= 1): speed up like else branch
    """  
    
    
    tiles_rows, tiles_cols, sizeTiles_x, sizeTiles_y, rc, cc, overlap, rr0_all, cc0_all = metaInfo
    
    numImages, rout, cout, fout = X_data.shape
    
    
    r = tiles_rows*sizeTiles_y+2*overlap
    c = tiles_cols*sizeTiles_x+2*overlap
    
    
    
    myImageOut = np.zeros((r, c, fout), X_data.dtype)
    
        
    
    for count in range(0, numImages):
        rr0 = rr0_all[count]
        cc0 = cc0_all[count]
        
        rrp = min(r, rr0 + sizeTiles_y+overlap).astype(int)
        ccp = min(c, cc0 + sizeTiles_x+overlap).astype(int)
        
        imageOut = X_data[count, overlap:overlap+sizeTiles_x, overlap:overlap+sizeTiles_x, :]
        #imageOut = np.fliplr(imageOut)
        
        myImageOut[rr0+overlap:rrp, cc0+overlap:ccp, :] = imageOut
    
    
    # #crop
    myImageOut = myImageOut[overlap:overlap+rc, overlap:overlap+cc, :]
    #squeeze to original size
    myImageOut = np.squeeze(myImageOut)
    return myImageOut

def patches2Image(X_data, metaInfo, indArray = np.array([]), order = 'row-wise', start = 1):
    """ Undo the method image2Patches, merging patches to an image/tensor.
        
        Input:
        param X_data    numpy array with dim [num_patches, sizeTiles_y, sizeTiles_x, channels]
        param indArray  linear index of the patches in X_data. Necessary if X_data hast gaps.
        param order     order of the patches "row-wise" or "col-wise"
        param start     start index of the patches 1: -> (1,1) or 0: -> (0,0)
        
        Output:
        return image      numpy array with dim [num_patches, sizeTiles_y, sizeTiles_x, channels]

        TODO: OVERLAP
        TODO if (np.shape(indArray)[0] <= 1): speed up like else branch
    """  
    tiles_rows, tiles_cols, pad_left, pad_right, pad_top, pad_bottom = metaInfo.astype(int)
    if (np.shape(indArray)[0] <= 1):
        image = np.array([])
        tt = 0
        
        if(order == 'col-wise'):
            for col_i in range(0,int(tiles_cols)):
                image_Column = np.array([])
                for row_i in range(0,int(tiles_rows)):
                    image_Column = np.concatenate((image_Column, X_data[tt,:,:,:]), axis=0) if image_Column.size else X_data[tt,:,:,:]
                    tt +=1
                image = np.concatenate((image, image_Column), axis=1) if image.size else image_Column
                
        if(order == 'row-wise'):
            for row_i in range(0,int(tiles_rows)):
                image_Column = np.array([])
                for col_i in range(0,int(tiles_cols)):
                    image_Column = np.concatenate((image_Column, X_data[tt,:,:,:]), axis=0) if image_Column.size else X_data[tt,:,:,:]
                    tt +=1
                image = np.concatenate((image, image_Column), axis=1) if image.size else image_Column                 
    else:     
        sizeTiles_y = X_data.shape[1]
        sizeTiles_x = X_data.shape[2]
        channels = X_data.shape[3]          
        sizeImage_x = tiles_cols*sizeTiles_x
        sizeImage_y = tiles_rows*sizeTiles_y
        image = np.zeros([sizeImage_y, sizeImage_x, channels])        
        patchIndex_cols = np.arange(0, tiles_cols * sizeTiles_x - 1, sizeTiles_x, int)
        patchIndex_cols = np.append(patchIndex_cols, sizeImage_x - 1)
        patchIndex_rows = np.arange(0, tiles_rows * sizeTiles_y - 1, sizeTiles_y, int)
        patchIndex_rows = np.append(patchIndex_rows, sizeImage_y - 1)   
        tt = 0
        if(order == 'col-wise'):
            for col_i in range(start, patchIndex_cols.shape[0] - start):
                for row_i in range(start, patchIndex_rows.shape[0] - start):         
                    linearInd =  sub2Ind(row_i, col_i , tiles_rows, tiles_cols, order, start)
                    if(indArray[tt] == linearInd):
                        #print('(' + str(row_i) + ', ' + str(col_i) +')' + ':' + str(linearInd) + ' ' + str(tt))
                        indX_from = patchIndex_cols[col_i]
                        indX_to = patchIndex_cols[col_i + 1]
                        indY_from = patchIndex_rows[row_i]
                        indY_to = patchIndex_rows[row_i + 1]                    
                        image[indY_from : indY_to, indX_from : indX_to, :] = X_data[tt]
                        tt +=1 
                    if (tt == np.size(indArray)):
                        break
                if (tt == np.size(indArray)):
                    break       
        if(order == 'row-wise'):
            for col_i in range(start, patchIndex_cols.shape[0] - start):
                for row_i in range(start, patchIndex_rows.shape[0] - start):         
                    linearInd =  sub2Ind(row_i, col_i , tiles_rows, tiles_cols, order, start)
                    if(indArray[tt] == linearInd):
                        #print('(' + str(row_i) + ', ' + str(col_i) +')' + ':' + str(linearInd) + ' ' + str(tt))
                        indX_from = patchIndex_cols[col_i]
                        indX_to = patchIndex_cols[col_i + 1]
                        indY_from = patchIndex_rows[row_i]
                        indY_to = patchIndex_rows[row_i + 1]                    
                        image[indY_from : indY_to, indX_from : indX_to, :] = X_data[tt]
                        tt +=1 
                    if (tt == np.size(indArray)):
                        break
                if (tt == np.size(indArray)):
                    break
    image = image[pad_top : (image.shape[0]- pad_bottom+1), pad_left : (image.shape[1]) - pad_right+1, :]  
    return image


def sub2Ind(r_i, c_i , num_rows, num_cols, order='row-wise', start = 1):
    """ Convert row and col index to linear index.
        
        Input:
        param r_i       row index
        param c_i       col index
        param order     order of index "row-wise" or "col-wise"
        param num_rows  int value number of rows
        param num_cols  int value number of cols
        param start     start index of the patches: 1 -> (1,1) or 0 -> (0,0)
        
        Output:
        return [r, c]      row and col index
    """
    if(start == 1):
        if(order == 'col-wise'):
            linearInd = num_rows*(c_i-1) + r_i 
        if(order == 'row-wise'):
             linearInd = num_cols*(r_i-1) + c_i  
    if(start == 0):
        if(order == 'col-wise'):
            linearInd = num_rows*(c_i) + r_i 
        if(order == 'row-wise'):
             linearInd = num_cols*(r_i) + c_i  
    return [int(linearInd)]


def ind2Sub(linearInd, num_rows, num_cols, order = 'row-wise', start = 1):
    """ Convert linear index to row and col index.
        
        Input:
        param linearInd int index
        param order     order of index "row-wise" or "col-wise"
        param num_rows  int value number of rows
        param num_cols  int value number of cols
        param start     start index of the patches: 1 -> (1) or 0 -> (0)
        
        Output:
        return [r, c]   row and col index
    """
    if start == 1:
        if order == 'row-wise':
            r_i = np.ceil(linearInd/num_cols)
            c_i = linearInd - ((r_i-1)*num_cols)
        if order == 'col-wise':
            c_i = np.ceil(linearInd/num_rows)
            r_i = linearInd - ((c_i-1)*num_rows)
    if start == 0:
        if order == 'row-wise':
            r_i = np.floor(linearInd/num_cols)
            c_i = linearInd - (r_i*num_cols)
        if order == 'col-wise':
            c_i = np.floor(linearInd/num_rows)
            r_i = linearInd - (c_i*num_rows)
    return [int(r_i) , int(c_i)]


def patchIndexSub2GeoTrans(r_i, c_i, sizePatches_y, sizePatches_x, geoTransform_origin,  start = 1):
    """ Calculate the geo transformation for one patch according to the origin 
        image geo transformation and a the row and col index.
        
        Input:
        param indexSub              row and col index
        param sizePatches_y         size of the patch in y (rows) direction
        param sizePatches_x         size of the patch in x (cols) directions
        param geoTransform_origin   geo trans. of the origin image (see gdal)

        param start                 start index of the patches: 1 -> (1,1) or 0 -> (0,0)
        
        Output:
        return geoTransform         geo trans. of current patch (see gdal)

        TODO: OVERLAP
        TODO: Is order necessary?? order ='row-wise'    param order                 order of index "row-wise" or "col-wise"
    """
    #r_i = indexSub[0]
    #c_i = indexSub[1]
    #if(order == 'row-wise'):
    easting = geoTransform_origin[0] + (c_i-start) * sizePatches_x * geoTransform_origin[1]
    northing = geoTransform_origin[3] + (r_i-start) * sizePatches_y * geoTransform_origin[5]
    #if(order == 'col-wise'):
        #easting = geoTransform_origin[0] +  (r_i-start) * sizePatches_y * geoTransform_origin[1]
        #northing = geoTransform_origin[3] + (c_i-start) * sizePatches_x * geoTransform_origin[5]
    geoTransform = (easting, geoTransform_origin[1], 0, northing, 0, geoTransform_origin[5]) 
    return geoTransform


def patchIndexInd2GeoTrans(indexInd, sizePatches_y, sizePatches_x, numPatches_y, numPatches_x, geoTransform_origin, order = 'row-wise',  start = 1):
    """ Calculate the geo transformation for one patch according to the origin 
        image geo transformation and a linear patch index.
        
        Input:
        param indexInd              linear index
        param sizePatches_y         size of the patch in y (rows) direction
        param sizePatches_x         size of the patch in x (cols) directions
        param numPatches_y          number of the patches in y (rows) direction
        param numPatches_x           number of the patches in x (cols) directions
        param geoTransform_origin   geo trans. of the origin image (see gdal)
        param order                 order of index "row-wise" or "col-wise"
        param start                 start index of the patches: 1 -> (1,1) or 0 -> (0,0)
        
        Output:
        return geoTransform         geo trans. of current patch (see gdal)

        TODO: OVERLAP
        TODO: ERROR USING order ?
    """
    sub = ind2Sub(indexInd, numPatches_y, numPatches_x, order, start)
    geoTransform = patchIndexSub2GeoTrans(sub[0], sub[1], sizePatches_y, sizePatches_x, geoTransform_origin, start)
    return geoTransform

def array2GeoTiff(filename_dst, array, geotransform, geoprojection = '', dst_datatype = gdal.GDT_Byte):
    """ Write an numpy array to an geotiff file.
        
        Input:
        param filename              filename for the output file.
        param array                 numpy array with image data [rows, cols, bands]
        param geotransform          geo trans. (X, r_x 0, Y, 0, -r_y) (see gdal)
        param geoprojection         geo projection (see gdal)
    """
    rows = array.shape[0]
    cols = array.shape[1]
    numBands = array.shape[2]    
    dst_name = 'GTiff'
    driver = gdal.GetDriverByName(dst_name)
    filename_dst = changePrefix(filename_dst, '.tif')
    dst_ds = driver.Create(filename_dst, cols, rows, numBands, dst_datatype)
    for band in range(1, numBands + 1):
        dst_ds.GetRasterBand(band).WriteArray(array[:, :, band - 1])
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(geoprojection)
    dst_ds = None


def raster2Size(filename_src, filename_dst, size, interpolation = gdalconst.GRA_Bilinear):
    """ Load and resize an geotiff and save the result in a new file.
        
        Input:
        param filename_src      filename of the source file
        param filename_dst      filename of the resulting file
        param size              size in x and y direction of the new file or
                                resize factor
        param interpolation     interpolation method:
                                    gdalconst.GRA_Bilinear
                                    gdalconst.GRA_Lanczos
                                    gdalconst.GRA_Average
                                    gdalconst.GRA_NearestNeighbour
                                    ...
    """
    dataGeo = gdal.Open(filename_src, gdalconst.GA_ReadOnly) 
    geoTransform = dataGeo.GetGeoTransform()
    geoProjection = dataGeo.GetProjection()
    if isinstance(size, tuple):
        rows = size[0]
        cols = size[1]
    else:
         rows = int(dataGeo.RasterYSize * size)
         cols = int(dataGeo.RasterXSize * size) 
    res_east = geoTransform[1]*dataGeo.RasterXSize/ rows
    res_north = geoTransform[-1]*dataGeo.RasterXSize/ cols
    driver= gdal.GetDriverByName('GTiff')
    filename_dst = changePrefix(filename_dst, '.tiff')
    dst_ds = driver.Create(filename_dst, rows, cols, dataGeo.RasterCount, dataGeo.GetRasterBand(1).DataType)
    dst_ds.SetGeoTransform((geoTransform[0], res_east, 0, geoTransform[3], 0, res_north))
    dst_ds.SetProjection(geoProjection)
    gdal.ReprojectImage(dataGeo, dst_ds, None, None,interpolation)


def raster2Resolution(filename_src, filename_dst, resolution, interpolation = gdalconst.GRA_Bilinear):
    """ Load and resize an geotiff and save the result with an specific resolution.

        Input:
        param filename_src      filename of the source file
        param filename_dst      filename of the resulting file
        param resolution        new file resolution
                                resize factor
        param interpolation     interpolation method:
                                    gdalconst.GRA_Bilinear
                                    gdalconst.GRA_Lanczos
                                    gdalconst.GRA_Average
                                    gdalconst.GRA_NearestNeighbour
                                    ...
    """
    dataGeo = gdal.Open(filename_src, gdalconst.GA_ReadOnly)
    geoTransform = dataGeo.GetGeoTransform()
    geoProjection = dataGeo.GetProjection()
    if isinstance(resolution, tuple):
        res_east = resolution[0]
        res_north = - np.abs(resolution[1])
    else:
        res_east = resolution
        res_north = - np.abs(res_east)
    rows = int(np.round(geoTransform[1] * dataGeo.RasterYSize / res_east))
    cols = int(np.abs(np.round(geoTransform[-1] * dataGeo.RasterXSize /res_north)))
    driver = gdal.GetDriverByName('GTiff')
    filename_dst = changePrefix(filename_dst, '.tiff')
    dst_ds = driver.Create(filename_dst, rows, cols, dataGeo.RasterCount, dataGeo.GetRasterBand(1).DataType) #gdalconst.GDT_Int32
    dst_ds.SetGeoTransform((geoTransform[0], res_east, 0, geoTransform[3], 0, res_north))
    dst_ds.SetProjection(geoProjection)
    gdal.ReprojectImage(dataGeo, dst_ds, None, None, interpolation)


def reprojectRaster(filename_src, filename_dst, epsg_code = 'EPSG:25832'):
    """ Load and reproject a given geotiff and save the result in a new file.
        Input:
            param filename_src      filename of the source file
            param filename_dst      filename of the resulting file
            param epsg_code         epsg code: 'EPSG:XXXXX'
    """
    gdal.Warp(filename_dst, filename_src, dstSRS= epsg_code)


def resampleArray(array, size, interpolation = cv2.INTER_LINEAR):
    """ Resize a given numpy array according to a specific interpolation method.
        
        Input:
        param array              numpy array [rows, cols, channels]
        param size               new size [new_rows, new_cols] or resize factor
        param interpolation      opencv interpolation method:
                                   cv2.INTER_AREA 
                                   cv2.INTER_NEAREST 
                                   cv2.INTER_LINEAR 
                                   cv2.INTER_CUBIC
                                   cv2.INTER_LANCZOS4 
                                   
        Output:
        return array_resampled         geo trans. of current patch (see gdal)
    """
    if isinstance(size, tuple):
        rows = size[0]
        cols = size[1]
    else:
         rows = int(array.shape[0] * size)
         cols = int(array.shape[1] * size) 
    array_resampled = cv2.resize( array, (rows, cols), interpolation = interpolation )
    return(array_resampled)


def checkPefix(input_string, prefix):
    """ Check if a given string fit to a prefix.

        Input:
        param input_string      string value
        param prefix            string prefix to compare

        Output:
        return check            boolean value
    """
    check = bool(input_string[- len(prefix):].find(prefix) + 1)
    return check


def changePrefix(input_string, prefix):
    """ Change the data typ prefix of a given string.

        Input:
        param input_string      string value
        param prefix            string prefix to compare

        Output:
        return input_string     string with new prefix
    """
    if not checkPefix(input_string, prefix):
        if input_string.find('.') != -1:
            input_string = input_string[:input_string.find('.')] + prefix
        else:
            input_string = input_string + prefix
    return input_string


def filename2Coord(filename_src, mult = 1000, offset = -32*10**6):
    """ Convert a string to coordinates. The string must consists of numbers and the coord. must delimited by "_".

        Input:
        param filename_src      string value
        param mult              scale value for the coord. (ex. for UTM coord.)
        param offset            offset of the first coord. (ex. for UTM coord.)

        Output:
        return easting          float value of the first coord.
        return northing         float value of the second coord.
    """
    file = filename_src[:filename_src.find(".")]
    easting = float(file[0:file.find("_")]) * mult + offset
    northing = float(file[file.find("_") + 1 :])* mult
    return easting, northing


def createBBoxGeom(ul_x, ul_y, lr_x, lr_y):
    """ Create the bounding box geometry from two given points .

        Input:
        param ul_x         first coord. of the upper left point
        param ul_y         second coord. of the upper left point
        param lr_x         first coord. of the  lower right point
        param lr_y         second coord. of the lower right point

        Output:
        return bboxGeom   gdal geometry object with the bounding box
    """
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(ul_x, ul_y)
    ring.AddPoint(lr_x, ul_y)
    ring.AddPoint(lr_x, lr_y)
    ring.AddPoint(ul_x, lr_y)
    ring.AddPoint(ul_x, ul_y)
    bboxGeom = ogr.Geometry(ogr.wkbPolygon)
    bboxGeom.AddGeometry(ring)
    return bboxGeom


def coord2BBox(easting, northing, size_east, size_north = None, res_east = 1, res_north = None, ref_point = "ul"):
    """ Calculate the bounding box geometry from the image properties.

        Input:
        param easting           first coord. of the reference point
        param northing          second coord. of the reference point
        param size_east         image dimension in east direction (number of cols)
        param size_north        image dimension in north direction (number of rows)
        param res_east          pixel resolution in east direction
        param res_north         pixel resolution in north direction
        param ref_point         reference point for the image. Lower left "ll" or upper left "ul"

        Output:
        return rasterGeometry   gdal geometry object with bounding box
    """
    if size_north is None:
        size_north = size_east
    if res_north is None:
        res_north = res_east
    if ref_point == "ll":
        northing = northing + (size_east*np.abs(res_north))
    ul_x = easting
    ul_y = northing
    lr_x = easting + (size_north * res_east)
    lr_y = northing + (size_north * res_north)
    bboxGeometry = createBBoxGeom(ul_x, ul_y, lr_x, lr_y)
    return bboxGeometry

#TODO Comment
def coord2RowCol(easting, northing, min_east, max_north, delta_east, delta_north):
    row = (easting - min_east) / delta_east + 1
    col = (max_north - northing) / delta_north + 1
    return row, col

#TODO Comment
def loadGeoTiff(foldername_src):
    dataGeo = gdal.Open(foldername_src, gdalconst.GA_ReadOnly)
    image = dataGeo.ReadAsArray()
    geotransform = dataGeo.GetGeoTransform()
    geoprojection = dataGeo.GetProjection()
    return image, geotransform, geoprojection

#TODO Comment
def loadTiff(foldername_src):
    dataGeo = gdal.Open(foldername_src, gdalconst.GA_ReadOnly)
    image = dataGeo.ReadAsArray()
    return image

#TODO Comment
import timeit
def loadImages(foldername_src, prefix):
    #gdal.SetCacheMax(2 ** 30)
    start = timeit.default_timer()
    X_data = np.array([])
    for file in os.listdir(foldername_src):
        print(foldername_src + file)
        if file.endswith(prefix):
            startRead = timeit.default_timer()
            #dataGeo = gdal.Open(file, gdalconst.GA_ReadOnly)
            dataGeo = gdal.Open(foldername_src   + file, gdalconst.GA_ReadOnly)
            image = dataGeo.ReadAsArray()
            stopRead = timeit.default_timer()
            print('     Time Read: ', stopRead - startRead)
            image = image[np.newaxis, :]
            X_data = np.concatenate((X_data, image), axis=0) if X_data.size else image
            stopRead = timeit.default_timer()
            print('     Time add: ', stopRead - startRead)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    return X_data

"""
def coords2TileIndex(values_east, values_north, start = '1', order = 'row-wise'):
    index_list = list()
    for i in range(0,len(values_east)):
        if order ='row_wise':

        if order ='col_wise':

    return index_list
"""

def loadH5(path, listTag):
    """ Enter a h5 container and load the specified data. If tags is type list the function return a list of datasets.

        Input:
        param path            path to h5-file
        param listTag         tag which specifiy the requested dataset
        or
        param listTag         list of tags

        Output:
        return listData      numpy array or list of numpy arrays
    """

    if (type(listTag) == list):
        np.shape(listTag)
        listData = list()
        with h5py.File(path,'r') as hf:
            for tag in listTag:    
                X = hf.get(tag)
                X= np.array(X)
                listData.append(X)
        return  listData
    else:
        with h5py.File(path,'r') as hf:
            X = hf.get(listTag)
            X= np.array(X)
        return  X
