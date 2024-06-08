import ee
import geemap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from mgwr.utils import shift_colormap, truncate_colormap


def gee_to_mgwr(gee_image, dependent_variable, independent_variables,
                scale, region_of_interest, projection):
    """
    Conducts MGWR analysis on a Google Earth Engine image.

    Args:
        gee_image (ee.Image): Google Earth Engine image to analyze.
        dependent_variable (str): Name of the band in the image to use as the dependent variable.
        independent_variables (list): List of strings representing the names of bands in the image to use as independent variables.
        scale (float): Scale in meters at which to evaluate the MGWR.
        region_of_interest (ee.Geometry): Region of interest to analyze.
        projection (ee.Projection): Projection of the input image.

    Returns:
        list: A list containing:
            - mgwr_results (MGWRResults): Results of the MGWR analysis.
            - band_names (list): Names of the bands used in the analysis.
            - dependent_variable_data (pd.Series): Values of the dependent variable.
            - independent_variables_data (pd.DataFrame): Values of the independent variables.
            - mask (ee.Image): Mask used to filter out invalid pixels.
    """

    band_names = [dependent_variable] + independent_variables
    n_proc = 2  # Number of processors for parallel processing.
    pool = mp.Pool(n_proc)

    # Create a mask by combining individual band masks.
    mask = gee_image.select(band_names).mask().reduce(ee.Reducer.product())

    # Add coordinate data to the image.
    lon_lat = gee_image.pixelLonLat()
    gee_image = gee_image.addBands(lon_lat).updateMask(mask)

    # Aggregate the image into grids.
    agg_grid = geemap.create_grid(region_of_interest, scale, projection)
    agg_model_fc = geemap.zonal_stats(
        gee_image,
        agg_grid,
        statistics_type='MEAN',
        scale=scale,
        crs=projection,
        return_fc=True,
        tile_scale=16.0
    )

    # Extract data from the aggregated image.
    data_dict = agg_model_fc.reduceColumns(ee.Reducer.toList().repeat(4), band_names + ['longitude', 'latitude'])
    agg_model_df = pd.DataFrame(list(zip(*data_dict.getInfo()['list'])), columns=band_names + ['longitude', 'latitude'])

    # Extract the coordinates and data.
    lon = agg_model_df['longitude']
    lat = agg_model_df['latitude']
    coord_list = list(zip(lon, lat))
    dependent_variable_data = agg_model_df[dependent_variable]
    independent_variables_data = agg_model_df[independent_variables]

    # Normalize the data.
    n_X = (independent_variables_data - independent_variables_data.mean()) / independent_variables_data.std()
    e_y = (dependent_variable_data - dependent_variable_data.mean()) / dependent_variable_data.std()

    # Calculate the MGWR.
    mgwr_selector = Sel_BW(coord_list, e_y.values[:, None], n_X.values, spherical=True, multi=False)
    mgwr_bw = mgwr_selector.search(pool=pool)
    mgwr_results = GWR(coord_list, e_y.values[:, None], n_X.values, mgwr_bw, constant=False).fit(pool=pool)

    # Close the pool and return the results.
    pool.close()
    pool.join()

    print(f'scale: {scale}')
    print(f'AIC: {mgwr_results.aic}')
    print(f'AICc: {mgwr_results.aicc}')
    print(f'BIC: {mgwr_results.bic}')
    print(f'Adj_R2: {mgwr_results.adj_R2}')
    print(f'R2: {mgwr_results.R2}')

    return [mgwr_results, band_names, dependent_variable_data, independent_variables_data, mask]

def package_box_region(gee_image, scale, region_of_interest, mgwr_results,
                       dependent_variable_data, independent_variables_data,
                       projection):
    """
    Packages the results of an MGWR analysis into a format suitable for image conversion.

    Args:
        gee_image (ee.Image): Google Earth Engine image used in the analysis.
        scale (float): Scale in meters at which the MGWR was performed.
        region_of_interest (ee.Geometry): Region of interest for the analysis.
        mgwr_results (MGWRResults): Results of the MGWR analysis.
        dependent_variable_data (pd.Series): Values of the dependent variable.
        independent_variables_data (pd.DataFrame): Values of the independent variables.
        projection (ee.Projection): Projection of the input image.

    Returns:
        tuple: A tuple containing:
            - data_mat_list (list): List of numpy arrays containing data for each band.
            - transform_mat (list): Transformation matrix for the image.
    """

    # Calculate the bounding box of the region of interest.
    bounding_box = geemap.create_grid(region_of_interest, scale, projection).geometry().bounds()

    # Extract pixel coordinates and values within the bounding box.
    pixel_data = gee_image.pixelLonLat().reduceRegion(
        reducer=ee.Reducer.toList(),
        geometry=bounding_box,
        scale=scale,
        crs=projection,
        maxPixels=10000000,
        bestEffort=True
    ).getInfo()
    pixel_df = pd.DataFrame.from_dict(pixel_data)

    # Extract MGWR results coordinates.
    mgwr_coords = pd.DataFrame(mgwr_results.model.coords, columns=['longitude', 'latitude'])

    # Package the data into arrays.
    band_array = mgwr_results.params.T
    tval_array = mgwr_results.filter_tvals().T
    predy_array = mgwr_results.predy.T
    y_norm = mgwr_results.y
    X_norm = mgwr_results.X.T
    local_R2 = mgwr_results.localR2.T
    RSS = mgwr_results.RSS.T
    TSS = mgwr_results.TSS.T
    BSE = mgwr_results.bse.T

    data_arrays = np.vstack([
        band_array,
        predy_array,
        tval_array,
        dependent_variable_data,
        independent_variables_data.T,
        y_norm,
        X_norm,
        local_R2,
        RSS,
        TSS,
        BSE
    ])

    data_gdf = mgwr_coords.join(pd.DataFrame(data_arrays).T)
    rect_df_with_data = pixel_df.merge(data_gdf, on=['longitude', 'latitude'], how='left').fillna(-9999)

    # Calculate unique longitudes and latitudes.
    unique_lons = np.unique(rect_df_with_data['longitude'])
    unique_lats = np.unique(rect_df_with_data['latitude'])
    shape = (len(unique_lats), len(unique_lons))

    # Create a list of data matrices for each band.
    data_mat_list = [
        np.flipud(rect_df_with_data[col].values.reshape(shape))
        for col in rect_df_with_data.iloc[:, 2:]
    ]

    # Calculate upper left coordinate and transformation matrix.
    roi_coords = np.array(bounding_box.getInfo()['coordinates']).T
    west_lon = roi_coords[0].min()
    north_lat = roi_coords[1].max()
    lat_scale = (north_lat - roi_coords[1].min()) / shape[0]
    lon_scale = (roi_coords[0].max() - west_lon) / shape[1]
    transform_mat = [lon_scale, 0, west_lon, 0, -lat_scale, north_lat]

    return data_mat_list, transform_mat

def mgwr_to_ee(image, DV_band, IV_bands, scales, roi, projection):
    """
    Converts MGWR results to Earth Engine images for multiple scales.

    Args:
        image (ee.Image): Google Earth Engine image to analyze.
        DV_band (str): Dependent variable band name.
        IV_bands (list): List of independent variable band names.
        scales (list): List of scales for analysis.
        roi (ee.Geometry): Region of interest for the analysis.
        projection (ee.Projection): Projection of the input image.

    Returns:
        tuple: A tuple containing:
            - DV_list (list): List of dependent variable data for each scale.
            - IV_list (list): List of independent variable data for each scale.
            - mgwr_list (list): List of MGWR results for each scale.
            - ee.ImageCollection: Collection of Earth Engine images with MGWR results.
    """

    image_list = []
    mgwr_list = []
    DV_list = []
    IV_list = []

    for scale in scales:
        # MGWR results
        mgwr_results, band_names, DV_data, IV_data, mask = gee_to_mgwr(
            image, DV_band, IV_bands, scale, roi, projection)

        data_mat, transform_mat = package_box_region(image, scale, roi, mgwr_results,
                                                     DV_data, IV_data, projection)

        # Names for conversion into image
        data_names = [name + '_coef' for name in band_names[1:]]
        t_value_names = [name + '_t_values' for name in data_names]
        data_names += ['predicted_y'] + t_value_names + band_names + \
                      [name + '_normalized' for name in band_names] + ['local_R2'] + \
                      ['RSS'] + ['TSS'] + [name + '_BSE' for name in band_names[1:]]

        # Transpose data
        data_np = np.transpose(data_mat, (0, 2, 1))

        # Conversion into image
        mgwr_img = geemap.numpy_to_ee(data_np, 'EPSG:4326', transform=transform_mat, band_names=data_names)

        # Apply original masks
        mgwr_img = mgwr_img.updateMask(mgwr_img.neq(-9999))

        # Masking with t-value function
        for band, mask_band in zip(data_names, t_value_names):
            mgwr_img = mgwr_img.addBands(
                mgwr_img.select(band).updateMask(mgwr_img.select(mask_band).neq(0)).rename(band + '_masked'))

        # Set properties metadata on image
        mgwr_img = mgwr_img.set({'scale': scale, 'AIC': mgwr_results.aic, 'AICc': mgwr_results.aicc,
                                 'BIC': mgwr_results.bic, 'Adj_R2': mgwr_results.adj_R2, 'R2': mgwr_results.R2,
                                 'BWs': str(mgwr_results.model.bw)})

        mgwr_list.append(mgwr_results)
        image_list.append(mgwr_img)
        DV_list.append(DV_data)
        IV_list.append(IV_data)

    return DV_list, IV_list, mgwr_list, ee.ImageCollection.fromImages(image_list)

def get_min_max_vis(image,band_name,scale=None,region=None):
  max_=geemap.image_max_value(image.select(band_name),region,scale).getInfo()[band_name]
  min_=geemap.image_min_value(image.select(band_name),region,scale).getInfo()[band_name]

  cmap = plt.cm.seismic
  if (min_ < 0) & (max_ < 0):
    cmap = truncate_colormap(cmap, 0.0, 0.5)
  elif (min_ > 0) & (max_ > 0):
    cmap = truncate_colormap(cmap, 0.5, 1.0)
  else:
    cmap = shift_colormap(cmap, start=0.0, midpoint=1 - max_/(max_ + abs(min_)),
                          stop=1., name=str(np.random.random_sample()) )

  palette = []
  for i in range(cmap.N):
    rgba = cmap(i)
    # rgb2hex accepts rgb or rgba
    palette.append(matplotlib.colors.rgb2hex(rgba))



  vis_params = {'min':min_,'max':max_,'palette':palette,
                'bands':band_name}



  return vis_params

def mgwr_add_to_map(collection,band_name,region,project=False,projection=None,num_images=1):
  listOfImages = collection.toList(collection.size())

  max_=geemap.image_max_value(collection.max().select(band_name),region).getInfo()[band_name]
  min_=geemap.image_min_value(collection.min().select(band_name),region).getInfo()[band_name]

  cmap = plt.cm.seismic
  if (min_ < 0) & (max_ < 0):
    cmap = truncate_colormap(cmap, 0.0, 0.5)
  elif (min_ > 0) & (max_ > 0):
    cmap = truncate_colormap(cmap, 0.5, 1.0)
  else:
    cmap = shift_colormap(cmap, start=0.0, midpoint=1 - max_/(max_ + abs(min_)),
                          stop=1., name=str(np.random.random_sample()) )

  palette = []
  for i in range(cmap.N):
    rgba = cmap(i)
    # rgb2hex accepts rgb or rgba
    palette.append(matplotlib.colors.rgb2hex(rgba))



  vis_params = {'min':min_,'max':max_,'palette':palette,
                'bands':band_name}

  Map.add_colorbar(vis_params, label=band_name, layer_name=band_name+' colorbar')

  for i in range(num_images):
    image = ee.Image(listOfImages.get(i));
    scale = image.get('scale')
    if project:
      Map.addLayer(image.reproject(projection,scale=scale),vis_params,band_name
                   #+ ', scale: '
                   + ', image: '
                   + str(i)
      #+ str(scale.getInfo())
      )
    else:
      Map.addLayer(image,vis_params,band_name
                   + ', image: '
                   + str(i)
                   #+ ', scale: '
      #+ str(scale.getInfo())
      )
  return vis_params
