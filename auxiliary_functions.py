import numpy as np
import rasterio
from rasterio.warp import transform_geom
from rasterio.mask import mask
import os


def DN_to_reflectance(DN_array):
    """
    Convert Sentinel-2 Digital Number (DN) values to reflectance.

    Parameters:
        - DN_array: Array of Sentinel-2 DN values

    Returns:
        - reflectance: Array of reflectance values
    """

    # Assume you have Sentinel-2 DN data

    # Conversion factors
    gain = 0.0001
    offset = 0

    # Convert DN to radiance
    radiance = (DN_array * gain) + offset

    # Convert radiance to reflectance 
    reflectance = radiance / np.pi

    return reflectance

def calculate_ndvi(B04_img, B08_img, aoi):
    """
    Calculate the Normalized Difference Vegetation Index (NDVI) from Sentinel-2 images.

    Parameters:
        - B04_img: Path to the Sentinel-2 Band 4 image file (red channel)
        - B08_img: Path to the Sentinel-2 Band 8 image file (near-infrared channel)
        - aoi: Shapely Polygon object representing the Area of Interest (AOI)

    Returns:
        - ndvi: NDVI array calculated from the input images within the AOI
    """

    # Open the Band 4 image file (red channel)
    with rasterio.open(B04_img) as B04_dataset:
        # Get the CRS of the image
        crs = B04_dataset.crs

        # Transform the AOI geometry to match the image CRS
        aoi_transformed_B04 = transform_geom('EPSG:4326', crs, aoi.__geo_interface__)

        # Clip the Band 4 image to the AOI
        B04_clipped, B04_clipped_transform = mask(B04_dataset, [aoi_transformed_B04], crop=True)
        B04_data = B04_clipped[0, :, :].astype(float)

        # Convert DN values to reflectance values
        B04_data_reflectance = DN_to_reflectance(B04_data)

    # Open the Band 8 image file (near-infrared channel)
    with rasterio.open(B08_img) as B08_dataset:
        # Get the CRS of the image
        crs = B08_dataset.crs

        # Transform the AOI geometry to match the image CRS
        aoi_transformed_B08 = transform_geom('EPSG:4326', crs, aoi.__geo_interface__)

        # Clip the Band 8 image to the AOI
        B08_clipped, B08_clipped_transform = mask(B08_dataset, [aoi_transformed_B08], crop=True)
        B08_data = B08_clipped[0, :, :].astype(float)

        # Convert DN values to reflectance values
        B08_data_reflectance = DN_to_reflectance(B08_data)

        # Calculate the NDVI using the reflectance values
        ndvi = (B08_data_reflectance - B04_data_reflectance) / (B08_data_reflectance + B04_data_reflectance)
        
        
        # Set the output file path
        source_file = os.path.basename(B08_img)
        output_name = 'NDVI_' + os.path.splitext(source_file)[0]+'.tif'
        output_path = os.path.join('temp',output_name)
        # Create a new raster file with the same geospatial properties as the input bands
        profile = B08_dataset.profile
        profile.update(driver='GTiff', dtype=rasterio.float32, count=1)

        with rasterio.open(output_path, 'w', **profile) as output_ds:
            # Write the NDVI array to the output raster file
            output_ds.write(ndvi, 1)    

    return ndvi,output_path



def classify_ndvi(ndvi_file, thresholds):
    """
    Classify an NDVI image into classes based on thresholds.

    Parameters:
        - ndvi_file: Path to the NDVI TIFF file
        - thresholds: List of threshold values for classifying the NDVI

    Returns:
        - classified_image: NumPy array representing the classified image
    """

    # Open the NDVI image file
    with rasterio.open(ndvi_file) as src:
        # Read the NDVI image as a NumPy array
        ndvi_image = src.read(1)
        profile = src.profile

    # Replace 0 for Nan
    ndvi_image = np.where(ndvi_image == 0, np.nan, ndvi_image)


    # Classify the NDVI image based on thresholds
    classified_image = np.zeros_like(ndvi_image, dtype=np.uint8)
    for i, threshold in enumerate(thresholds):
        classified_image[ndvi_image >= threshold] = i + 1

    # Create a new raster file with the same geospatial properties as the NDVI image
    profile.update(driver='GTiff', dtype=rasterio.uint8, count=1)

    # Set the output file path
    source_file = os.path.basename(ndvi_file)
    output_name = 'Classified' + os.path.splitext(source_file)[0]+'.tif'
    output_path = os.path.join('output',output_name)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(classified_image, 1)

    return classified_image, output_path



def measure_class_area(classified_image, class_label, pixel_size):
    """
    Measure the area of a specific class in the classified image.

    Parameters:
        - classified_image: Numpy array representing the classified image
        - class_label: Integer value representing the class label
        - pixel_size: Size of each pixel in meters

    Returns:
        - class_area: Area of the specified class in hectares
    """

    # Identify pixels corresponding to the desired class
    class_pixels = np.where(classified_image == class_label)

    # Calculate the number of pixels in the identified area
    num_pixels = len(class_pixels[0])

    # Multiply the number of pixels by the area of each pixel to obtain the total area in hectares
    class_area = (num_pixels * (pixel_size ** 2)) / 10000

    return class_area