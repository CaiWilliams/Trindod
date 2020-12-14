import rasterio
import numpy as np
def getCoordinatePixel(map,lon,lat):
    # open map
    dataset = rasterio.open(map)
    # get pixel x+y of the coordinate
    py, px = dataset.index(lon, lat)
    # create 1x1px window of the pixel
    window = rasterio.windows.Window(px - 1//2, py - 1//2, 1, 1)
    # read rgb values of the window
    clip = dataset.read(window=window)
    return(clip)

for i in range(-180,180,1):
    print(getCoordinatePixel("PVYeild\World_PVOUT_GISdata_LTAy_DailySum_GlobalSolarAtlas_GEOTIFF\PVOUT.tif",30,i))