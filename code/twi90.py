import numpy as _np; _np.in1d = _np.isin
import numpy as np, rasterio
from rasterio.transform import from_origin
from pysheds.grid import Grid
import time
z=np.load('dem90.npy').astype('float64')
z=np.where(np.isnan(z), -9999, z)
tf=from_origin(315210, 1482030, 90, 90)
import rasterio.crs
with rasterio.open('dem90_fill.tif','w',driver='GTiff',height=z.shape[0],width=z.shape[1],count=1,dtype='float64',crs='EPSG:32644',transform=tf,nodata=-9999) as o:
    o.write(z,1)
t=time.time()
grid=Grid.from_raster('dem90_fill.tif')
r=grid.read_raster('dem90_fill.tif')
r=grid.fill_depressions(r)
r=grid.resolve_flats(r)
fdir=grid.flowdir(r)
acc=grid.accumulation(fdir)
print('acc done in', round(time.time()-t,1),'s', flush=True)
np.save('flowacc90.npy', np.asarray(acc,dtype='float64'))
print('acc max', float(np.nanmax(np.asarray(acc))), flush=True)
