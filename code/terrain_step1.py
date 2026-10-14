import rasterio, numpy as np
from scipy import ndimage
with rasterio.open('/mnt/user-data/uploads/chennai_dem_glo30_32644.tif') as src:
    dem=src.read(1).astype('float64'); tf=src.transform
with rasterio.open('/mnt/user-data/uploads/chennai_gsw_occurrence_32644.tif') as src:
    occ=src.read(1).astype('float64')
dem_f=np.where(np.isnan(dem), np.nan, dem)
z=np.nan_to_num(dem_f, nan=0.0)
cs=30.0
p=np.pad(z,1,mode='edge')
a=p[:-2,:-2]; b=p[:-2,1:-1]; c=p[:-2,2:]
d=p[1:-1,:-2]; f=p[1:-1,2:]
g=p[2:,:-2]; h=p[2:,1:-1]; i=p[2:,2:]
dzdx=((c+2*f+i)-(a+2*d+g))/(8*cs)
dzdy=((g+2*h+i)-(a+2*b+c))/(8*cs)
slope_deg=np.degrees(np.arctan(np.hypot(dzdx,dzdy)))
slope_deg[np.isnan(dem_f)]=np.nan
water=occ>50
dist=ndimage.distance_transform_edt(~water)*cs
np.save('dem.npy', dem_f); np.save('slope_deg.npy', slope_deg); np.save('dist_water.npy', dist); np.save('occ.npy', occ)
print('slope deg: min/max/median', np.nanmin(slope_deg).round(2), np.nanmax(slope_deg).round(2), np.nanmedian(slope_deg).round(2))
print('dist water m: min/max/median', dist.min(), dist.max().round(0), np.median(dist).round(0))
print('water cells frac', round(water.mean(),4), ' sea-like DEM==0 frac', round((z==0).mean(),4))
