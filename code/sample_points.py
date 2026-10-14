import numpy as np, pandas as pd, rasterio, os
lab=pd.read_csv('labels.csv')
with rasterio.open('/mnt/user-data/uploads/chennai_dem_glo30_32644.tif') as src:
    rows,cols=rasterio.transform.rowcol(src.transform, lab.x.values, lab.y.values)
rows=np.array(rows); cols=np.array(cols)
H,W=np.load('dem.npy').shape
inb=(rows>=0)&(rows<H)&(cols>=0)&(cols<W)
print('points inside raster:', int(inb.sum()), 'of', len(lab))
def take(name):
    a=np.load(name)
    out=np.full(len(lab), np.nan)
    out[inb]=a[rows[inb],cols[inb]]
    return out
out=pd.DataFrame({'id':np.arange(len(lab)),'x':lab.x,'y':lab.y,'flood':lab.flood,'event_year':lab.event_year,'label_source':lab.label_source})
out['elevation']=take('dem.npy')
out['slope_deg']=take('slope_deg.npy')
out['dist_water_m']=take('dist_water.npy')
out['gsw_occurrence']=take('occ.npy')
if os.path.exists('flowacc.npy'):
    out['flow_acc']=take('flowacc.npy')
out.to_csv('terrain_points.csv',index=False)
print(out.isna().sum().to_string())
print(out.groupby(['event_year','flood'])[['elevation','slope_deg','dist_water_m']].median().round(2).to_string())
