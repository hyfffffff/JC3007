import geopandas

file_json = 'Chinamap.json'
file_shp = 'Chinamap/Chinamap.shp'

data = geopandas.read_file(file_json)
data.to_file(file_shp, driver='ESRI Shapefile', encoding='utf-8')