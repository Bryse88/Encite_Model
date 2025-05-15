import yaml

def load_region_config():
    with open('configs/data_source_config.yaml') as f:
        return yaml.safe_load(f)

def is_designated_location(lat, lng, config):
    for region in config['designated_regions']:
        if region['lat_range'][0] <= lat <= region['lat_range'][1] and \
           region['lng_range'][0] <= lng <= region['lng_range'][1]:
            return region['name']
    return None
