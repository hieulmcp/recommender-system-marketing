import googlemaps
import os

def get_latitude_longitude(address_):
    api_key = 'AIzaSyBDlRnrQcAU4fP6cSoD6NoORVis1ztKx8g'
    gmaps_client = googlemaps.Client(api_key)

    geocode_result = gmaps_client.geocode(address_)

    result = geocode_result[0]

    print('Address:..', result['formatted_address'])
    print('Latitude:..',result['geometry']['location']['lat'])
    print('Longitude:..',result['geometry']['location']['lng'])