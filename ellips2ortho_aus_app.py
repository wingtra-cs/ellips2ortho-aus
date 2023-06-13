import subprocess
import sys

GDAL = 'https://github.com/girder/large_image_wheels/raw/wheelhouse/GDAL-3.8.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl#sha256=86bdb99f6481b6bc1751f55295b118fe18d1dba3d0327863bb16741f9ded7409'
@st.cache_data
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install(GDAL)

from osgeo import gdal
from scipy.interpolate import griddata
import math as m
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
import zipfile

# Convert geographic coordinates to cartesian

def geo_to_cart(lat, lon, h):
    '''
    Parameters
    ----------
    lat : float
        latitude of point.
    lon : float
        longitude of point.
    h : float
        ellipsoidal height of point.

    Returns
    -------
    x : float
        cartesian X value of point.
    y : float
        cartesian Y value of point.
    z : flpat
        cartesian Z value of point.

    '''
    
    lat_rad = (lat/180)*m.pi
    lon_rad = (lon/180)*m.pi
    
    a = 6378137.00
    f = 1/298.257222101
    
    e2 = 2*f - f*f
    v = a/(m.sqrt(1 - e2*m.sin(lat_rad)*m.sin(lat_rad)))

    x = (h + v)*m.cos(lat_rad)*m.cos(lon_rad)
    y = (h + v)*m.cos(lat_rad)*m.sin(lon_rad)
    z = ((1 - e2)*v + h)*m.sin(lat_rad)
    
    return x, y, z

# Transform between cartesian GDA94 and GDA20

def cart_to_cart(source_datum, x0, y0, z0):
    '''
    
    Parameters
    ----------
    source_datum : str
        CRS of base station used for PPK geotagging.
    x0 : float
        initial cartesian X value of point.
    y0 : float
        initial cartesian Y value of point.
    z0 : float
        initial cartesian Z value of point.

    Returns
    -------
    x1 : TYPE
        converted cartesian X value of point at new CRS.
    y1 : TYPE
        converted cartesian X value of point at new CRS.
    z1 : TYPE
        converted cartesian X value of point at new CRS.

    '''

    Tx = 0.06155
    Ty = -0.01087
    Tz = -0.04019
    Sc = -0.009994
    Rx = -0.0394924
    Ry = -0.0327221
    Rz = -0.0328979
    
    RxRad = ((Rx/3600)/180)*m.pi
    RyRad = ((Ry/3600)/180)*m.pi
    RzRad = ((Rz/3600)/180)*m.pi
    Scale = 1 + (Sc/1000000)
    
    T = np.zeros(shape=(3,1))
    T[0][0] = Tx
    T[1][0] = Ty
    T[2][0] = Tz
    
    R = np.zeros(shape=(3,3))
    R[0][0] = m.cos(RyRad)*m.cos(RzRad)
    R[0][1] = m.cos(RyRad)*m.sin(RzRad)
    R[0][2] = -m.sin(RyRad)
    
    R[1][0] = m.sin(RxRad)*m.sin(RyRad)*m.cos(RzRad) - m.cos(RxRad)*m.sin(RzRad) 
    R[1][1] = m.sin(RxRad)*m.sin(RyRad)*m.sin(RzRad) + m.cos(RxRad)*m.cos(RzRad) 
    R[1][2] = m.sin(RxRad)*m.cos(RyRad)

    R[2][0] = m.cos(RxRad)*m.sin(RyRad)*m.cos(RzRad) + m.sin(RxRad)*m.sin(RzRad) 
    R[2][1] = m.cos(RxRad)*m.sin(RyRad)*m.sin(RzRad) - m.sin(RxRad)*m.cos(RzRad) 
    R[2][2] = m.cos(RxRad)*m.cos(RyRad)
    
    Xold = np.zeros(shape=(3,1))
    Xold[0][0] = x0
    Xold[1][0] = y0
    Xold[2][0] = z0
    
    Rinv = np.linalg.inv(R)
    
    if source_datum == 'GDA94':
        Xnew = T + Scale*np.matmul(R, Xold)
    else:
        Xnew = np.matmul(Rinv, (1/Scale)*(Xold - T))
    
    x1, y1, z1 = Xnew[0][0], Xnew[1][0], Xnew[2][0]
    
    return x1, y1, z1

# Convert between carteisan to geographic coordiantes

def cart_to_geo(x1, y1, z1):
    '''

    Parameters
    ----------
    x1 : float
        cartesian X value of point.
    y1 : float
        cartesian Y value of point.
    z1 : float
        cartesian Z value of point.

    Returns
    -------
    lat : float
        latitude of point.
    lon : float
        longitude of point.
    h : float
        ellipsoidal height of point.

    '''
    
    a = 6378137.00
    f = 1/298.257222101
    
    e2 = 2*f - f*f
    p = m.sqrt(x1*x1 + y1*y1)
    r = m.sqrt(p*p + z1*z1)
    u = m.atan((z1/p)*((1-f) + (e2*a)/r))
    
    lat_top = z1*(1-f) + e2*a*m.sin(u)*m.sin(u)*m.sin(u)
    lat_bot = (1-f)*(p - e2*a*m.cos(u)*m.cos(u)*m.cos(u))
    
    lon_rad = m.atan(y1/x1)
    if lon_rad < 0:
        lon = 180*(m.pi + lon_rad)/m.pi
    else:
        lon = 180*lon_rad/m.pi
    
    lat_rad = m.atan(lat_top/lat_bot)
    lat = 180*lat_rad/m.pi
    
    h = p*m.cos(lat_rad) + z1*m.sin(lat_rad) - a*m.sqrt(1 - e2*m.sin(lat_rad)*m.sin(lat_rad))

    return lat, lon, h

# GDA94 <> GDA2020 conversion

def gda_conv(source_datum, lat, lon, h):
    '''

    Parameters
    ----------
    source_datum : str
        CRS of base station used for PPK geotagging.
    lat : float
        latitude of point at oiriginal CRS.
    lon : float
        longitude of point at oiriginal CRS.
    h : float
        ellipsoidal height of point at oiriginal CRS.

    Returns
    -------
    float
        latitude, longitude, and ellipsoidal height of point at new CRS.

    '''
    
    x, y, z = geo_to_cart(lat, lon, h)
    x1, y1, z1 = cart_to_cart(source_datum, x, y, z)
    
    return cart_to_geo(x1, y1, z1)

# Interpolation of Geoid Undulation

def interpolate_raster(file, lat, lon):
    '''
    
    Parameters
    ----------
    file : raster object
        geoid used.
    lat : float
        latitude of point.
    lon : float
        longitude of point.

    Returns
    -------
    interp_val: float
        interpolated value of geoid at point.

    '''
    f = gdal.Open(file)
    band = f.GetRasterBand(1)
    
    # Get Raster Information
    transform = f.GetGeoTransform()
    res = transform[1]
    
    # Define point position as row and column in raster
    column = (lon - transform[0]) / transform[1]
    row = (lat - transform[3]) / transform[5]
    
    # Create a 5 x 5 grid of surrounding the point
    surround_data = (band.ReadAsArray(np.floor(column-2), np.floor(row-2), 5, 5))
    lon_c = transform[0] + np.floor(column) * res
    lat_c = transform[3] - np.floor(row) * res
    
    # Extract geoid undulation values of the 5 x 5 grid
    count = -1
    pos = np.zeros((25,2))
    surround_data_v = np.zeros((25,1))
    
    for k in range(-2,3):
        for j in range(-2,3):
            count += 1
            pos[count] = (lon_c+j*res, lat_c-k*res)
            surround_data_v[count] = surround_data[k+2,j+2]
    
    # Do a cubic interpolation of surrounding data and extract value at point
    interp_val = griddata(pos, surround_data_v, (lon, lat), method='cubic')

    return interp_val[0]

def main():   
    # Application Formatting
    
    st.set_page_config(layout="wide")
    
    st.title('Ellipsoidal to Orthometric Heights (Australia)')
    
    st.sidebar.image('./logo.png', width = 260)
    st.sidebar.markdown('#')
    st.sidebar.write('The application uses the binary files provided by Geoscience Australia to look up the geoid height at a particular location and to then compute the orthometric height.')
    st.sidebar.write('The selection of a geoid model automatically adjusts the horizontal coordinate system to either GDA 94 (AusGeoid09) or GDA 2020 (AusGeoid2020).')
    st.sidebar.write('If you have any questions regarding the application, please contact us at support@wingtra.com.')
    st.sidebar.markdown('#')
    st.sidebar.info('This is a prototype application. Wingtra AG does not guarantee correct functionality. Use with discretion.')
    
    # Upload button for CSVs
    
    uploaded_csvs = st.file_uploader('Please Select Geotags CSV.', accept_multiple_files=True)
    uploaded = False
    
    for uploaded_csv in uploaded_csvs: 
        if uploaded_csv is not None:
            uploaded = True
    
    # Checking if upload of all CSVs is successful
    
    required_columns = ['# image name',
                        'latitude [decimal degrees]',
                        'longitude [decimal degrees]',
                        'altitude [meter]',
                        'accuracy horizontal [meter]',
                        'accuracy vertical [meter]']
    
    if uploaded:
        dfs = []
        filenames = []
        df_dict = {}
        
        for ctr, uploaded_csv in enumerate(uploaded_csvs):
            df = pd.read_csv(uploaded_csv, index_col=False)       
            dfs.append(df)
            df_dict[uploaded_csv.name] = ctr
            filenames.append(uploaded_csv.name)
            
            lat = 'latitude [decimal degrees]'
            lon = 'longitude [decimal degrees]'
            height = 'altitude [meter]'
            
            # Check if CSV is in the correct format
            
            format_check = True
            for column in required_columns:
                if column not in list(df.columns):
                    msg = f'{column} is not in {uploaded_csv.name}.'
                    st.text(msg)
                    format_check = False
            
            if not format_check:
                msg = f'{uploaded_csv.name} is not in the correct format. Delete or reupload to proceed.'
                st.error(msg)
                st.stop()

            # Check if locations are within the United States
            
            url = 'http://api.geonames.org/countryCode?lat='
            geo_request = url + f'{df[lat][0]}&lng={df[lon][0]}&type=json&username=irwinamago'
            country = requests.get(geo_request).json()['countryName']
            
            if country != 'Australia':
                msg = f'Locations in {uploaded_csv.name} are outside Australia. Please remove to proceed.'
                st.error(msg)
                st.stop()
        
        st.success('All CSVs checked and uploaded successfully.')
        
        map_options = filenames.copy()
        map_options.insert(0, '<select>')
        option = st.selectbox('Select geotags CSV to visualize', map_options)
        
        # Option to visualize any of the CSVs
        
        if option != '<select>':
            points_df = pd.concat([dfs[df_dict[option]][lat], dfs[df_dict[option]][lon]], axis=1, keys=['lat','lon'])
            
            st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/satellite-streets-v11',
            initial_view_state=pdk.ViewState(
                latitude=points_df['lat'].mean(),
                longitude=points_df['lon'].mean(),
                zoom=14,
                pitch=0,
             ),
             layers=[
                 pdk.Layer(
                     'ScatterplotLayer',
                     data=points_df,
                     get_position='[lon, lat]',
                     get_color='[70, 130, 180, 200]',
                     get_radius=20,
                 ),
                 ],
             ))
        
        # Base Datum Selection
        datum_dict = {'GDA94': 1, 'GDA2020': 2}
        datum_select = st.selectbox('Please Choose CRS of Base Location Used in PPK Geotagging', 
                                    ('<select>', 'GDA94', 'GDA2020'))    
        if datum_select != '<select>':
            st.write('You selected:', datum_select)
        
        # Geoid Selection
        geoid_dict = {'AusGeoid09': 1, 'AusGeoid2020': 2}
        geoid_select = st.selectbox('Please Choose Desired Geoid', ('<select>', 'AusGeoid09', 'AusGeoid2020'))
        
        if geoid_select != '<select>':
            st.write('You selected:', geoid_select)
        
        if uploaded and datum_select != '<select>' and geoid_select != '<select>':
            if st.button('CONVERT HEIGHTS'):
                aws_server = '/vsicurl/https://geoid.s3-ap-southeast-2.amazonaws.com/'
                geoid09 = aws_server + 'AUSGeoid/AUSGeoid09_V1.01.tif'
                geoid20 = aws_server + 'AUSGeoid/AUSGeoid2020_RELEASEV20170908.tif'
                                   
                for df in dfs:
                    
                    # Resolve base CRS and geoid conflict
                    if datum_dict[datum_select] != geoid_dict[geoid_select]:
                        lat_new = []
                        lon_new = []
                        h_new = []
                                
                        for la, lo, h in zip(df[lat], df[lon], df[height]):
                            lat_conv, lon_conv, h_conv = gda_conv(datum_select, la, lo, h)
                            lat_new.append(lat_conv)
                            lon_new.append(lon_conv)
                            h_new.append(h_conv)
                        
                        df[lat] = lat_new
                        df[lon] = lon_new
                        df[height] = h_new
                    
                    # Height Conversion
                    if geoid_select == 'AusGeoid09':
                        ortho = []
                        
                        
                        for la, lo, h in zip(df[lat], df[lon], df[height]):
                            N = interpolate_raster(geoid09, la, lo)
                            ortho.append(h - N)                 
    
                        df[height] = ortho
                        df.rename(columns={lat: 'latitude GDA94 [decimal degrees]',
                                           lon: 'longitude GDA94 [decimal degrees]',
                                           height: 'orthometric height AusGeoid09 [meters]'}, inplace=True)
                    
                    else:
                        ortho = []
                                                                                                          
                        for la, lo, h in zip(df[lat], df[lon], df[height]):
                            N = interpolate_raster(geoid20, la, lo)
                            ortho.append(h - N)
               
                        df[height] = ortho            
                        df.rename(columns={lat: 'latitude GDA20 [decimal degrees]',
                                           lon: 'longitude GDA20 [decimal degrees]',
                                           height: 'orthometric height AusGeoid20 [meters]'}, inplace=True)
        
                st.success('Height conversion finished. Click button below to download new CSV.')
        
                # Create the zip file, convert the dataframes to CSV, and save inside the zip
                
                if len(dfs) == 1:
                    csv = dfs[0].to_csv(index=False).encode('utf-8')
                    filename = filenames[0].split('.')[0] + '_orthometric.csv'
    
                    st.download_button(
                         label="Download Converted Geotags CSV",
                         data=csv,
                         file_name=filename,
                         mime='text/csv',
                     )
                    
                else:                
                    with zipfile.ZipFile('Converted_CSV.zip', 'w') as csv_zip:
                        for ctr, df in enumerate(dfs):
                            csv_zip.writestr(filenames[ctr].split('.')[0] + '_orthometric.csv', df.to_csv(index=False).encode('utf-8'))   
                    
                    # Download button for the zip file
                    
                    fp = open('Converted_CSV.zip', 'rb')
                    st.download_button(
                        label="Download Converted Geotags CSV",
                        data=fp,
                        file_name='Converted_CSV.zip',
                        mime='application/zip',
                )
        st.stop()
    else:
        st.stop()

if __name__ == "__main__":
    main()
