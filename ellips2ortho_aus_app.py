import math
import numpy
import pandas as pd
import pydeck as pdk
import rasterio.sample
import requests
import streamlit as st
import zipfile

st.set_page_config(layout="wide")

st.title('Ellipsoidal to Orthometric Heights (Australia)')

st.sidebar.image('./logo.png', width = 260)
st.sidebar.markdown('#')
st.sidebar.write('The application uses the binary files provided by Geoscience Australia to look up the geoid height at a particular location and to then compute the orthometric height.')
st.sidebar.write('The selection of a geoid model automatically adjusts the horizontal coordinate system to either GDA 94 (AusGeoid09) or GDA 2020 (AusGeoid2020).')
st.sidebar.write('If you have any questions regarding the application, please contact us at support@wingtra.com.')
st.sidebar.markdown('#')
st.sidebar.info('This is a prototype application. Wingtra AG does not guarantee correct functionality. Use with discretion.')

def geo_to_cart(lat, lon, h):
    lat_rad = (lat/180)*math.pi
    lon_rad = (lon/180)*math.pi
    
    a = 6378137.00
    f = 1/298.257222101
    
    e2 = 2*f - f*f
    v = a/(math.sqrt(1 - e2*math.sin(lat_rad)*math.sin(lat_rad)))

    x = (h + v)*math.cos(lat_rad)*math.cos(lon_rad)
    y = (h + v)*math.cos(lat_rad)*math.sin(lon_rad)
    z = ((1 - e2)*v + h)*math.sin(lat_rad)
    
    return x,y,z

def cart_to_cart(x0, y0, z0):
    Tx = 0.06155
    Ty = -0.01087
    Tz = -0.04019
    Sc = -0.009994
    Rx = -0.0394924
    Ry = -0.0327221
    Rz = -0.0328979
    
    RxRad = ((Rx/3600)/180)*math.pi
    RyRad = ((Ry/3600)/180)*math.pi
    RzRad = ((Rz/3600)/180)*math.pi
    Scale = 1 + (Sc/1000000)
    
    T = numpy.zeros(shape=(3,1))
    T[0][0] = Tx
    T[1][0] = Ty
    T[2][0] = Tz
    
    R = numpy.zeros(shape=(3,3))
    R[0][0] = math.cos(RyRad)*math.cos(RzRad)
    R[0][1] = math.cos(RyRad)*math.sin(RzRad)
    R[0][2] = -math.sin(RyRad)
    
    R[1][0] = math.sin(RxRad)*math.sin(RyRad)*math.cos(RzRad) - math.cos(RxRad)*math.sin(RzRad) 
    R[1][1] = math.sin(RxRad)*math.sin(RyRad)*math.sin(RzRad) + math.cos(RxRad)*math.cos(RzRad) 
    R[1][2] = math.sin(RxRad)*math.cos(RyRad)

    R[2][0] = math.cos(RxRad)*math.sin(RyRad)*math.cos(RzRad) + math.sin(RxRad)*math.sin(RzRad) 
    R[2][1] = math.cos(RxRad)*math.sin(RyRad)*math.sin(RzRad) - math.sin(RxRad)*math.cos(RzRad) 
    R[2][2] = math.cos(RxRad)*math.cos(RyRad)
    
    Xold = numpy.zeros(shape=(3,1))
    Xold[0][0] = x0
    Xold[1][0] = y0
    Xold[2][0] = z0

    Xnew = T + Scale*numpy.matmul(R, Xold)
    
    return Xnew[0][0], Xnew[1][0], Xnew[2][0]

def cart_to_geo(x1, y1, z1):
    a = 6378137.00
    f = 1/298.257222101
    
    e2 = 2*f - f*f
    p = math.sqrt(x1*x1 + y1*y1)
    r = math.sqrt(p*p + z1*z1)
    u = math.atan((z1/p)*((1-f) + (e2*a)/r))
    
    lat_top = z1*(1-f) + e2*a*math.sin(u)*math.sin(u)*math.sin(u)
    lat_bot = (1-f)*(p - e2*a*math.cos(u)*math.cos(u)*math.cos(u))
    
    lon_rad = math.atan(y1/x1)
    if lon_rad < 0:
        lon = 180*(math.pi + lon_rad)/math.pi
    else:
        lon = 180*lon_rad/math.pi
    
    lat_rad = math.atan(lat_top/lat_bot)
    lat = 180*lat_rad/math.pi
    
    h = p*math.cos(lat_rad) + z1*math.sin(lat_rad) - a*math.sqrt(1 - e2*math.sin(lat_rad)*math.sin(lat_rad))

    return lat, lon, h

def gda94_to_gda2020(lat, lon, h):
    x, y, z = geo_to_cart(lat, lon, h)
    x1, y1, z1 = cart_to_cart(x, y, z)
    
    return cart_to_geo(x1, y1, z1)

# Upload button for CSVs

uploaded_csvs = st.file_uploader('Please Select Geotags CSV.', accept_multiple_files=True)
uploaded = False

for uploaded_csv in uploaded_csvs: 
    if uploaded_csv is not None:
        uploaded = True
    else:
        uplaoded = False

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
    ctr = 0
    for uploaded_csv in uploaded_csvs:
        df = pd.read_csv(uploaded_csv, index_col=False)       
        dfs.append(df)
        df_dict[uploaded_csv.name] = ctr
        filenames.append(uploaded_csv.name)
        
        lat = 'latitude [decimal degrees]'
        lon = 'longitude [decimal degrees]'
        height = 'altitude [meter]'
        
        ctr += 1
        
        # Check if locations are within the United States
        
        url = 'http://api.geonames.org/countryCode?lat='
        geo_request = url + str(df[lat][0]) + '&lng=' + str(df[lon][0]) + '&type=json&username=irwinamago'
        country = requests.get(geo_request).json()['countryName']
        
        if country != 'Australia':
            msg = 'Locations in ' + uploaded_csv.name + ' are outside Australia. Please remove to proceed.'
            st.error(msg)
            st.stop()

        # Check if CSV is in the correct format
        
        format_check = True
        for column in required_columns:
            if column not in list(df.columns):
                st.text(column + ' is not in ' + uploaded_csv.name + '.')
                format_check = False
        
        if not format_check:
            msg = uploaded_csv.name + ' is not in the correct format. Delete or reupload to proceed.'
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
    
    # Geoid Selection
    
    geoid_select = st.selectbox('Please Choose Desired Geoid', ('<select>', 'AusGeoid09', 'AusGeoid2020'))
    if geoid_select != '<select>':
        st.write('You selected:', geoid_select)
    
    if uploaded and geoid_select != '<select>':
        if st.button('CONVERT HEIGHTS'):
            aws_server = '/vsicurl/https://geoid.s3-ap-southeast-2.amazonaws.com/'
            geoid09_file = aws_server + 'AUSGeoid/AUSGeoid09_V1.01.tif'
            geoid20_file = aws_server + 'AUSGeoid/AUSGeoid2020_RELEASEV20170908.tif'
            file_ctr = 0
            
            for df in dfs:
                if geoid_select == 'AusGeoid09':
                    ortho = []
                    geoid09 = rasterio.open(geoid09_file, crs='EPSG:4939')
                    points = list(zip(df[lon].tolist(), df[lat].tolist()))
        
                    i = 0
                    for val in geoid09.sample(points):
                        ortho.append(df[height][i] - val[0])
                        i += 1
        
                    df[height] = ortho
                    df.rename(columns={lat: 'latitude GDA94 [decimal degrees]',
                                       lon: 'longitude GDA94 [decimal degrees]',
                                       height: 'orthometric height AusGeoid09 [meters]'}, inplace=True)
        
                else:
                    ortho = []
                    geoid20 = rasterio.open(geoid20_file, crs='EPSG:7843')
        
                    # Convert Coordinates
                    lat_gda20 = []
                    lon_gda20 = []
                    h_gda20 = []
        
                    for x in range(len(df[lat])):
                        la, lo, h = gda94_to_gda2020(df[lat][x], df[lon][x], df[height][x])
                        lat_gda20.append(la)
                        lon_gda20.append(lo)
                        h_gda20.append(h)
        
                    points = list(zip(lon_gda20,lat_gda20))
        
                    i = 0
                    for val in geoid20.sample(points):
                        ortho.append(h_gda20[i] - val[0])
                        i += 1
        
                    df[lat] = lat_gda20
                    df[lon] = lon_gda20
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
                    file_ctr = 0
                    for df in dfs:
                        csv_zip.writestr(filenames[file_ctr].split('.')[0] + '_orthometric.csv', df.to_csv(index=False).encode('utf-8'))
                        file_ctr += 1   
                
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
