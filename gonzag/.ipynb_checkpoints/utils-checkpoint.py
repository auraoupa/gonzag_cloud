#!/usr/bin/env python3
# -*- Mode: Python; coding: utf-8; indent-tabs-mode: nil; tab-width: 4 -*-
#
#   ///// https://github.com/brodeau/gonzag \\\\\
#
#       L. Brodeau, 2021
#
############################################################################

from sys import exit
from math import radians, cos, sin, asin, sqrt, pi, tan, log, atan2, copysign
import numpy as nmp
from .config import ldebug, R_eq, R_pl, deg2km

def MsgExit( cmsg ):
    print('\n ERROR: '+cmsg+' !\n')
    exit(0)


def degE_to_degWE( X ):
    '''
    # From longitude in 0 -- 360 frame to -180 -- +180 frame...
    '''
    if nmp.shape( X ) == ():
        # X is a scalar
        return     copysign(1., 180.-X)*        min(X,     abs(X-360.))
    else:
        # X is an array
        return nmp.copysign(1., 180.-X)*nmp.minimum(X, nmp.abs(X-360.))


def find_j_i_min(x):
    '''
    # Yes, reinventing the wheel here, but it turns out
    # it is faster this way!
    '''
    k = x.argmin()
    nx = x.shape[1]
    return k//nx, k%nx

def find_j_i_max(x):
    '''
    # Yes, reinventing the wheel here, but it turns out
    # it is faster this way!
    '''
    k = x.argmax()
    nx = x.shape[1]
    return k//nx, k%nx



def GridResolution( X ):
    '''
    # X    : the 2D array of the model grid longitude
    '''
    ny = X.shape[0]
    vx = nmp.abs(X[ny//2,1:] - X[ny//2,:-1])
    res = nmp.mean( vx[nmp.where(vx < 120.)] )
    if ldebug: print(' *** [GridResolution()] Based on the longitude array, the model resolution ~= ', res.values, ' degrees \n')
    return res.values


def IsEastWestPeriodic( X ):
    '''
    # X    : the 2D array of the model grid longitude [0:360]
    #  RETURNS: iper: -1 => no E-W periodicity ; iper>=0 => E-W periodicity with iper overlaping points!
    '''
    ny = X.shape[0]
    jj = ny//2 ; # we test at the center...
    iper = -1
    dx = X[jj,1] - X[jj,0]
    lon_last_p1 = (X[jj,-1]+dx)%360.
    for it in range(5):
        if lon_last_p1 == X[jj,it]%360.:
            iper = it
            break
    return iper



def SearchBoxSize( res_mod, width_box ):
    '''
    # Returns half of the width, in number of grid points, of the small zoom-box
    # of the source (model) domain in which NearestPoint() will initially attempt
    # to locate  the nearest point, before falling back on the whole source (model)
    # domain if unsuccessful.
    #  => the smaller the faster the search...
    #
    # * res_mod:   horizontal resolution of the model data, in km
    # * width_box: width of the zoom-box in km
    #
    # TODO: shoud take into account the speed of the satellite
    '''
    return int(0.5*width_box/res_mod)



#def IsGlobalLongitudeWise( X, resd=1. ):
#    '''
#    # X    : the 2D array of the model grid longitude
#    # resd : rough order of magnitude of the resolutio of the model grid in degrees
#    # RETURNS: boolean + lon_min and lon_max in the [-180:+180] frame
#    '''
#    X = nmp.mod(X, 360.) ; # no concern, it should already have been done earlier anyway...
#    print( 'X =', X)
#    print( 'X =', degE_to_degWE(X))
#    xmin1 = nmp.amin(degE_to_degWE(X)) ; # in [-180:+180] frame...
#    xmax1 = nmp.amax(degE_to_degWE(X)) ; #     "      "
#    xmin2 = nmp.amin(X) ; # in [0:360] frame...
#    xmax2 = nmp.amax(X) ; #     "     "
#    print(' xmin2, xmax2 =', xmin2, xmax2 )
#    print(' xmin1, xmax1 =', xmin1, xmax1 )
#    l1 = ( xmin1<0. and xmin1>5.*resd-180 ) or ( xmax1>0. and xmax1<180.-5.*resd )
#    l2 = ( xmin2<1.5*resd and xmax2>360.-1.5*resd )
#    #print('l1, l2 =', l1, l2)
#    if not l1:
#        if l2 :
#            lglobal = True
#            print(' *** From model longitude => looks like global setup (longitude-wise)', xmin1, xmax1,'\n')
#        else:
#            MsgExit('cannot find if regional or global source domain...')
##    else:
#        print(' *** From model longitude => looks like regional setup (longitude-wise)', xmin1, xmax1)
#        lglobal = False
#        print('     => will disregard alongtrack points with lon < '+str(xmin1)+' and lon > '+str(xmax1),'\n')
#    #
#    return lglobal, xmin1, xmax1


def IsGlobalLongitudeWise( X, mask , resd=1. ):
    '''
    # LIMITATION: longitude has to increase in the x-direction (second dimension) of X (i increases => lon increases)
    # X    : the 2D array of the model grid longitude
    # resd : rough order of magnitude of the resolutio of the model grid in degrees
    # RETURNS: boolean, boolean, lon_min, lon_max
    '''
    #
    nx   = X.shape[1]
    X    = nmp.mod(X, 360.) ; # no concern, it should already have been done earlier anyway...
    Xm   = nmp.ma.masked_where( mask==0, X    )
    xmin = nmp.amin(Xm) ; # in [0:360] frame...
    xmax = nmp.amax(Xm) ; #     "     "
    imin = nmp.argmin(Xm)
    imax = nmp.argmax(Xm)
    #
    xb = degE_to_degWE(Xm)
    xminB = nmp.amin(xb) ; # in [-180:+180] frame...
    xmaxB = nmp.amax(xb) ; #     "      "
    #
    l360    = True   ; # we'll be in the [0:360] frame...
    lglobal = False
    #
    if (xmin<1.5*resd and xmax>360.-1.5*resd) and (imax > imin):
        # Global longitude domain
        lglobal = True
        xmin=0. ; xmax=360.
        #
    elif (xminB%360. > xmaxB%360.) and (imax < imin):
        # (xminB%360. > xmaxB%360.) is True in the 2 singular cases: domain icludes Greenwhich Meridian or -180:180 transition...
        l360 = False
        xmin = xminB
        xmax = xmaxB
    #
    del xb
    return lglobal, l360, xmin, xmax

def GetTimeOverlap( dataset_sat, dataset_mod, timevar_sat, timevar_mod ):
    '''
    # Get time overlap from model segment
    # Get satellite dates corresponding
    # (not the same year)
    '''
    import numpy as nmp
    import pandas as pd
    from .io import GetTimeInfo
    #
    nts, range_sat = GetTimeInfo( dataset_sat, timevar_sat )
    ntm, range_mod = GetTimeInfo( dataset_mod, timevar_mod )
    #
    (zt1_s,zt2_s) = range_sat
    (zt1_m,zt2_m) = range_mod
    if ldebug: print('\n *** Earliest/latest dates:\n   => for satellite data:',zt1_s,zt2_s,'\n   => for model     data:',zt1_m,zt2_m,'\n')
    doy1_m=pd.to_datetime(zt1_m).dayofyear
    doy2_m=pd.to_datetime(zt2_m).dayofyear
    year_s=pd.to_datetime(zt1_s).year
    date1=nmp.asarray(year_s+1, dtype='datetime64[Y]')-1970+nmp.asarray(doy1_m, dtype='timedelta64[D]')-1
    date2=nmp.asarray(year_s+1, dtype='datetime64[Y]')-1970+nmp.asarray(doy2_m, dtype='timedelta64[D]')-1

    return (date1, date2), (nts, ntm)

def scan_idx( rvt, rt1, rt2 ):
    '''
    # Finding indices when we can start and stop when scanning the track file:
    # * vt: vector containing dates of Satellite data
    # * rt1, rt2: the 2 dates of interest (first and last) (from model)
    # RETURNS: the two corresponding position indices
    '''
    import numpy as nmp
    from datetime import datetime as dtm
    
    idx1=nmp.where(rvt>rt1)
    idx2=nmp.where(rvt<rt2)

    kt1=idx1[0].min()
    kt2=idx2[0].max()

    return kt1, kt2


def GridAngle( xlat, xlon ):
    ''' To be used with a NEMO ORCA-type of grid, to get an idea of the local distortion (rotation)
    #   of the grid
    # Returns local distortion of the grid in degrees [-180,180]
    '''
    to_rad = pi/180.
    pio4   = pi/4.
    (Ny,Nx) = nmp.shape(xlat)
    #
    angle = nmp.zeros((Ny,Nx))
    for ji in range(Nx):
        for jj in range(1,Ny-1):

            if abs( xlon[jj+1,ji]%360. - xlon[jj-1,ji]% 360. ) < 1.e-8:
                sint = 0.
                cost = 1.
                #
            else:
                zt0 = tan( pio4 - to_rad*xlat[jj,ji]/2. )
                # North pole direction & modulous (at t-point)
                zxnpt = 0. - 2. * cos( to_rad*xlon[jj,ji] ) * zt0
                zynpt = 0. - 2. * sin( to_rad*xlon[jj,ji] ) * zt0
                znnpt            = zxnpt*zxnpt + zynpt*zynpt
                #
                # j-direction: v-point segment direction (around t-point)
                zt1 = tan( pio4 - to_rad*xlat[jj+1,ji]/2. )
                zt2 = tan( pio4 - to_rad*xlat[jj-1,ji]/2. )
                zxvvt =  2.*( cos(to_rad*xlon[jj+1,ji])*zt1 - cos(to_rad*xlon[jj-1,ji])*zt2 )
                zyvvt =  2.*( sin(to_rad*xlon[jj+1,ji])*zt1 - sin(to_rad*xlon[jj-1,ji])*zt2 )
                znvvt = sqrt( znnpt * ( zxvvt*zxvvt + zyvvt*zyvvt )  )
                znvvt = max( znvvt, 1.e-12 )
                #
                sint = ( zxnpt*zyvvt - zynpt*zxvvt ) / znvvt
                cost = ( zxnpt*zxvvt + zynpt*zyvvt ) / znvvt
                angle[jj,ji] = atan2( sint, cost ) * 180./pi
        #
    return angle[:,:]




def RadiusEarth( lat ):
    '''
    Returns the radius of Earth in km as a function of latitude provided in degree N.
    '''
    latr = radians(lat) #converting into radians
    c    = (R_eq**2*cos(latr))**2
    d    = (R_pl**2*sin(latr))**2
    e    = (R_eq*cos(latr))**2
    f    = (R_pl*sin(latr))**2
    R    = sqrt((c+d)/(e+f))
    #print('\nRadius of Earth at '+str(round(lat,2))+'N = '+str(round(R,2))+' km\n')
    return R


def haversine_sclr( lat1, lon1, lat2, lon2 ):
    '''
    Returns the distance in km at the surface of the earth
    between two GPS points (degreesN, degreesE)
    '''
    R = RadiusEarth( 0.5*(lat1 + lat2) )
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))
    return R * c


def Haversine( plat, plon, xlat, xlon ):
    '''
    # ! VECTOR VERSION !
    # Returns the distance in km at the surface of the earth
    # between two GPS points (degreesN, degreesE)
    # (plat,plon)  : a point
    # xlat, xlon : 2D arrays
    #
    # Here we do not need accuracy on Earth radius, since the call
    # to this function is suposely made to find nearest point
    '''
    to_rad = pi/180.
    #
    #R = RadiusEarth( plat ) ; # it's the target location that matters...
    R = 6360.
    #
    a1 = nmp.sin( 0.5 * ((xlat - plat)*to_rad) )
    a2 = nmp.sin( 0.5 * ((xlon - plon)*to_rad) )
    a3 = nmp.cos( xlat*to_rad ) * cos(plat*to_rad)
    #
    return 2.*R*nmp.arcsin( nmp.sqrt( a1*a1 + a3 * a2*a2 ) )


def PlotMesh( pcoor_trg, Ys, Xs, isrc_msh, wghts, fig_name='mesh.png' ):
    '''
    isrc_msh: 2D integer array of shape (4,2)
    wghts:    1D real array of shape (4,)
    '''
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    #
    (yT,xT)                             = pcoor_trg
    [ [j1,i1],[j2,i2],[j3,i3],[j4,i4] ] = isrc_msh[:,:]
    [ wb1, wb2, wb3, wb4 ]              = wghts[:]
    #
    fig = plt.figure(num = 1, figsize=[7,5], facecolor='w', edgecolor='k')
    ax1 = plt.axes([0.09, 0.07, 0.6, 0.9])
    plt.plot( [  yT ] , [ xT ]  , marker='o', ms=15, color='k', label='target point' ) ; # target point !
    plt.plot( [ Xs[j1,i1] ], [ Ys[j1,i1] ], marker='o', ms=10, label='P1: w='+str(round(wb1,3)) ) ; # nearest point !
    plt.plot( [ Xs[j2,i2] ], [ Ys[j2,i2] ], marker='o', ms=10, label='P2: w='+str(round(wb2,3)) ) ; #
    plt.plot( [ Xs[j3,i3] ], [ Ys[j3,i3] ], marker='o', ms=10, label='P3: w='+str(round(wb3,3)) ) ; # target point !
    plt.plot( [ Xs[j4,i4] ], [ Ys[j4,i4] ], marker='o', ms=10, label='P4: w='+str(round(wb4,3)) ) ; # target point !
    ax1.legend(loc='center left', bbox_to_anchor=(1.07, 0.5), fancybox=True)
    plt.savefig(fig_name, dpi=100, transparent=False)
    plt.close(1)

def CuttingTracksDerive( slat ):
    '''
    Returns a list of indexes couples ([i1,i2], [i3,i4], ...) separating the tracks of the satellite data
    between ascending and descending latitude
    slat: latitude vector of every points of satellite data
    '''
    it1=0
    it2=0
    index_tracks=[]
    slat_shift=nmp.roll(slat,-1)
    deriv_lat=slat_shift-slat
    while it2 < len(slat):
        it1=it2
        ind=nmp.where(nmp.sign(deriv_lat[it1:])==-1*nmp.sign(deriv_lat[it1]))
        if len(ind[0])>0:
            it=nmp.min(ind)
            it2=it1+it
            index_tracks.append([it1,it2])
        else:
            break
    return index_tracks

def RemoveNaNSSH( index_tracks, sssh):
    '''
    Cleans up a list of indexes couples ([i1,i2], [i3,i4], ...) separating the tracks of the satellite data
    by removing points with NaN as ssh
    index_tracks: list of indexes couples defining the tracks for satellite vector
    sssh: sea surface height of satellite data
    '''
    k=0
    while k < len(index_tracks):
        ssh_track=sssh[index_tracks[k][0]:index_tracks[k][1]]
        if len(nmp.where(nmp.isnan(ssh_track)==1)) > 0:
            del index_tracks[k]
            break
        else:
               k+=1
    return index_tracks

def RemoveIsolatedPointsinTracks( index_tracks, slat, slon):
    '''
    Cleans up a list of indexes couples ([i1,i2], [i3,i4], ...) separating the tracks of the satellite data
    by removing singleton (points distant of more than 2deg in latitude or longitude)
    index_tracks: list of indexes couples defining the tracks for satellite vector
    slat: latitude vector of every points of satellite data
    slon: longitude vector of every points of satellite data
    '''
    k=0
    while k < len(index_tracks):
        lat_sat_track=slat[index_tracks[k][0]:index_tracks[k][1]]
        lon_sat_track=slon[index_tracks[k][0]:index_tracks[k][1]]
        if len(lat_sat_track) == 0:
            del index_tracks[k]
            break
        else:
            lat_sat_track_shift=nmp.roll(lat_sat_track,-1)
            lat_sat_track_reverse=nmp.roll(lat_sat_track,1)
            diff_lat_sat_track=lat_sat_track-lat_sat_track_shift
            diff_lat_sat_track_reverse=lat_sat_track-lat_sat_track_reverse
            lon_sat_track_shift=nmp.roll(lon_sat_track,-1)
            lon_sat_track_reverse=nmp.roll(lon_sat_track,1)
            diff_lon_sat_track=lon_sat_track-lon_sat_track_shift
            diff_lon_sat_track_reverse=lon_sat_track-lon_sat_track_reverse
            index_track=nmp.arange(index_tracks[k][0],index_tracks[k][1])
            break_points=0
            for t in nmp.arange(len(index_track)-1): #last diff is the diff between first and last so its ok to be over 2
                if nmp.abs(diff_lat_sat_track[t]) > 2 or nmp.abs(diff_lon_sat_track[t]) > 2:
                    index_tracks.append([index_tracks[k][0],index_track[t]])
                    index_tracks.append([index_track[t+1],index_tracks[k][1]])
                    del index_tracks[k]
                    break_points=1
                    break
            if nmp.abs(diff_lat_sat_track_reverse[-1]) > 2 or nmp.abs(diff_lon_sat_track_reverse[-1]) > 2: #to check if the last point is far away from the rest of the segment
                index_tracks.append([index_tracks[k][0],index_tracks[k][1]-1])
                index_tracks.append([index_tracks[k][1],index_tracks[k][1]])
                del index_tracks[k]
                break_points=1
                break
            if break_points == 0:
                k+=1
    return index_tracks

def RemoveTracks( index_tracks ):
    '''
    Cleans up a list of indexes couples ([i1,i2], [i3,i4], ...) separating the tracks of the satellite data
    by removing tracks with size lower than 1
    index_tracks: list of indexes couples defining the tracks for satellite vector
    '''
    k=0
    while k < len(index_tracks):
        len_track = index_tracks[k][1]-index_tracks[k][0]
        if len_track <= 1:
            del index_tracks[k]
        else:
            k=k+1
    return index_tracks

def SeparateTracks( slat, slon, ssh ):
    '''
    Returns a list of indexes couples ([i1,i2], [i3,i4], ...) separating the tracks of the satellite data
    in consistent pieces
    slat: latitude vector of every points of satellite data
    slon: longitude vector of every points of satellite data
    '''
    index_tracks_asc_desc = CuttingTracksDerive( slat )
    index_tracks_clean_ssh = RemoveNaNSSH( index_tracks_asc_desc, ssh)
    index_tracks_clean = RemoveIsolatedPointsinTracks( index_tracks_clean_ssh, slat, slon)
    index_tracks_nosingleton = RemoveTracks( index_tracks_clean )
    index_tracks_nosing_clean = RemoveIsolatedPointsinTracks( index_tracks_nosingleton, slat, slon)
    return index_tracks_nosing_clean

# C L A S S E S



class ModGrid:
    '''
    # Will provide: size=nt, shape=(ny,nx), time[:], lat[:,:], lon[:,:] of Model data...
    # mask
    # domain_bounds (= [ lat_min, lon_min, lat_max, lon_max ])
    '''
    def __init__( self, dataset, period, varlon, varlat, vartime, gridset, varlsm, distorded_grid=False ):
        '''
        # * dataset: DataArray containing model data
        # * varlon, varlat: name of the latitude and longitude variable
        # * gridset, varlsm: file and variable to get land-sea mask...
        '''
        from .io import GetTimeVector, GetModelCoor, GetModelLSM, Save2Dfield


        self.file = dataset

        rvt = GetTimeVector( dataset, vartime, lquiet=True )
        (rtu1,rtu2)=nmp.array(period, dtype='datetime64')
        jt1, jt2 = scan_idx( rvt, rtu1, rtu2 )

        nt = jt2 - jt1 + 1

        self.jt1   = jt1
        self.jt2   = jt2

        vtime = GetTimeVector( dataset, vartime, kt1=jt1, kt2=jt2 )
        
        self.size = nt
        self.time = vtime


        zlat =          GetModelCoor( dataset, 'latitude', varlat )
        zlon = nmp.mod( GetModelCoor( dataset, 'longitude', varlon ) , 360. )
        if len(zlat.shape)==1 and len(zlon.shape)==1:
            # Must build the 2D version:
            print(' *** Model latitude and longitude arrays are 1D => building the 2D version!')
            ny = len(zlat) ;  nx = len(zlon)
            xlat = nmp.zeros((ny,nx))
            xlon = nmp.zeros((ny,nx))
            for jx in range(nx): xlat[:,jx] = zlat[:]
            for jy in range(ny): xlon[jy,:] = zlon[:]
            self.lat = xlat
            self.lon = xlon
            del xlat, xlon
        else:
            # Already 2D:
            if zlat.shape != zlon.shape: MsgExit('[SatTrack()] => lat and lon disagree in shape')
            self.lat = zlat
            self.lon = zlon
        del zlat, zlon

        self.shape = self.lat.shape

        # Land-sea mask

        self.mask = GetModelLSM( gridset, varlsm ) ;
        if self.mask.shape != self.shape: MsgExit('model land-sea mask has a wrong shape => '+str(self.mask.shape))
#        if ldebug: Save2Dfield( 'mask_model.nc', self.mask, name='mask' )

        # Horizontal resolution
        self.HResDeg = GridResolution( self.lon )
        if self.HResDeg>5. or self.HResDeg<0.001: MsgExit('Model resolution found is surprising, prefer to stop => check "GetModelResolution()" in utils.py')
        self.HResKM  = self.HResDeg*deg2km

        # Globality and East-West periodicity ?
        self.IsLonGlobal, self.l360, lon_min, lon_max = IsGlobalLongitudeWise( self.lon, mask=self.mask , resd=self.HResDeg)
        if self.IsLonGlobal:
            self.EWPer = IsEastWestPeriodic( self.lon )
        else:
            self.EWPer = -1

        latm = nmp.ma.masked_where( self.mask==0, self.lat    )
        lat_min = nmp.amin(latm)
        lat_max = nmp.amax(latm)

        # Local distortion angle:
        self.IsDistorded = distorded_grid
        if distorded_grid:
            print(' *** Computing angle distortion of the model grid ("-D" option invoked)...')
            self.xangle = GridAngle( self.lat, self.lon )
            if ldebug: Save2Dfield( 'model_grid_disortion.nc', self.xangle, name='angle', mask=self.mask )
        else:
            print(' *** Skipping computation of angle distortion of the model grid! ("-D" option not invoked)...')
            self.xangle = nmp.zeros(self.shape)

        self.domain_bounds = [ lat_min, lon_min, lat_max, lon_max ]

        if not self.l360: self.lon = degE_to_degWE( self.lon )

        # Summary:
        print('\n *** About model gridded (source) domain:')
        print('     * shape = ',self.shape)
        print('     * horizontal resolution: ',self.HResDeg,' degrees or ',self.HResKM,' km')
        print('     * Is this a global domain w.r.t longitude: ', self.IsLonGlobal)
        if self.IsLonGlobal:
            print('       ==> East West periodicity: ', (self.EWPer>=0), ', with an overlap of ',self.EWPer,' points')
        else:
            print('       ==> this is a regional domain')
            if self.l360:
                print('       ==> working in the [0:360] frame...')
            else:
                print('       ==> working in the [-180:180] frame...')
        print('     * lon_min, lon_max = ', round(lon_min,2), round(lon_max,2))
        print('     * lat_min, lat_max = ', round(lat_min,2), round(lat_max,2))
        print('     * should we pay attention to possible STRONG local distorsion in the grid: ', self.IsDistorded)
        print('     * number of time records of interest for the interpolation to come: ', self.size)
        print('       ==> time record dates: '+str(rtu1)+' to '+str(rtu2)+', included\n')









class SatTrack:
    '''
    # Will provide: size, time[:], lat[:], lon[:] of Satellite track
    '''
    def __init__( self, dataset, period, name_time_sat, name_ssh, domain_bounds=[-90.,0. , 90.,360.], l_0_360=True ):
        '''
        # *  dataset: DataArray containing satellite track
        # *  name_ssh : name of the variable containing ssh [string]
        # ** domain_bounds: bound of region we are interested in => [ lat_min, lon_min, lat_max, lon_max ]
        '''
        from .io import GetTimeVector, GetSatCoor, GetSatSSH




        self.file = dataset

        print(' *** [SatTrack()] Analyzing the time vector in dataset ...')


        rvt = GetTimeVector( dataset, name_time_sat, lquiet=True )
        (rtu1,rtu2)=nmp.array(period, dtype='datetime64')
        jt1, jt2 = scan_idx( rvt, rtu1, rtu2 )

        nt = jt2 - jt1 + 1

        self.jt1   = jt1
        self.jt2   = jt2

        vtime = GetTimeVector( dataset, name_time_sat, kt1=jt1, kt2=jt2 )


        vlat  =        GetSatCoor( dataset, 'latitude' , jt1,jt2 )
        vlon  =        GetSatCoor( dataset, 'longitude', jt1,jt2 )

        # Make sure we are in the same convention as model data
        # (model data can be in [-180:180] range if regional domain that crosses Greenwhich meridian...
        if l_0_360:
            vlon = nmp.mod( vlon, 360. )
        else:
            vlon = degE_to_degWE( vlon )

        #print(' lolo: track size before removing points outside of model domain: '+str(len(vtime)))
        [ ymin,xmin , ymax,xmax ] = domain_bounds
        #print('lolo: lat_min, lat_max =', ymin, ymax)
        #print('lolo: lon_min, lon_max =', xmin, xmax)
        keepit = nmp.where( (vlat[:]>=ymin) & (vlat[:]<=ymax) & (vlon[:]>=xmin) & (vlon[:]<=xmax) )
        #print(' lolo: keepit =', keepit )

        self.time  = vtime[keepit]
        self.lat   =  vlat[keepit]
        self.lon   =  vlon[keepit]

        self.size = len(self.time)
        self.keepit = keepit
        
        vssh = GetSatSSH( self.file, name_ssh,  kt1=jt1, kt2=jt2-1)
        self.index_tracks = SeparateTracks( self.lat.values, self.lon.values, vssh )

        del vtime, vlat, vlon, vssh
        #print(' lolo: track size AFTER removing points outside of model domain: '+str(self.size))

        print('\n *** About satellite track (target) domain:')
        print('     * number of time records of interest for the interpolation to come: ', self.size)
        print('       ==> time record indices: '+str(jt1)+' to '+str(jt2)+', included\n')
        print('       separated in '+str(len(self.index_tracks))+' tracks')
