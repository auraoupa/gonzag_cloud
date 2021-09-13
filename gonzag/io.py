#!/usr/bin/env python3
# -*- Mode: Python; coding: utf-8; indent-tabs-mode: nil; tab-width: 4 -*-
#
#   ///// https://github.com/brodeau/gonzag \\\\\
#
#       L. Brodeau, 2021
#
############################################################################

import numpy as nmp
import xarray as xr
from calendar import timegm
from datetime import datetime as dtm
from .config import ldebug, ivrb, rmissval
from .utils  import MsgExit

cabout_nc = 'Created with Gonzag package => https://github.com/brodeau/gonzag'


def GetTimeInfo( dataset, varname ):
    '''
    # Inspect time dimension
    # Get number of time-records + first and last date
    '''
    from datetime import datetime as dtm
    
    if ldebug: print(' *** [GetTimeInfo()] Getting calendar/time info in dataset ...')
    nt = dataset[varname].size
    clndr = dataset[varname]
    if type(clndr.values[0]) == nmp.float32:
        dt1 = dtm.utcfromtimestamp(clndr[0]) ; dt2 = dtm.utcfromtimestamp(clndr[nt-1])
    else:
        dt1 = clndr[0].values ; dt2 = clndr[nt-1].values
    #
    if ldebug: print('   => first and last date: ',dt1.values,'--',dt2.values)
    #
    return nt, (dt1,dt2)




def GetTimeVector( dataset, varname, kt1=0, kt2=0, isubsamp=1, lquiet=False ):
    '''
    # Get the time vector in the dataset
    #
    # INPUT:
    #  * kt1, kt2 : read from index kt1 to kt2 (if these 2 are != 0)
    #  * isubsamp: subsampling !!!
    #  * lquiet: shut the f* up!
    #
    # OUTPUT:
    #  * rvte: datetime vector
    '''
    ltalk = ( ldebug and not lquiet )
    rvt = dataset[varname]
    if type(rvt.values[0]) == nmp.float32:
        clndr=[]
        for k in nmp.arange(len(rvt)):
            clndr.append(nmp.array(dtm.utcfromtimestamp(rvt[k]), dtype='datetime64')) 
        clndr=nmp.array(clndr)
    else:
        clndr=rvt.values
        
    if kt1>0 and kt2>0:
        if kt1>=kt2: MsgExit('mind the indices when calling GetTimeVector()')
        vdate = clndr[kt1:kt2+1:isubsamp]
        cc = 'read...'
    else:
        vdate = clndr[::isubsamp]
        cc = 'in TOTAL!'

    if ivrb>0 and ltalk: print('   => '+str(len(vdate))+' records '+cc+'\n')
    return vdate



def GetModelCoor( dataset, what, ncvar ):
    '''
    #   list_dim = list(id_f.dimensions.keys()) ;  print(" list dim:", list_dim)
    '''
    if   what ==  'latitude': ii = 0
    elif what == 'longitude': ii = 1
    else: MsgExit(' "what" argument of "GetModelCoor()" only supports "latitude" and "longitude"')
    #
    nb_dim = len(dataset[ncvar].dims)
    if   nb_dim==1: xwhat = dataset[ncvar][:]
    elif nb_dim==2: xwhat = dataset[ncvar][:,:]
    elif nb_dim==3: xwhat = dataset[ncvar][0,:,:]
    else: MsgExit('FIX ME! Model '+what+' has a weird number of dimensions')
    if ldebug: print(' *** [GetModelCoor()] Read model '+what+' (variable is "'+ncvar+'", with '+str(nb_dim)+' dimensions!',nmp.shape(xwhat),'\n')
    #
    return xwhat


def GetModelLSM( dataset, what ):
    '''
    # Returns the land-sea mask on the source/moded domain: "1" => ocean point, "0" => land point
    # => 2D array [integer]
    '''
    print('\n *** what we use to define model land-sea mask:\n    => "'+what+'" in dataset \n')
    l_fill_val = (what[:10]=='_FillValue')
    l_nonzero_val = (what[:10]=='_IsNotZero')
    ncvar = what
    if l_fill_val: ncvar = what[11:]
    if l_nonzero_val: ncvar = what[11:]
    #
    ndim = len(dataset[ncvar].dims)
    if l_fill_val:
        # Mask is constructed out of variable and its missing value
        if not ndim in [3,4]: MsgExit(ncvar+' is expected to have 3 or 4 dimensions')
        if ndim==3: xmsk = 1 - nmp.isnan(dataset[ncvar][0,:,:])
        if ndim==4: xmsk = 1 - nmp.isnan(dataset[ncvar][0,0,:,:])
    elif l_nonzero_val:
        # Mask is constructed out of variable and where it is not 0
        if not ndim in [3,4]: MsgExit(ncvar+' is expected to have 3 or 4 dimensions')
        if ndim==3: xmsk = dataset[ncvar][0,:,:].values > 0
        if ndim==4: xmsk = dataset[ncvar][0,0,:,:].values > 0
    else:
        # Mask is read in mask file...
        if   ndim==2: xmsk = dataset[ncvar][:,:]
        elif ndim==3: xmsk = dataset[ncvar][0,:,:]
        elif ndim==4: xmsk = dataset[ncvar][0,0,:,:]
        else: MsgExit('FIX ME! Mask dataset has a weird number of dimensions:'+str(ndims))
    #
    return xmsk.astype(int)


def GetModel2DVar( dataset, ncvar, kt=0 ):
    '''
    #   Fetches the 2D field "ncvar" at time record kt into "dataset"
    '''
    if ldebug: print(' *** [GetModel2DVar()] Reading model "'+ncvar+'" at record kt='+str(kt)+' in dataset')
    nb_dim = len(dataset[ncvar].dims)
    if nb_dim==3:
        x2d = dataset[ncvar][kt,:,:]
    elif nb_dim==4:
        x2d = dataset[ncvar][kt,0,:,:] ; # taking surface field!
    else: MsgExit('FIX ME! Model dataset has a weird number of dimensions: '+str(nb_dim))
    if ldebug: print('')
    return x2d.load()




def GetSatCoor( dataset, what,  kt1=0, kt2=0 ):
    '''
    # Get latitude (what=='latitude') OR longitude (what=='longitude') vector
    # in the netCDF file, (from index kt1 to kt2, if these 2 are != 0!)
    '''
    cv_coor_test = nmp.array([[ 'lat','latitude', 'LATITUDE',  'none' ],
                              [ 'lon','longitude','LONGITUDE', 'none' ]])
    if   what ==  'latitude': ii = 0
    elif what == 'longitude': ii = 1
    else: MsgExit('"what" argument of "GetSatCoor()" only supports "latitude" and "longitude"')
    #
    for ncvar in cv_coor_test[ii,:]:
        if ncvar in dataset.coords: break
    if ncvar == 'none': MsgExit('could not find '+what+' array into satellite file (possible fix: "cv_coor_test" in "GetSatCoor()")')
    #
    if ldebug: print(' *** [GetSatCoor()] reading "'+ncvar+'" in dataset ...')
    nb_dim = len(dataset[ncvar].dims)
    if nb_dim==1:
        if kt1>0 and kt2>0:
            if kt1>=kt2: MsgExit('mind the indices when calling GetSatCoor()')
            vwhat = dataset.variables[ncvar][kt1:kt2+1]
            cc = 'read...'
        else:
            vwhat = dataset.variables[ncvar][:]
            cc = 'in TOTAL!'
    else:
        MsgExit('FIX ME! Satellite '+what+' has a weird number of dimensions (we expect only 1: the time-record!)')
    if ldebug: print('   => '+str(vwhat.size)+' records '+cc+'\n')
    #
    return vwhat


def GetSatSSH( dataset, ncvar,  kt1=-1, kt2=-1, ikeep=[] ):
    '''
    # Get vector time-series of 'ncvar' in file 'ncfile'!
    #  - from index kt1 to kt2, if these 2 are != -1!
    #  - if (ikeep != []) => only retains parts of the data for which indices are provide into ikeep
    #          (ikeep is an array obtained as the result of a "numpy.where()"
    '''
    if ldebug: print(' *** [GetSatSSH()] Reading satellite "'+ncvar+' in dataset')
    if kt1>-1 and kt2>-1:
        if kt1>=kt2: MsgExit('mind the indices when calling GetSatSSH()')
        vssh = dataset[ncvar][kt1:kt2+1]
    else:
        vssh = dataset[ncvar][:]
    if len(ikeep) > 0:
        # Keep specified part of the data
        vssh = vssh[ikeep]
    #
    if nmp.ma.is_masked(vssh): vssh[nmp.where( nmp.ma.getmask(vssh) )] = rmissval
    if ldebug: print('')
    return vssh.load()




### OUTPUT :


def Save2Dfield( ncfile, XFLD, xlon=[], xlat=[], name='field', unit='', long_name='', mask=[], clon='nav_lon', clat='nav_lat' ):
    #Mask
    (nj,ni) = nmp.shape(XFLD)
    if nmp.shape(mask) != (0,):
        xtmp = nmp.zeros((nj,ni))
        xtmp[:,:] = XFLD[:,:]
        idx_land = nmp.where( mask < 0.5)
        xtmp[idx_land] = nmp.nan
    else:
        xtmp=XFLD

    #Turn fields into dataset
    foo=xr.DataArray(xtmp,dims=['y','x'])
    foo.name=name
    if unit      != '': foo.attrs["units"] = unit
    if long_name != '': foo.attrs["long_name"] = long_name

    #Save to netcdf
    ds=foo.to_dataset()
    ds.attrs=dict(about=cabout_nc)
    ds.to_netcdf(ncfile,'w', format='NETCDF4')

def SaveTimeSeries( ivt, xd, vvar, ncfile, time_units='unknown', vunits=[], vlnm=[], missing_val=-9999. ):
    '''
    #  * ivt: time vector of length Nt, unit: UNIX Epoch time            [integer]
    #         => aka "seconds since 1970-01-01 00:00:00"
    #  *  xd: 2D numpy array that contains Nf time series of length Nt   [real]
    #          => hence of shape (Nf,Nt)
    #  * vvar: vector of length Nf of the Nf variable names                         [string]
    #  * vunits, vlnm: vectors of length Nf of the Nf variable units and long names [string]
    #  * missing_val: value for missing values...                        [real]
    '''
    (Nf,Nt) = xd.shape
    if len(ivt) != Nt: MsgExit('SaveTimeSeries() => disagreement in the number of records between "ivt" and "xd"')
    if len(vvar)!= Nf: MsgExit('SaveTimeSeries() => disagreement in the number of fields between "vvar" and "xd"')
    l_f_units = (nmp.shape(vunits)==(Nf,)) ; l_f_lnm = (nmp.shape(vlnm)==(Nf,))
    #
#    print('\n *** About to write file "'+ncfile+'"...')

    if Nf == 1:
      foo=xr.DataArray(xd,dims=['time'],coords=[ivt.astype(nmp.float64)])
      foo.name=vvar
      foo.attrs["units"] = vunits
      foo.attrs["long_name"] = vlnm
      ds=foo.to_dataset(name = vvar)
      footime=xr.DataArray(ivt, dims=['time'],coords=[ivt.astype(nmp.float64)])
      ds['time_counter']=footime
      ds.attrs=dict(about=cabout_nc)
      ds.to_netcdf(ncfile,'w', format='NETCDF4')
    else:
      foo0=xr.DataArray(xd[0],dims=['time'],coords=[ivt.astype(nmp.float64)])
      foo0.name=vvar[0]
      foo0.attrs["units"] = vunits[0]
      foo0.attrs["long_name"] = vlnm[0]
      ds=foo0.to_dataset(name = vvar[0])
      for jf in range(1,Nf):
        foo=xr.DataArray(xd[jf],dims=['time'],coords=[ivt.astype(nmp.float64)])
        foo.name=vvar[jf]
        foo.attrs["units"] = vunits[jf]
        foo.attrs["long_name"] = vlnm[jf]
        ds[vvar[jf]]=foo
      footime=xr.DataArray(ivt, dims=['time'],coords=[ivt.astype(nmp.float64)])
      ds['time_counter']=footime
      ds.attrs=dict(about=cabout_nc)
      ds.to_netcdf(ncfile,'w', format='NETCDF4')


    print(' *** "'+ncfile+'" successfully written!\n')
    return 0
