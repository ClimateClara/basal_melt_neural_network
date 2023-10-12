import numpy as np
from matplotlib import cm
import cartopy
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import matplotlib as mpl
import cmocean
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.ticker import LogFormatterSciNotation
import matplotlib.colors as colors

def sigdigit(a,n):
    """round a to n significant digits

      Examples:
        nico.sigdigit([0.,1.111111,0.],2)          -> array([0. , 1.1, 0. ])
        nico.sigdigit([999.9,1.111111,-323.684],2) -> array([1000. , 1.1, -320. ])
        nico.sigdigit(2.2222222222,3)              -> array([2.22])
        nico.sigdigit(0.,3)                        -> array([0.])
        nico.sigdigit([0.,0.,0.],3)                -> array([0., 0., 0.])

   """
    
    aa=np.array(a)
    masked = aa==0
    bb=np.ones(np.size(aa))
    if np.size(bb[~masked]) != 0:
        bb[~masked]=np.power(10,np.floor(np.log10(np.abs(aa[~masked]))))
        return np.rint(10**(n-1)*aa/bb)*10**(1-n)*bb
    else:
        return bb*0.e0


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    sx = np.size(x)
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    #y=np.convolve(w/w.sum(),s,mode='valid')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[np.size(x[window_len-1:0:-1]):np.size(x[window_len-1:0:-1])+sx]

#===========================================================================
# Local functions to handle symmetric-log color bars:

def symlog_transform(linthresh,linscale, a):
    """Inplace transformation."""
    linscale_adj = (linscale / (1.0 - np.e ** -1))
    with np.errstate(invalid="ignore"):
      masked = np.abs(a) > linthresh
    sign = np.sign(a[masked])
    log = (linscale_adj + np.log(np.abs(a[masked]) / linthresh))
    log *= sign * linthresh
    a[masked] = log
    a[~masked] *= linscale_adj
    return a

def symlog_inv_transform(linthresh,linscale, a):
    """Inverse inplace Transformation."""
    linscale_adj = (linscale / (1.0 - np.e ** -1))
    masked = np.abs(a) > (linthresh * linscale_adj)
    sign = np.sign(a[masked])
    exp = np.exp(sign * a[masked] / linthresh - linscale_adj)
    exp *= sign * linthresh
    a[masked] = exp
    a[~masked] /= linscale_adj
    return a

def map_with_contourf_coolwarm(melt_2D, grounded_msk, icesheet_msk, mparam):
    fig, ax = plt.subplots()
    fig.set_size_inches(8.25/1.3, 8.25/1.5/1.25)

    # Customize colormap :
    # NB: modify the Ncool to Nwarm ratio (total=256) to place zero as desired.
    Ncool=86
    Nwarm=256-Ncool
    #------------------------------------------
    # Defining IPCC colormap:
    #LinL = np.loadtxt('IPCC_cryo_div.txt')
    LinL = np.loadtxt(inputpath_colorbar+'IPCC_cryo_div.txt')
    LinL = LinL*0.01
    #
    b3=LinL[:,2] # value of blue at sample n
    b2=LinL[:,2] # value of blue at sample n
    b1=np.linspace(0,1,len(b2)) # position of sample n - ranges from 0 to 1
    # setting up columns for list
    g3=LinL[:,1]
    g2=LinL[:,1]
    g1=np.linspace(0,1,len(g2))
    r3=LinL[:,0]
    r2=LinL[:,0]
    r1=np.linspace(0,1,len(r2))
    # creating list
    R=zip(r1,r2,r3)
    G=zip(g1,g2,g3)
    B=zip(b1,b2,b3)
    # transposing list
    RGB=zip(R,B,G)
    rgb=zip(*RGB)
    # print rgb
    # creating dictionary
    k=['red', 'green', 'blue']
    LinearL=dict(zip(k,rgb)) # makes a dictionary from 2 lists
    ipcc_cmap=mpl.colors.LinearSegmentedColormap('ipcc',LinearL,256)
    #---------------------------------
    # moving the zero of colorbar
    cool = cm.get_cmap(cm.coolwarm_r, Ncool)
    tmp1 = cool(np.linspace(0.5, 0.85, Ncool)) # decrease 0.70 to have more white in the middle light-blue colors
    print(tmp1.shape)
    warm = cm.get_cmap(cm.coolwarm_r, Nwarm)
    tmp2 = warm(np.linspace(0, 0.5, Nwarm)) # increase 0.20 to have more white in the middle light-yellow colors
    print(tmp2.shape)
    newcolors = np.append(tmp1[::-1,:],tmp2[::-1,:],axis=0)
    newcmp = ListedColormap(newcolors)

    # extreme color range values and corresponding tick levels of the symmetric-log contourf levels:
    minval=-5.0
    maxval=135.0
    lin_threshold=1.0
    lin_scale=1.0
    [min_exp,max_exp]=symlog_transform(lin_threshold,lin_scale,np.array([minval,maxval]))
    lev_exp = np.arange( np.floor(min_exp),  np.ceil(max_exp)+1 )
    levs = symlog_inv_transform(lin_threshold,lin_scale,lev_exp)
    levs = sigdigit(levs,2)

    cax=ax.contourf(ref_melt_2D.x,ref_melt_2D.y,melt_2D,levs,cmap=newcmp,norm=mpl.colors.SymLogNorm(linthresh=lin_threshold, linscale=lin_scale,vmin=minval, vmax=maxval),zorder=0)
    #ax.contour(ref_melt_2D.x,ref_melt_2D.y,basnb,np.linspace(0.5,20.5,21),linewidths=0.5,colors='gray',zorder=5)
    ax.contour(ref_melt_2D.x,ref_melt_2D.y,grounded_msk,linewidths=0.5,colors='black',zorder=10)
    ax.contour(ref_melt_2D.x,ref_melt_2D.y,icesheet_msk,linewidths=0.5,colors='black',zorder=15)
    #ax.contour(ref_melt_2D.x,ref_melt_2D.y,box_msk,linewidths=0.5,colors='blue',zorder=10)

    # Zoom on Amundsen:
    zoomfac=2.85
    xll_ori = -2000e3
    yll_ori =  -900e3
    xur_ori = -1450e3
    yur_ori =  -150e3
    xll_des =   -50e3
    yll_des =  -500e3
    xur_des = xll_des + zoomfac * (xur_ori-xll_ori)
    yur_des = yll_des + zoomfac * (yur_ori-yll_ori)
    ax.plot([xll_ori, xur_ori, xur_ori, xll_ori, xll_ori],[yll_ori, yll_ori, yur_ori, yur_ori, yll_ori],'k',linewidth=0.6,zorder=20)
    ax.fill([xll_des, xur_des, xur_des, xll_des, xll_des],[yll_des, yll_des, yur_des, yur_des, yll_des],'w',edgecolor='k',zorder=25)

    i1=np.argmin(np.abs(ref_melt_2D.x.values-xll_ori))
    i2=np.argmin(np.abs(ref_melt_2D.x.values-xur_ori))+1
    j1=np.argmin(np.abs(ref_melt_2D.y.values-yll_ori))
    j2=np.argmin(np.abs(ref_melt_2D.y.values-yur_ori))+1
    xzoom= xll_des + zoomfac * (ref_melt_2D.x-xll_ori)
    yzoom= yll_des + zoomfac * (ref_melt_2D.y-yll_ori)

    print(i1, i2, j1, j2)
    print(np.shape(ref_melt_2D.values), np.shape(xzoom.values))
    ax.contourf(xzoom.isel(x=range(i1,i2)),yzoom.isel(y=range(j2,j1)),melt_2D.isel(x=range(i1,i2),y=range(j2,j1)),levs,cmap=newcmp,norm=mpl.colors.SymLogNorm(linthresh=lin_threshold, linscale=lin_scale,vmin=minval, vmax=maxval),zorder=30)
    ax.contour(xzoom.isel(x=range(i1,i2)),yzoom.isel(y=range(j2,j1)),grounded_msk.isel(x=range(i1,i2),y=range(j2,j1)),linewidths=0.5,colors='black',zorder=30)
    ax.contour(xzoom.isel(x=range(i1,i2)),yzoom.isel(y=range(j2,j1)),icesheet_msk.isel(x=range(i1,i2),y=range(j2,j1)),linewidths=0.5,colors='black',zorder=40)
    #ax.contour(xzoom.isel(x=range(i1,i2)),yzoom.isel(y=range(j2,j1)),box_msk.isel(x=range(i1,i2),y=range(j2,j1)),linewidths=0.15,colors='blue',zorder=35)
    ax.plot([xll_des, xur_des, xur_des, xll_des, xll_des],[yll_des, yll_des, yur_des, yur_des, yll_des],'k',linewidth=1.0,zorder=45)

    #-----

    ratio=1.00
    ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

    # colorbar :
    formatter = LogFormatterSciNotation(10, labelOnlyBase=False, minor_thresholds=(np.inf, np.inf)) # "(np.inf, np.inf)" so that all ticks will be labeled 
    cbar = fig.colorbar(cax, format=formatter, fraction=0.035, pad=0.02, ticks=levs)
    cbar.ax.set_title('m ice/yr') #,size=8
    cbar.outline.set_linewidth(0.3)
    cbar.ax.tick_params(which='both') #labelsize=6,

    #-----

    ax.set_xlim(-2800e3,2800e3)
    ax.set_ylim(-2300e3,2300e3)
    ax.set_title(mparam)
    
    plt.tight_layout()
    return fig

def myround(x, base=5):
    return (base * np.ceil(x/base)).astype(int)