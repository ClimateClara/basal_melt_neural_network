# ==================================================================
#  ____                 _   _           _  __        __         _
# |  _ \ _ __ __ _  ___| |_(_) ___ __ _| | \ \      / /__  _ __| | __
# | |_) | '__/ _` |/ __| __| |/ __/ _` | |  \ \ /\ / / _ \| '__| |/ /
# |  __/| | | (_| | (__| |_| | (_| (_| | |   \ V  V / (_) | |  |   <
# |_|   |_|  \__,_|\___|\__|_|\___\__,_|_|    \_/\_/ \___/|_|  |_|\_\
#                                                   Fidle module pwk                                   
# ==================================================================
# A simple module to host some common functions for practical work
# Jean-Luc Parouty 2020

import os,sys
import glob
import shutil
from datetime import datetime
import itertools
import datetime, time
import json

import math
import numpy as np
from collections.abc import Iterable

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from IPython.display import display,Image,Markdown,HTML

import fidle.config as config


datasets_dir  = None
notebook_id   = None
running_mode  = None
run_dir       = None

_save_figs    = False
_figs_dir     = './figs'
_figs_name    = 'fig_'
_figs_id      = 0

_start_time   = None
_end_time     = None
_chrono_start = None
_chrono_stop  = None

# -------------------------------------------------------------
# init_all
# -------------------------------------------------------------
#
def init(name=None, run_directory='./run'):
    global notebook_id
    global datasets_dir
    global run_dir
    global _start_time
    
    # ---- Parameters from config.py
    #
    notebook_id = config.DEFAULT_NOTEBOOK_NAME if name is None else name
    mplstyle    = config.FIDLE_MPLSTYLE
    cssfile     = config.FIDLE_CSSFILE
    
    # ---- Load matplotlib style and css
    #
    matplotlib.style.use(mplstyle)
    load_cssfile(cssfile)
    
    # ---- datasets location
    #
    datasets_dir = os.getenv('FIDLE_DATASETS_DIR', False)
    if datasets_dir is False:
        error_datasets_not_found()
    # Resolve tilde...
    datasets_dir=os.path.expanduser(datasets_dir)
        
    # ---- run_dir
    #
    attrs   = override('run_dir', return_attributes=True)
    run_dir = attrs.get('run_dir', run_directory)

    # Solution 2, for fun ;-)
    # run_dir = run_directory
    # override('run_dir', module_name='__main__')
    # override('run_dir', module_name=__name__, verbose=False)

    mkdir(run_dir)
    
    # ---- Update Keras cache
    #
    updated = update_keras_cache()
    
    # ---- Today and now
    #
    _start_time = datetime.datetime.now()
    
    # ---- Hello world
    #
    display_md('<br>**FIDLE 2020 - Practical Work Module**')
    print('Version              :', config.VERSION)
    print('Notebook id          :', notebook_id)
    print('Run time             :', _start_time.strftime("%A %d %B %Y, %H:%M:%S"))
    print('TensorFlow version   :', tf.__version__)
    print('Keras version        :', tf.keras.__version__)
    print('Datasets dir         :', datasets_dir)
    print('Run dir              :', run_dir)
    print('Update keras cache   :', updated)

    # ---- Save figs or not
    #
    save_figs = os.getenv('FIDLE_SAVE_FIGS', str(config.SAVE_FIGS) )
    if save_figs.lower() == 'true':
        set_save_fig(save=True, figs_dir=f'{run_dir}/figs', figs_name='fig_', figs_id=0)

    return datasets_dir

# ------------------------------------------------------------------
# Update keras cache
# ------------------------------------------------------------------
# Try to sync ~/.keras/cache with datasets/keras_cache
# because sometime, we cannot access to internet... (IDRIS..)
#
def update_keras_cache():
    updated = False
    if os.path.isdir(f'{datasets_dir}/keras_cache'):
        from_dir = f'{datasets_dir}/keras_cache/*.*'
        to_dir   = os.path.expanduser('~/.keras/datasets')
        mkdir(to_dir)
        for pathname in glob.glob(from_dir):
            filename=os.path.basename(pathname)
            destname=f'{to_dir}/{filename}'
            if not os.path.isfile(destname):
                shutil.copy(pathname, destname)
                updated=True
    return updated

# ------------------------------------------------------------------
# Where are my datasets ?
# ------------------------------------------------------------------
#
def error_datasets_not_found():        
    display_md('## ATTENTION !!\n----')
    print('Le dossier contenant les datasets est introuvable\n')
    print('Pour que les notebooks puissent les localiser, vous devez :\n')
    print('         1/ Récupérer le dossier datasets')
    print('            Une archive (datasets.tar) est disponible via le repository Fidle.\n')
    print("         2/ Préciser la localisation de ce dossier datasets via la variable")
    print("            d'environnement : FIDLE_DATASETS_DIR.\n")
    print('Exemple :')
    print("   Dans votre fichier .bashrc :")
    print('   export FIDLE_DATASETS_DIR=~/datasets')
    display_md('----')
    assert False, 'datasets folder not found, please set FIDLE_DATASETS_DIR env var.'

    
    
def override(*names, module_name='__main__', verbose=True, return_attributes=False):
    '''
    Try to override attributes given par name with environment variables.
    Environment variables name must be : FIDLE_OVERRIDE_<NOTEBOOK-ID>_<NAME>
    If no env variable is available for a given name, nothing is change.
    If type is str, substitution is done with 'notebook_id' and 'datasets_dir'
    Example : override('image_size','nb_epochs')
    params:
       names : list of attributes names as a str list
               if empty, all attributes can be override
    return :
       dict {name=new value}
    '''
    # ---- Where to override
    #
    module=sys.modules[module_name]
    
    # ---- No names : mean all
    #
    if len(names)==0:
        names=[]
        for name in dir(module):
            if name.startswith('_'): continue
            v=getattr(module,name)
            if type(v) not in [str, int, float, bool, tuple, list, dict]: continue
            names.append(name)
            
    # ---- Search for names
    #
    overrides={}
    for name in names:
        
        # ---- Environment variable name
        #
        env_name  = f'FIDLE_OVERRIDE_{notebook_id}_{name}'
        env_value = os.environ.get(env_name) 

        # ---- Environment variable : Doesn't exist
        #
        if env_value is None: continue

        # ---- Environment variable : Exist
        #
        value_old  = getattr(module,name)
        value_type = type(value_old)
        
        if value_type in [ str ] : 
            new_value = env_value.format(datasets_dir=datasets_dir, notebook_id=notebook_id)

        if value_type in [ int, float, bool, tuple, list, dict, type(None)]:
            new_value = eval(env_value)
    
        # ---- Override value
        #
        setattr(module,name,new_value)
        overrides[name]=new_value

    if verbose and len(overrides)>0:
        display_md('**\*\* Overrided parameters : \*\***')
        for name,value in overrides.items():
            print(f'{name:20s} : {value}')
            
    if return_attributes:
        return overrides
       
    
# -------------------------------------------------------------
# Folder cooking
# -------------------------------------------------------------
#
def tag_now():
    return datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

def mkdir(path):
    os.makedirs(path, mode=0o750, exist_ok=True)
      
def get_directory_size(path):
    """
    Return the directory size, but only 1 level
    args:
        path : directory path
    return:
        size in Mo
    """
    size=0
    for f in os.listdir(path):
        if os.path.isfile(path+'/'+f):
            size+=os.path.getsize(path+'/'+f)
    return size/(1024*1024)


# -------------------------------------------------------------
# shuffle_dataset
# -------------------------------------------------------------
#
def shuffle_np_dataset(x, y):
    """
    Shuffle a dataset (x,y)
    args:
        x,y : dataset
    return:
        x,y mixed
    """
    assert (len(x) == len(y)), "x and y must have same size"
    p = np.random.permutation(len(x))
    return x[p], y[p]

def rescale_dataset(*data, scale=1):
    '''
    Rescale numpy array with 'scale' factor
    args:
        *data : arrays
        scale : scale factor
    return:
        arrays of rescaled data
    '''
    return [ d[:int(scale*len(d))] for d in data ]

def pick_dataset(*data,n=5):
    '''Return random subsets of n elements'''
    ii = np.random.choice(range(len(data[0])), n)
    out = [ d[ii] for d in data ]
    return out[0] if len(out)==1 else out

def update_progress(what,i,imax, redraw=False):
    """
    Display a text progress bar, as :
    My progress bar : ############# 34%
    args:
        what  : Progress bas name
        i     : Current progress
        imax  : Max value for i
    return:
        nothing
    """
    bar_length = min(40,imax)
    if (i%int(imax/bar_length))!=0 and i<imax and not redraw:
        return
    progress  = float(i/imax)
    block     = int(round(bar_length * progress))
    endofline = '\r' if progress<1 else '\n'
    text = "{:16s} [{}] {:>5.1f}% of {}".format( what, "#"*block+"-"*(bar_length-block), progress*100, imax)
    print(text, end=endofline)

    
def rmax(l):
    """
    Recursive max() for a given iterable of iterables
    Should be np.array of np.array or list of list, etc.
    args:
        l : Iterable of iterables
    return: 
        max value
    """
    maxi = float('-inf')
    for item in l:
        if isinstance(item, Iterable):
            t = rmax(item)
        else:
            t = item
        if t > maxi:
            maxi = t
    return maxi

def rmin(l):
    """
    Recursive min() for a given iterable of iterables
    Should be np.array of np.array or list of list, etc.
    args:
        l : Iterable of iterables
    return: 
        min value
    """
    mini = float('inf')
    for item in l:
        if isinstance(item, Iterable):
            t = rmin(item)
        else:
            t = item
        if t < mini:
            mini = t
    return mini

# -------------------------------------------------------------
# show_images
# -------------------------------------------------------------
#
def plot_images(x,y=None, indices='all', columns=12, x_size=1, y_size=1,
                colorbar=False, y_pred=None, cm='binary', norm=None, y_padding=0.35, spines_alpha=1,
                fontsize=20, interpolation='lanczos', save_as='auto'):
    """
    Show some images in a grid, with legends
    args:
        x             : images - Shapes must be (-1,lx,ly) (-1,lx,ly,1) or (-1,lx,ly,3)
        y             : real classes or labels or None (None)
        indices       : indices of images to show or 'all' for all ('all')
        columns       : number of columns (12)
        x_size,y_size : figure size (1), (1)
        colorbar      : show colorbar (False)
        y_pred        : predicted classes (None)
        cm            : Matplotlib color map (binary)
        norm          : Matplotlib imshow normalization (None)
        y_padding     : Padding / rows (0.35)
        spines_alpha  : Spines alpha (1.)
        font_size     : Font size in px (20)
        save_as       : Filename to use if save figs is enable ('auto')
    returns: 
        nothing
    """
    if indices=='all': indices=range(len(x))
    if norm and len(norm) == 2: norm = matplotlib.colors.Normalize(vmin=norm[0], vmax=norm[1])
    draw_labels = (y is not None)
    draw_pred   = (y_pred is not None)
    rows        = math.ceil(len(indices)/columns)
    fig=plt.figure(figsize=(columns*x_size, rows*(y_size+y_padding)))
    n=1
    for i in indices:
        axs=fig.add_subplot(rows, columns, n)
        n+=1
        # ---- Shape is (lx,ly)
        if len(x[i].shape)==2:
            xx=x[i]
        # ---- Shape is (lx,ly,n)
        if len(x[i].shape)==3:
            (lx,ly,lz)=x[i].shape
            if lz==1: 
                xx=x[i].reshape(lx,ly)
            else:
                xx=x[i]
        img=axs.imshow(xx,   cmap = cm, norm=norm, interpolation=interpolation)
#         img=axs.imshow(xx,   cmap = cm, interpolation=interpolation)
        axs.spines['right'].set_visible(True)
        axs.spines['left'].set_visible(True)
        axs.spines['top'].set_visible(True)
        axs.spines['bottom'].set_visible(True)
        axs.spines['right'].set_alpha(spines_alpha)
        axs.spines['left'].set_alpha(spines_alpha)
        axs.spines['top'].set_alpha(spines_alpha)
        axs.spines['bottom'].set_alpha(spines_alpha)
        axs.set_yticks([])
        axs.set_xticks([])
        if draw_labels and not draw_pred:
            axs.set_xlabel(y[i],fontsize=fontsize)
        if draw_labels and draw_pred:
            if y[i]!=y_pred[i]:
                axs.set_xlabel(f'{y_pred[i]} ({y[i]})',fontsize=fontsize)
                axs.xaxis.label.set_color('red')
            else:
                axs.set_xlabel(y[i],fontsize=fontsize)
        if colorbar:
            fig.colorbar(img,orientation="vertical", shrink=0.65)
    save_fig(save_as)
    plt.show()

    
def plot_image(x,cm='binary', figsize=(4,4),interpolation='lanczos', save_as='auto'):
    """
    Draw a single image.
    Image shape can be (lx,ly), (lx,ly,1) or (lx,ly,n)
    args:
        x       : image as np array
        cm      : color map ('binary')
        figsize : fig size (4,4)
    """
    # ---- Shape is (lx,ly)
    if len(x.shape)==2:
        xx=x
    # ---- Shape is (lx,ly,n)
    if len(x.shape)==3:
        (lx,ly,lz)=x.shape
        if lz==1: 
            xx=x.reshape(lx,ly)
        else:
            xx=x
    # ---- Draw it
    plt.figure(figsize=figsize)
    plt.imshow(xx,   cmap = cm, interpolation=interpolation)
    save_fig(save_as)
    plt.show()


# -------------------------------------------------------------
# show_history
# -------------------------------------------------------------
#
def plot_history(history, figsize=(8,6), 
                 plot={"Accuracy":['accuracy','val_accuracy'], 'Loss':['loss', 'val_loss']},
                 save_as='auto'):
    """
    Show history
    args:
        history: history
        figsize: fig size
        plot: list of data to plot : {<title>:[<metrics>,...], ...}
    """
    fig_id=0
    for title,curves in plot.items():
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.ylabel(title)
        plt.xlabel('Epoch')
        for c in curves:
            plt.plot(history.history[c])
        plt.legend(curves, loc='upper left')
        if save_as=='auto':
            figname='auto'
        else:
            figname=f'{save_as}_{fig_id}'
            fig_id+=1
        save_fig(figname)
        plt.show()

    
    
def plot_confusion_matrix(y_true,y_pred,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          figsize=(10, 8),
                          digit_format='{:0.2f}',
                          save_as='auto'):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    cm = confusion_matrix( y_true,y_pred, normalize=None, labels=target_names)
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, digit_format.format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    save_fig(save_as)
    plt.show()


    
def display_confusion_matrix(y_true,y_pred,labels=None,color='green',
                             font_size='12pt', title="#### Confusion matrix is :"):
    """
    Show a confusion matrix for a predictions.
    see : sklearn.metrics.confusion_matrix

    Args:
        y_true:       Real classes
        y_pred:       Predicted classes
        labels:       List of classes to show in the cm
        color:        Color for the palette (green)
        font_size:    Values font size 
        title:        the text to display at the top of the matrix        
    """
    assert (labels!=None),"Label must be set"
    
    if title != None :  display(Markdown(title)) 
    
    cm = confusion_matrix( y_true,y_pred, normalize="true", labels=labels)
    df=pd.DataFrame(cm)

#     cmap = sn.light_palette(color, as_cmap=True)

    colorsList = ['whitesmoke','bisque']
    cmap = matplotlib.colors.ListedColormap(colorsList)
    cmap = matplotlib.colors.ListedColormap(cmap(np.linspace(0, 1, 256)))

    df.style.set_properties(**{'font-size': '20pt'})
    display(df.style.format('{:.2f}') \
            .background_gradient(cmap=cmap)
            .set_properties(**{'font-size': font_size}))
    
    
def plot_donut(values, labels, colors=["lightsteelblue","coral"], figsize=(6,6), title=None, save_as='auto'):
    """
    Draw a donut
    args:
        values   : list of values
        labels   : list of labels
        colors   : list of color (["lightsteelblue","coral"])
        figsize  : size of figure ( (6,6) )
    return:
        nothing
    """
    # ---- Title or not
    if title != None :  display(Markdown(title))
    # ---- Donut
    plt.figure(figsize=figsize)
    # ---- Draw a pie  chart..
    plt.pie(values, labels=labels, 
            colors = colors, autopct='%1.1f%%', startangle=70, pctdistance=0.85,
            textprops={'fontsize': 18},
            wedgeprops={"edgecolor":"w",'linewidth': 5, 'linestyle': 'solid', 'antialiased': True})
    # ---- ..with a white circle
    circle = plt.Circle((0,0),0.70,fc='white')
    ax = plt.gca()
    ax.add_artist(circle)
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')  
    plt.tight_layout()
    save_fig(save_as)
    plt.show()
    

    
def plot_multivariate_serie(sequence, labels=None, predictions=None, only_features=None,
                            columns=3, width=5,height=4,wspace=0.3,hspace=0.2,ms=6,lw=1,
                            save_as='auto', time_dt=1, hide_ticks=False):
    
    sequence_len = len(sequence)
    features_len = sequence.shape[1]
    if only_features is None : only_features=range(features_len)
    if labels is None        : labels=range(features_len)
    
    t  = np.arange(sequence_len)    
    if predictions is None:
        dt = 0
    else:
        dt = len(predictions)
        sequence_with_pred = sequence.copy()
        sequence_with_pred[-dt:]=predictions

    rows = math.ceil(features_len/columns)
    fig  = plt.figure(figsize=(columns*width, rows*height))
    fig.subplots_adjust(wspace=0.3,hspace=0.2)
    n=1
    for i in only_features:
        ax=fig.add_subplot(rows, columns, n)
        
        # ---- Real sequence without prediction
        #
        ax.plot( t[:-dt],sequence[:-dt,i], 'o',  markersize=ms, color='C0', zorder=2)
        ax.plot( t,sequence[:,i],          '-',  linewidth=lw,  color='C0', label=labels[i],zorder=1)

        # ---- What we expect
        #
        ax.plot(t[-dt:], sequence[-dt:,i], 'o', markeredgecolor='C0',markerfacecolor='white',ms=6)
        
        if predictions is not None:
            ax.plot(t[-dt-1:], sequence_with_pred[-dt-1:,i], '--',  lw=lw, fillstyle='full',  ms=ms, color='C1',zorder=1)
            ax.plot(t[-dt:],   predictions[:,i],             'o',   lw=lw, fillstyle='full',  ms=ms, color='C1',zorder=2)

        if hide_ticks:
            ax.set_yticks([])
            ax.set_xticks([])
        
        ax.legend(loc="upper left")
        n+=1
    save_fig(save_as)
    plt.show()

    
# -------------------------------------------------------------
# Show 2d series and segments
# -------------------------------------------------------------
#

def plot_2d_serie(data, figsize=(10,8), monocolor=False, hide_ticks=True, lw=2, ms=4, save_as='auto'):
    """
    Plot a 2d dataset as a trajectory
    args:
        data:      Dataset to plot
        figsize:   Figure size ( (10,8))
        monocolor: Monocolor line or not. (False)
    return:
        nothing
    """
    # ---- Get x,y, min and max
    #
    n     = len(data)
    k     = int(n/100)
    x,y   = data[:,0], data[:,1]
 
    # ---- Draw it
    #
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()

    # ---- Monocolor or gray gradient
    #
    if monocolor:
        ax.plot(x,y)
    else:
        for i in range(0,100):
            a= (200-i)/200
            ax.plot(x[i*k:(i+1)*k+1], y[i*k:(i+1)*k+1], '-', color=(a,a,a),lw=lw,zorder=1)

    # ---- Last point
    #
    ax.plot(x[n-1], y[n-1], 'o', color='C1',ms=ms,zorder=2)
    
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('axis=0')
    ax.set_ylabel('axis=1')
    
    if hide_ticks:
        ax.set_yticks([])
        ax.set_xticks([])

    save_fig(save_as)
    plt.show()
    
    

    
def plot_2d_segment(sequence_real, sequence_pred, figsize=(10,8), ms=6, lw=1, hide_ticks=True, save_as='auto'):
    """
    Plot a 2d segment real and predicted
    args:
        sequence_real: Real sequence
        sequence_pred: Predicted sequence
        figsize:       Figure size ( (10,8) )
        ms:            Marker size (6)
    return:
        nothing
    """
    k = len(sequence_pred)
    x,y = sequence_real[:,0],sequence_real[:,1]
    u,v = sequence_pred[:,0],sequence_pred[:,1]
    
    fig = plt.figure(figsize=figsize)

    ax = plt.axes()
    
    # ---- Draw real sequence without prediction
    #
    ax.plot(x[:-k], y[:-k],   'o', color='C0', fillstyle='full', zorder=2, ms=ms)
    ax.plot(x, y,             '-', color='C0', lw=lw, zorder=1)
    
    # ---- What we expect
    #
    ax.plot(x[-k:], y[-k:], 'o', ms=ms, markeredgecolor='C0', markerfacecolor='white', zorder=2)

    # ---- What we have
    #
    ax.plot(u, v,                            'o',  color='C1',fillstyle='full',zorder=2, ms=ms)
    ax.plot( [x[-1-k],u[0]], [y[-1-k],v[0]], '--', color='C1',lw=lw, zorder=1)
    ax.plot(u, v,                            '--', color='C1',lw=lw, zorder=1)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('axis=0')
    ax.set_ylabel('axis=1')
    
    if hide_ticks:
        ax.set_yticks([])
        ax.set_xticks([])

    save_fig(save_as)
    plt.show()

    
    
    

    
def set_save_fig(save=True, figs_dir='./run/figs', figs_name='fig_', figs_id=0):
    """
    Set save_fig parameters
    Default figs name is <figs_name><figs_id>.{png|svg}
    args:
        save      : Boolean, True to save figs (True)
        figs_dir  : Path to save figs (./figs)
        figs_name : Default basename for figs (figs_)
        figs_id   : Start id for figs name (0)
    """
    global _save_figs, _figs_dir, _figs_name, _figs_id
    _save_figs = save
    _figs_dir  = figs_dir
    _figs_name = figs_name
    _figs_id   = figs_id
    print(f'Save figs            : {_save_figs}')
    print(f'Path figs            : {_figs_dir}')
    
    
def save_fig(filename='auto', png=True, svg=False):
    """
    Save current figure
    args:
        filename : Image filename ('auto')
        png      : Boolean. Save as png if True (True)
        svg      : Boolean. Save as svg if True (False)
    """
    global _save_figs, _figs_dir, _figs_name, _figs_id
    if filename is None : return
    if not _save_figs   : return
    mkdir(_figs_dir)
    if filename=='auto': 
        path=f'{_figs_dir}/{notebook_id}-{_figs_name}{_figs_id:02d}'
    else:
        path=f'{_figs_dir}/{notebook_id}-{filename}'
    if png : plt.savefig( f'{path}.png')
    if svg : plt.savefig( f'{path}.png')
    if filename=='auto': _figs_id+=1
    display_html(f'<div class="comment">Saved: {path}</div>')
    

def subtitle(t):
    display(Markdown(f'<br>**{t}**'))
    
def display_md(text):
    display(Markdown(text))

def display_html(text):
    display(HTML(text))
    
def display_img(img):
    display(Image(img))

def chrono_start():
    global _chrono_start, _chrono_stop
    _chrono_start=time.time()

# return delay in seconds or in humain format
def chrono_stop(hdelay=False):
    global _chrono_start, _chrono_stop
    _chrono_stop = time.time()
    sec = _chrono_stop - _chrono_start
    if hdelay : return hdelay_ms(sec)
    return sec
    
def chrono_show():
    print('\nDuration : ', hdelay_ms(time.time() - _chrono_start))
    
def hdelay(sec):
    return str(datetime.timedelta(seconds=int(sec)))    
    
# Return human delay like 01:14:28 543ms
# delay can be timedelta or seconds
def hdelay_ms(delay):
    if type(delay) is not datetime.timedelta:
        delay=datetime.timedelta(seconds=delay)
    sec = delay.total_seconds()
    hh = sec // 3600
    mm = (sec // 60) - (hh * 60)
    ss = sec - hh*3600 - mm*60
    ms = (sec - int(sec))*1000
    return f'{hh:02.0f}:{mm:02.0f}:{ss:02.0f} {ms:03.0f}ms'

def hsize(num, suffix='o'):
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return f'{num:3.1f} {unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f} Y{suffix}'

def load_cssfile(cssfile):
    if cssfile is None: return
    styles = open(cssfile, "r").read()
    display(HTML(styles))
    
     
def np_print(*args, precision=3, linewidth=120):
    with np.printoptions(precision=precision, linewidth=linewidth):
        for a in args:
            print(a)
    
     
def end():
    global _end_time
    _end_time = datetime.datetime.now()
        
    print('End time is :', time.strftime("%A %d %B %Y, %H:%M:%S"))
    print('Duration is :', hdelay_ms(_end_time - _start_time))
    print('This notebook ends here')
     
     

