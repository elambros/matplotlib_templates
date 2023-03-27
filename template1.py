import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import matplotlib.image as image
import pandas as pd
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.colors as mc
import colorsys

#https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def lighten_color(color, amount=0.5):
    """
		I GOT THIS FUNCTION FROM HERE VVVVVVVV
    https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])




def func(x, a, c):
    return a /x + c

def func2(x, a, c):
    return a /x**6 + c


def func_exp(x, a, b,c):
    #return a*np.e**(-b*x)+c
    return a*b**(-x) +c

def func_exp2(x, a, b,c,d):
    return a*np.e**(-b*x**2)+c
    #return a*b**(x) +c


def smoothme(x,y,smin,smax,newx):
    
    f = interp1d(x, y,kind='cubic')
    
    xp = np.linspace(smin,smax,100)
    ip = f(xp) #np.interp(xp,arr[:,0],arr[:,1])

    popt, pcov = curve_fit(func, xp, ip)

    #sp = savgol_filter(ip,7, 5) # window size 51, polynomial order 3
    
    return newx,func(newx, *popt)

def smoothme2(x,y,smin,smax,newx):
    
    f = interp1d(x, y,kind='cubic')
    
    xp = np.linspace(smin,smax,100)
    ip = f(xp) #np.interp(xp,arr[:,0],arr[:,1])

    popt, pcov = curve_fit(func_exp, xp, ip)
    
    #sp = savgol_filter(ip,7, 5) # window size 51, polynomial order 3
    
    return newx,func_exp(newx, *popt)

def smoothme3(x,y,smin,smax,newx):
    
    f = interp1d(x, y,kind='cubic')
    
    xp = np.linspace(smin,smax,100)
    ip = f(xp) #np.interp(xp,arr[:,0],arr[:,1])

    popt, pcov = curve_fit(func2, xp, ip)

    #sp = savgol_filter(ip,7, 5) # window size 51, polynomial order 3
    
    return newx,func2(newx, *popt)



def set_spines(axs,boff=True):
    for axis in ['bottom','left',]:
        axs.spines[axis].set_linewidth(2)

    for axis in ['top','right']:
        axs.spines[axis].set_linewidth(2)

    #for axis in ['left']:
    #    axs.spines[axis].set_bounds(-9,0)

    #for axis in ['bottom']:
    #    axs.spines[axis].set_bounds(0,100)
    
    if not boff:
        axs.spines['bottom'].set_linewidth(0)

def make_plot(axs,xt=True,yt=True,xlabel=True,ylabel=True):
    mfs=30
    nfs=20

    
    axs.axhline(y=-8,ls="--",xmin=0.03,xmax=0.97,c="k",lw=2,zorder=-1)
    axs.set_xlim(-3,102)
    axs.set_ylim(-10.5,0.5)
    
    axs.tick_params(length=10,width=2,direction="out")
    set_spines(axs)

    #xticks = list(range(0,101,20))
    #yticks = list(range(-9,1,1))
    
    #axs.set_xticks(xticks)
    #axs.set_yticks(yticks)

    if xt==True:
        xticks = list(range(0,101,20))
        axs.set_xticks(xticks)
        axs.set_xticklabels(xticks,fontsize=nfs)
    else:
        xticks=[]
        axs.set_xticks(xticks)
        axs.set_xticklabels([],fontsize=nfs)
        
    if yt==True:
        yticks = list(range(-10,1,1))
        axs.set_yticks(yticks)
        axs.set_yticklabels(yticks,fontsize=nfs)
    else:
        yticks=[]
        axs.set_yticks(yticks)
        axs.set_yticklabels([],fontsize=nfs)


    if xlabel==True:
        axs.set_xlabel(r"$\varepsilon$",fontsize=mfs)

    if ylabel==True:
        axs.set_ylabel(r'G$_{\mathrm{solv}}$ (kcal/mol)',fontsize=mfs)
        
def make_inset(axs,xt=True,yt=True,xlabel=True,ylabel=True):
    mfs=30
    nfs=20

    
    axs.axhline(y=-8,ls="--",xmin=0.03,xmax=0.97,c="k",lw=2,zorder=-1)
    axs.set_xlim(80,100)
    axs.set_ylim(-10.0,-7.5)
    
    axs.tick_params(length=10,width=2,direction="out")
    set_spines(axs)      

    if xt==True:
        xticks = [80,90,100]
        axs.set_xticks(xticks)
        axs.set_xticklabels(xticks,fontsize=nfs)
    else:
        xticks=[]
        axs.set_xticks(xticks)
        axs.set_xticklabels([],fontsize=nfs)
        
    if yt==True:
        yticks = [-10.0,-9.5,-9.0,-8.5,-8.0,-7.5]
        axs.set_yticks(yticks)
        axs.set_yticklabels(yticks,fontsize=nfs)
    else:
        yticks=[]
        axs.set_yticks(yticks)
        axs.set_yticklabels([],fontsize=nfs)


    if xlabel==True:
        axs.set_xlabel(r"$\varepsilon$",fontsize=mfs)

    if ylabel==True:
        axs.set_ylabel(r'G$_{\mathrm{solv}}$ (kcal/mol)',fontsize=mfs)

        
        
def plot_data(axs,x,y,c='tomato',marker=r'$\Join$'):
    

    newx=np.linspace(0.1,100,1000)

    xxx, yyy = smoothme(x,y,2,98,newx)
    
    axs.plot(xxx[yyy<0],yyy[yyy<0],c=lighten_color(c, amount=0.8),
             lw=2,path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
    axs.plot(x,y,ms=12,mew=2,mec=c,mfc="none",marker=marker,c=c,
             ls="None",lw=2.3,path_effects=[pe.Stroke(linewidth=2.5, foreground='k'), pe.Normal()])



def set_text(axs, text):
    mfs=40
    nfs=30
    axs.text(s=text,x=40,y=-1,fontsize=nfs, ha="center")



###################################
# DATA PLOTTING SECTION
###################################


fig = plt.figure(tight_layout=True,figsize=(10,20))
gs = gridspec.GridSpec(800,100)
#gsl = [gs[0:60,0:29],gs[0:60,30:59],gs[0:60,60:89],gs[0:60,90:119],

#gsl = [gs[0:100,0:200],gs[0:100,200:400],gs[0:100,400:600],gs[0:100,600:800]]
#gsl = [gs[0:100,0:200],gs[0:100,200:400],gs[0:100,400:600],gs[0:100,600:800]]

gsl = [gs[0:200,0:100],gs[200:400,0:100],gs[400:600,0:100],gs[600:800,0:100]]

letlist=['a)','b)','c)','d)']
for gs in list(range(0,len(gsl))):
    xticks=True
    yticks=True
    xlabel=True
    ylabel=True
    text=""

    if gs in [0]:
        xticks=False
        xlabel=False
        ylabel=False
        text="Model1"
        qlab="Model1"
        lab = text
    if gs in [1]:
        yticks=True
        xticks=False
        ylabel=False
        xlabel=False
        text="Model2"
        lab = "Model2"
        qlab="Model2"
    if gs in [2]:
        xlabel=False
        xticks=False
        ylabel=False
        text="Model3"
        qlab="Model3"
        lab = text
    if gs in [100000000]:
        xlabel=False
        ylabel=False
        yticks=False
        text="Model100"
        qlab=text
        lab = text
    if gs in [3]:
        xlabel=True
        ylabel=False
        yticks=True
        text="Model4"
        qlab=text
        lab = text


    axs = fig.add_subplot(gsl[gs])
    axins = inset_axes(axs, width="30%", height="40%", loc=1,borderpad=2.5)
    make_plot(axs,xt=xticks,yt=yticks,xlabel=xlabel,ylabel=ylabel)
    make_inset(axins,xt=True,yt=True,xlabel=False,ylabel=False)
    set_text(axs,text)
    axs.text(x=5,y=-0.5,s=letlist[gs],fontsize=20)

    if gs in [0,1,2,3]:


        unit = 627.509
        #plot_data(axs,data[lab][:,0],data[lab][:,2]*unit,c="tomato")
        #plot_data(axins,data[lab][:,0],data[lab][:,2]*unit,c="tomato")


    lines = [Line2D([0], [0], color='dodgerblue', linewidth=3, linestyle='-',marker=r'$\Join$',ms=12),
         #Line2D([0], [0], color='navy', linewidth=3, linestyle='-', marker="o",ms=12,mfc="none",mew=3),
         Line2D([0], [0], color='tomato', linewidth=3, linestyle='-',marker=r'$\Join$',ms=12)]
         #Line2D([0], [0], color='crimson', linewidth=3, linestyle='-', marker="o",ms=12,mfc="none",mew=3) ]
    labels = ['Label 1', 'Label 2']


    if gs in [0]:
        axins.legend(lines, labels,frameon=False, ncol=1, fontsize=18, loc="center", bbox_to_anchor=(0.2, 0.05, 0.5, 0.5))
        #axs.legend(lines, labels,frameon=False, ncol=1, fontsize=18, loc="center", bbox_to_anchor=(0.51, 0.45, 0.5, 0.5))
    #if gs in [3]:
    #
    #    lines=[Line2D([0], [0], color='navy', linewidth=3, linestyle='-', marker="o",ms=12,mfc="none",mew=3)]
    #    labels=["PySCF Classical"]
    #    axs.legend(lines, labels,frameon=False, ncol=2, fontsize=18, loc="center right", bbox_to_anchor=(0.51, 0.45, 0.5, 0.5))


fig.text(x=0.03,y=0.5,s=r'Y$_{\mathrm{axis}}$ (units)',fontsize=40,rotation="vertical",va="center",ha="center")

plt.show()
