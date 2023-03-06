import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


### Función tomada del código del doc Martín para guardar todas las imágenes en un documento PDF
def multipage(filename, figs=None, dpi=200):
    
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


### Función tomada del código del doc Martín para graficar una curva
def XYplot(x,y, xmin, xmax, xlab, ylab, fig_title, symbol_color, scale=True): 
    
    #        import numpy as np

    fig = plt.figure()  
    fig.suptitle(fig_title, fontsize=12)
    plt.plot(x, y,  symbol_color)
    plt.xlim(xmin, xmax)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid(True)
    if scale:
        plt.xscale('log')
    

    return fig


### Función tomada del código del doc Martín para graficar varias curvas en la misma figura
def XYplot_profiles(nt, time, y_time, x_sec_min, x_sec_max, nx, xlab, ylab, fig_title, legend):
    #print('Entre a XYplot_profiles')
# These are the colors that will be used in the plot
    color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']


    fig = plt.figure()  
    fig.suptitle(fig_title, fontsize=12)
    delta_x=(x_sec_max-x_sec_min)/nx
    x_sec=np.multiply(range(0,nx+1,1),delta_x)
    
#    plt.subplots(1, 1, figsize=(12, 14))

    plt.xlim(x_sec_min, x_sec_max)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    
    plt.grid(True) 
    yt=np.zeros(nx+1)

    
    for i in range(0,nt+1,1):
            for j in range(0,nx+1,1):
                yt[j]=y_time[i,j]
            #subfig = fig.add_subplot(1,1,1)             
            label = 't = ' + str(time[i])    
            plt.plot(x_sec, yt,  color=color_sequence[i % 20], label=label) 
            if legend:
               subfig.legend()           
#            subfig.grid(True)

#   plt.show(block=True)
    #plt.show()
    return fig 

def compare(u, u_exact):      # user_action function
        """Compare exact and computed solution."""
        diff = u_exact - u
        norm=LA.norm(diff,np.inf)
        print('Global norm:', norm)  
