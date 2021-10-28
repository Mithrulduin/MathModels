#This project is a simulation of the heat equation in 2D with periodic boundary conditions and custom initial condition.
#It is solved using Euler method with trapezoidal approximation. The grid is also customizible.
#Disclaimer: Certain packages or available solutions are intentionally omitted as they are the topic of the project.

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import time
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

#"spac" is a float which determines the density of the spacial grid.
#"temp" is a float which determines the density of the temporal grid.
#In some implementations, it is enough to determine the quotient spac/temp^2, and this number indeed determines
#the accuracy of the solution.
spac = 1.4
temp = 0.1

#The following constants determine the size of the temporal and spacial grid respectively.
size_temp = 800
size_spac = 100

#This is a general solution for the trapezoidal approximation of second derivation in our case.
def derivation_2(S, direction, t, x, y):
    if direction== 'x':
        result = (1/(spac*spac))*(S[t][(x-1)%size_spac][y]-2*S[t][x][y]+S[t][(x+1)%size_spac][y])
    if direction== 'y':
        result = (1 / (spac * spac)) * (S[t][x][(y-1)%size_spac] - 2 * S[t][x][y] + S[t][x][(y+1)%size_spac])
    return result

#This function calculates the whole state matrix. This solution has the advantage of saving values for every time t,
#but the matrix is bigger. Alternative solution would be to have a spacial matrix and create function to update
#the state by calling the function to calculate the next state.
def calculate_state(S):
    for i in range(1,S.shape[0]):
        for j in range(S.shape[1]):
            for k in range(S.shape[2]):
                S[i][j][k] = S[i-1][j][k]+ temp * (derivation_2(S, 'x', i-1, j, k) + derivation_2(S, 'y', i-1, j, k))


#This function returns a size_spac x size_spac matrix with a designated peak point and a smooth decay where k is a
#decay coefficient and c is the value at the peak
def peak_func (x_0, y_0, k, c):
    M = np.empty(shape = (size_spac, size_spac))
    for i in range(0, size_spac):
        for j in range(0, size_spac):
            M[i][j] = c * math.exp(-abs(k)*(math.pow((abs(i-x_0)%size_spac),4)+math.pow((abs(j-y_0)%size_spac),4)))
    return M

#The following function plots the data using plotly but the execution fails due to the size of the dataset.
def animate_plotly(frames):
    X = np.arange(-5, 5, 10 / size_spac)
    Y = np.arange(-5, 5, 10 / size_spac)
    X, Y = np.meshgrid(X, Y)
    X = np.tile(X.flatten(), size_temp)
    Y = np.tile(Y.flatten(), size_temp)
    out_arr = np.column_stack((np.repeat(np.arange(size_temp), size_spac * size_spac), X,
                               Y, S.flatten()))
    colorscale = [
        [0.0, 'rgb(25, 23, 10)'],
        [0.05, 'rgb(69, 48, 44)'],
        [0.1, 'rgb(114, 52, 47)'],
        [0.15, 'rgb(155, 58, 49)'],
        [0.2, 'rgb(194, 70, 51)'],
        [0.25, 'rgb(227, 91, 53)'],
        [0.3, 'rgb(250, 120, 56)'],
        [0.35, 'rgb(255, 152, 60)'],
        [0.4, 'rgb(255, 188, 65)'],
        [0.45, 'rgb(236, 220, 72)'],
        [0.5, 'rgb(202, 243, 80)'],
        [0.55, 'rgb(164, 252, 93)'],
        [0.6, 'rgb(123, 245, 119)'],
        [0.65, 'rgb(93, 225, 162)'],
        [0.7, 'rgb(84, 196, 212)'],
        [0.75, 'rgb(99, 168, 238)'],
        [0.8, 'rgb(139, 146, 233)'],
        [0.85, 'rgb(190, 139, 216)'],
        [0.9, 'rgb(231, 152, 213)'],
        [0.95, 'rgb(241, 180, 226)'],
        [1.0, 'rgb(206, 221, 250)']
    ]

    data = [dict(type='heatmap', x=X, y=Y, z=S[0], zmin=-3, zmax=3, zsmooth='best', colorscale=colorscale,
                 colorbar=dict(thickness=20, ticklen=4))]
    title = 'Heat equation'
    layout = dict(title=title,
                  autosize=False,
                  height=600,
                  width=600,
                  hovermode='closest',
                  xaxis=dict(range=[0, size_spac], autorange=False),
                  yaxis=dict(range=[0, size_spac], autorange=False),
                  showlegend=False,
                  updatemenus=[dict(type='buttons', showactive=False,
                                    y=1, x=-0.05, xanchor='right',
                                    yanchor='top', pad=dict(t=0, r=10),
                                    buttons=[dict(label='Play',
                                                  method='animate',
                                                  args=[None,
                                                        dict(frame=dict(duration=100,
                                                                        redraw=True),
                                                             transition=dict(duration=0),
                                                             fromcurrent=True,
                                                             mode='immediate')])])])
    frame = [
        dict(data=[dict(type='heatmap', z=S[k * frames].flatten(), zmax=3)], traces=[0], name='frame{}'.format(k), ) for
        k in range(int(size_temp / frames))]
    fig = dict(data=data, layout=layout, frames=frame)
    plt.icreate_animations(fig, filename='animheatmap')

#The following 2 functions togheter plot the dataset in 3D using matplolib
def update_3D(k, plot, ax, X, Y):
    ax.view_init(azim=k)
    plot[0]=ax.plot_surface(X, Y, S[k], cmap='magma')
    return plot

def animate_matplotlib_3D():
    X = np.arange(size_spac)
    Y = np.arange(size_spac)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot = [ax.plot_surface(X, Y, S[0], cmap='magma')]
    ax.set_zlim(S.min(), S.max())

    anim = animation.FuncAnimation(fig, update_3D, interval=8, frames=size_temp, repeat=True, fargs=(plot, ax, X, Y))
    anim.save('heat_eq_3D.gif', bitrate=500)
    plt.show()

#The following 2 functions togheter plot the dataset as a heatmap using matplolib
def update_2D(k):
    plot.set_data(S[k])
    return plot

def animate_matplotlib_2D():

    anim= animation.FuncAnimation(fig, update_2D, interval = 8, frames=size_temp, repeat=True, repeat_delay=4000)
    anim.save('heat_eq_heatmap.gif')
    plt.show()

    return anim

#The following function saves the data as .csv
def save_data(S):
    X = np.arange(size_spac)
    Y = np.arange(size_spac)
    Z = np.arange(size_temp)
    Z, X, Y = np.meshgrid(Z, X, Y)
    output = np.column_stack((Z.flatten(), X.flatten(), Y.flatten(), S.flatten()))
    np.savetxt('state.csv', output, delimiter=' ', fmt='%f')

#The following function loads the data from a .csv table
def load_data(name):
    dS = pd.read_csv(name, names=('time', 'x', 'y', 'state'), delimiter=' ', dtype=float)
    pd.to_numeric(dS['state'], downcast='float', errors='coerce')
    S = np.empty(shape=(size_temp, size_spac, size_spac), dtype=float)
    for counter, value in enumerate(dS['state']):
        S[counter // (size_spac * size_spac)][(counter // size_spac) % size_spac][counter % size_spac] = value
    return S


S = np.empty(shape=(size_temp, size_spac, size_spac))
np.random.seed(int(time.time()))
#The following loops create random initial state
for j in range (5):
    for i in range(20):
        S[0]+=peak_func(np.random.uniform(0,size_spac), np.random.uniform(0,size_spac), abs(np.random.normal(0.000005*math.pow(10,j), 0.000003*math.pow(10,j))), np.random.normal(0,4))
calculate_state(S)
fig = plt.figure()
plot = plt.imshow(S[0], cmap='magma', vmin=S.min(), vmax=S.max(), animated=True)
animate_matplotlib_2D()
