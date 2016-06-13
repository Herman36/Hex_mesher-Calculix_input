# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:22:25 2016

@author: herman
"""


#function hex_mesh to be called by calculix input generating program and by AEM programs

#input required:
    #2D or 3D                           =   option      integer
    #element size: l,w,h                =   size        array   [x,y,z]
    #rows, columns, stacks              =   grid        array   [x,y,z]
    #gap direction                      =   gap_drct    string
    #gap, note only in x and y          =   gap         array   [x,y]
    #linear or quadratic                =   quad        boolean


#output given
    #x,y and z co-ordinates             =   nodes       matrix  
    #element number and nodes           =   element     matrix
    #element center co-ordinates        =   el_center   matrix
    
import numpy as np
#import pylab as pl

def hex_mesh(option, size, grid, gap, gap_drct, quad, dispM):
    l = grid[0]
    t = grid[1]
    h = grid[2]
    if option == 2:         #2 dimentional mesh
        if quad == False:   #Linear and AEM elements
            if gap_drct == 'none':    #no gap between elements in either directions
                #defining nodes
                node_num = (l+1)*(t+1)
                nodes = np.zeros(shape=(node_num,3))               
                rhino = 0
                for y in range(0,(t+1)*size[1],size[1]):
                    for x in range(0,(l+1)*size[0],size[0]):
                        nodes[rhino,0] = rhino+1
                        nodes[rhino,1] = x
                        nodes[rhino,2] = y
                        rhino += 1
                #defining elements
                ele_num = l*t
                element = np.zeros(shape=(ele_num,5))
                zebra = 0
                for m in range(0,t,1):
                    for n in range(0,l,1):
                        element[zebra,0] = zebra+1
                        element[zebra,1] = n + m*l + m + 1
                        element[zebra,2] = n + m*l + m + 2
                        element[zebra,3] = n + m*l + m + l + 3
                        element[zebra,4] = n + m*l + m + l + 2
                        zebra += 1
            #-----------------------------------------------------------------#            
            elif gap_drct == 'x':   #gap in x direction only
                #defining nodes 
                node_num = (2*l)*(t+1)
                nodes = np.zeros(shape=(node_num,3))               
                rhino = 0
                for y in range(0,(t+1)*size[1],size[1]):
                    nodes[rhino,0] = rhino+1
                    nodes[rhino,1] = 0
                    nodes[rhino,2] = y
                    rhino += 1
                    for x in range(1,l,1):
                        nodes[rhino,0] = rhino+1
                        nodes[rhino,1] = x*size[0]+(x-1)*gap[0]
                        nodes[rhino,2] = y
                        nodes[rhino+1,0] = rhino+2
                        nodes[rhino+1,1] = x*size[0]+x*gap[0]
                        nodes[rhino+1,2] = y
                        rhino += 2
                    nodes[rhino,0] = rhino+1
                    nodes[rhino,1] = (x+1)*size[0]+x*gap[0]
                    nodes[rhino,2] = y
                    rhino += 1
                #defining elements
                ele_num = l*t
                element = np.zeros(shape=(ele_num,5))
                zebra = 0
                for m in range(0,t,1):
                    for n in range(0,l,1):
                        element[zebra,0] = zebra+1
                        element[zebra,1] = 2*n + 2*m*l + 1
                        element[zebra,2] = 2*n + 2*m*l + 2
                        element[zebra,3] = 2*n + 2*m*l + 2*l + 2 
                        element[zebra,4] = 2*n + 2*m*l + 2*l + 1 
                        zebra += 1
            #-----------------------------------------------------------------#
            elif gap_drct == 'y':   #gap in y direction only
                #defining nodes 
                node_num = (l+1)*(2*t)
                nodes = np.zeros(shape=(node_num,3))               
                rhino = 0
                for x in range(0,(l+1)*size[0],size[0]):
                    nodes[rhino,0] = rhino+1
                    nodes[rhino,1] = x
                    nodes[rhino,2] = 0
                    rhino += 1
                for y in range(1,t,1):
                    for x in range(0,(l+1)*size[0],size[0]):
                        nodes[rhino,0] = rhino+1
                        nodes[rhino,1] = x
                        nodes[rhino,2] = y*size[1]+(y-1)*gap[1]
                        rhino += 1
                    for x in range(0,(l+1)*size[0],size[0]):
                        nodes[rhino,0] = rhino+1
                        nodes[rhino,1] = x
                        nodes[rhino,2] = y*size[1]+y*gap[1]
                        rhino += 1
                for x in range(0,(l+1)*size[0],size[0]):
                    nodes[rhino,0] = rhino+1
                    nodes[rhino,1] = x
                    nodes[rhino,2] = (y+1)*size[1]+y*gap[1]
                    rhino += 1
                #defining elements  
                ele_num = l*t
                element = np.zeros(shape=(ele_num,5))
                zebra = 0
                for m in range(0,t,1):
                    for n in range(0,l,1):
                        element[zebra,0] = zebra+1
                        element[zebra,1] = n + 2*m*l + 2*m + 1
                        element[zebra,2] = n + 2*m*l + 2*m + 2
                        element[zebra,3] = n + 2*m*l + 2*m + l + 3
                        element[zebra,4] = n + 2*m*l + 2*m + l + 2
                        zebra += 1
            #-----------------------------------------------------------------#
            elif gap_drct == 'xy':      #gap in both directions
                #defining nodes
                node_num = 4*l*t
                nodes = np.zeros(shape=(node_num,3))               
                rhino = 0
                nodes[rhino,0] = rhino+1
                nodes[rhino,1] = 0
                nodes[rhino,2] = 0
                rhino += 1
                for x in range(1,l,1):
                    nodes[rhino,0] = rhino+1
                    nodes[rhino,1] = x*size[0]+(x-1)*gap[0]
                    nodes[rhino,2] = 0
                    nodes[rhino+1,0] = rhino+2
                    nodes[rhino+1,1] = x*size[0]+x*gap[0]
                    nodes[rhino+1,2] = 0
                    rhino += 2
                nodes[rhino,0] = rhino+1
                nodes[rhino,1] = (x+1)*size[0]+x*gap[0]
                nodes[rhino,2] = 0
                rhino += 1
                for y in range(1,t,1):
                    nodes[rhino,0] = rhino+1
                    nodes[rhino,1] = 0
                    nodes[rhino,2] = y*size[1]+(y-1)*gap[1]
                    rhino += 1
                    for x in range(1,l,1):
                        nodes[rhino,0] = rhino+1
                        nodes[rhino,1] = x*size[0]+(x-1)*gap[0]
                        nodes[rhino,2] = y*size[1]+(y-1)*gap[1]
                        nodes[rhino+1,0] = rhino+2
                        nodes[rhino+1,1] = x*size[0]+x*gap[0]
                        nodes[rhino+1,2] = y*size[1]+(y-1)*gap[1]
                        rhino += 2
                    nodes[rhino,0] = rhino+1
                    nodes[rhino,1] = (x+1)*size[0]+x*gap[0]
                    nodes[rhino,2] = y*size[1]+(y-1)*gap[1]
                    rhino += 1
                    #---------------------------------------#
                    nodes[rhino,0] = rhino+1
                    nodes[rhino,1] = 0
                    nodes[rhino,2] = y*size[1]+y*gap[1]
                    rhino += 1
                    for x in range(1,l,1):
                        nodes[rhino,0] = rhino+1
                        nodes[rhino,1] = x*size[0]+(x-1)*gap[0]
                        nodes[rhino,2] = y*size[1]+y*gap[1]
                        nodes[rhino+1,0] = rhino+2
                        nodes[rhino+1,1] = x*size[0]+x*gap[0]
                        nodes[rhino+1,2] = y*size[1]+y*gap[1]
                        rhino += 2
                    nodes[rhino,0] = rhino+1
                    nodes[rhino,1] = (x+1)*size[0]+x*gap[0]
                    nodes[rhino,2] = y*size[1]+y*gap[1]
                    rhino += 1
                #-----------------------------------------#
                nodes[rhino,0] = rhino+1
                nodes[rhino,1] = 0
                nodes[rhino,2] = (y+1)*size[1]+y*gap[1]
                rhino += 1
                for x in range(1,l,1):
                    nodes[rhino,0] = rhino+1
                    nodes[rhino,1] = x*size[0]+(x-1)*gap[0]
                    nodes[rhino,2] = (y+1)*size[1]+y*gap[1]
                    nodes[rhino+1,0] = rhino+2
                    nodes[rhino+1,1] = x*size[0]+x*gap[0]
                    nodes[rhino+1,2] = (y+1)*size[1]+y*gap[1]
                    rhino += 2
                nodes[rhino,0] = rhino+1
                nodes[rhino,1] = (x+1)*size[0]+x*gap[0]
                nodes[rhino,2] = (y+1)*size[1]+y*gap[1]
                rhino += 1
                #defining elements  
                ele_num = l*t
                element = np.zeros(shape=(ele_num,5))
                zebra = 0
                for m in range(0,t,1):
                    for n in range(0,l,1):
                        element[zebra,0] = zebra+1
                        element[zebra,1] = 2*n + 3*m*l + 4*m + 1
                        element[zebra,2] = 2*n + 3*m*l + 4*m + 2
                        element[zebra,3] = 2*n + 3*m*l + 4*m + 2*l + 2
                        element[zebra,4] = 2*n + 3*m*l + 4*m + 2*l + 1
                        zebra += 1
               
               #end of linear gap
            #-----------------------------------------------------------------#
            #Plotting of nodes
            if dispM == True:
                from matplotlib import pyplot as plt
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(nodes[:,1],nodes[:,2],'b*')
                tiger = 0
                for xy in zip(nodes[:,1], nodes[:,2]):
                    tiger += 1
                    ax.annotate('[%s]' % tiger, xy=xy, textcoords='data')
                plt.grid()
                plt.show()
            #end of linear
        #---------------------------------------------------------------------#
        else:
            from hex_quad_mesher import hex_mesh_quad
            nodes, element = hex_mesh_quad(option, size, grid, gap, gap_drct, dispM)
            #end of quad
        #---------------------------------------------------------------------#
        #end of option 2
    #-------------------------------------------------------------------------#
    elif option == 3:         #3 dimentional mesh
        if quad == False:   #Linear and AEM elements
            if gap_drct == 'none':      #no gap between elements in either x or y direction
                #defining nodes
                ele_num = l*t
                base_node_num = (l+1)*(t+1)
                node_num = base_node_num*(h+1)
                nodes = np.zeros(shape=(node_num,4))
                rhino = 0
                for z in range(0,(h+1)*size[2],size[2]):
                    for y in range(0,(t+1)*size[1],size[1]):
                        for x in range(0,(l+1)*size[0],size[0]):
                            nodes[rhino,0] = rhino+1
                            nodes[rhino,1] = x
                            nodes[rhino,2] = y
                            nodes[rhino,3] = z
                            rhino += 1
                #defining elements
                ele_num = l*t*h
                element = np.zeros(shape=(ele_num,9))
                zebra = 0
                for k in range(0,h,1):
                    for m in range(0,t,1):
                        for n in range(0,l,1):
                            element[zebra,0] = zebra+1
                            element[zebra,1] = n + m*(l+1) + k*base_node_num + 1
                            element[zebra,2] = n + m*(l+1) + k*base_node_num + 2
                            element[zebra,3] = n + m*(l+1) + k*base_node_num + l + 3
                            element[zebra,4] = n + m*(l+1) + k*base_node_num + l + 2
                            element[zebra,5] = n + m*(l+1) + (k+1)*base_node_num + 1
                            element[zebra,6] = n + m*(l+1) + (k+1)*base_node_num + 2
                            element[zebra,7] = n + m*(l+1) + (k+1)*base_node_num + l + 3
                            element[zebra,8] = n + m*(l+1) + (k+1)*base_node_num + l + 2
                            zebra += 1
            #-----------------------------------------------------------------#
            elif gap_drct == 'x':       #gap in x direction only
                #defining nodes
                base_node_num = (2*l)*(t+1)
                node_num = base_node_num*(h+1)
                nodes = np.zeros(shape=(node_num,4))
                rhino = 0
                for z in range(0,(h+1)*size[2],size[2]):
                    for y in range(0,(t+1)*size[1],size[1]):
                        nodes[rhino,0] = rhino+1
                        nodes[rhino,1] = 0
                        nodes[rhino,2] = y
                        nodes[rhino,3] = z
                        rhino += 1
                        for x in range(1,l,1):
                            nodes[rhino,0] = rhino+1
                            nodes[rhino,1] = x*size[0]+(x-1)*gap[0]
                            nodes[rhino,2] = y
                            nodes[rhino,3] = z
                            nodes[rhino+1,0] = rhino+2
                            nodes[rhino+1,1] = x*size[0]+x*gap[0]
                            nodes[rhino+1,2] = y
                            nodes[rhino+1,3] = z
                            rhino += 2
                        nodes[rhino,0] = rhino+1
                        nodes[rhino,1] = (x+1)*size[0]+x*gap[0]
                        nodes[rhino,2] = y
                        nodes[rhino,3] = z
                        rhino += 1
                #defining elements
                ele_num = l*t*h
                element = np.zeros(shape=(ele_num,9))
                zebra = 0
                for k in range(0,h,1):
                    for m in range(0,t,1):
                        for n in range(0,l,1):
                            element[zebra,0] = zebra+1
                            element[zebra,1] = 2*n + 2*l*m + k*base_node_num + 1
                            element[zebra,2] = 2*n + 2*l*m + k*base_node_num + 2
                            element[zebra,3] = 2*n + 2*l*(m+1) + k*base_node_num + 2
                            element[zebra,4] = 2*n + 2*l*(m+1) + k*base_node_num + 1
                            element[zebra,5] = 2*n + 2*l*m + (k+1)*base_node_num + 1
                            element[zebra,6] = 2*n + 2*l*m + (k+1)*base_node_num + 2
                            element[zebra,7] = 2*n + 2*l*(m+1) + (k+1)*base_node_num + 2
                            element[zebra,8] = 2*n + 2*l*(m+1) + (k+1)*base_node_num + 1
                            zebra += 1
            #-----------------------------------------------------------------#
            elif gap_drct == 'y':       #gap in y direction only
                #defining nodes
                base_node_num = 2*t*(l+1)
                node_num = base_node_num*(h+1)
                nodes = np.zeros(shape=(node_num,4))
                rhino = 0
                for z in range(0,(h+1)*size[2],size[2]):
                    for x in range(0,(l+1)*size[0],size[0]):
                        nodes[rhino,0] = rhino+1
                        nodes[rhino,1] = x
                        nodes[rhino,2] = 0
                        nodes[rhino,3] = z
                        rhino += 1
                    for y in range(1,t,1):
                        for x in range(0,(l+1)*size[0],size[0]):
                            nodes[rhino,0] = rhino+1
                            nodes[rhino,1] = x
                            nodes[rhino,2] = y*size[1]+(y-1)*gap[1]
                            nodes[rhino,3] = z
                            rhino += 1
                        for x in range(0,(l+1)*size[0],size[0]):
                            nodes[rhino,0] = rhino+1
                            nodes[rhino,1] = x
                            nodes[rhino,2] = y*size[1]+y*gap[1]
                            nodes[rhino,3] = z
                            rhino += 1
                    for x in range(0,(l+1)*size[0],size[0]):
                        nodes[rhino,0] = rhino+1
                        nodes[rhino,1] = x
                        nodes[rhino,2] = (y+1)*size[1]+y*gap[1]
                        nodes[rhino,3] = z
                        rhino += 1
                #defining elements
                ele_num = l*t*h
                element = np.zeros(shape=(ele_num,9))
                zebra = 0
                for k in range(0,h,1):
                    for m in range(0,t,1):
                        for n in range(0,l,1):
                            element[zebra,0] = zebra+1
                            element[zebra,1] = n + 2*m*(l+1) + k*base_node_num + 1
                            element[zebra,2] = n + 2*m*(l+1) + k*base_node_num + 2
                            element[zebra,3] = n + 2*m*(l+1) + k*base_node_num + l + 3
                            element[zebra,4] = n + 2*m*(l+1) + k*base_node_num + l + 2
                            element[zebra,5] = n + 2*m*(l+1) + (k+1)*base_node_num + 1
                            element[zebra,6] = n + 2*m*(l+1) + (k+1)*base_node_num + 2
                            element[zebra,7] = n + 2*m*(l+1) + (k+1)*base_node_num + l + 3
                            element[zebra,8] = n + 2*m*(l+1) + (k+1)*base_node_num + l + 2
                            zebra += 1
            #-----------------------------------------------------------------#
            elif gap_drct == 'xy':      #gap in both directions
                #defining nodes
                base_node_num = 4*t*l
                node_num = base_node_num*(h+1)
                nodes = np.zeros(shape=(node_num,4))               
                rhino = 0
                for z in range(0,(h+1)*size[2],size[2]):
                    nodes[rhino,0] = rhino+1
                    nodes[rhino,1] = 0
                    nodes[rhino,2] = 0
                    nodes[rhino,3] = z
                    rhino += 1
                    for x in range(1,l,1):
                        nodes[rhino,0] = rhino+1
                        nodes[rhino,1] = x*size[0]+(x-1)*gap[0]
                        nodes[rhino,2] = 0
                        nodes[rhino,3] = z
                        nodes[rhino+1,0] = rhino+2
                        nodes[rhino+1,1] = x*size[0]+x*gap[0]
                        nodes[rhino+1,2] = 0
                        nodes[rhino+1,3] = z
                        rhino += 2
                    nodes[rhino,0] = rhino+1
                    nodes[rhino,1] = (x+1)*size[0]+x*gap[0]
                    nodes[rhino,2] = 0
                    nodes[rhino,3] = z
                    rhino += 1
                    for y in range(1,t,1):
                        nodes[rhino,0] = rhino+1
                        nodes[rhino,1] = 0
                        nodes[rhino,2] = y*size[1]+(y-1)*gap[1]
                        nodes[rhino,3] = z
                        rhino += 1
                        for x in range(1,l,1):
                            nodes[rhino,0] = rhino+1
                            nodes[rhino,1] = x*size[0]+(x-1)*gap[0]
                            nodes[rhino,2] = y*size[1]+(y-1)*gap[1]
                            nodes[rhino,3] = z
                            nodes[rhino+1,0] = rhino+2
                            nodes[rhino+1,1] = x*size[0]+x*gap[0]
                            nodes[rhino+1,2] = y*size[1]+(y-1)*gap[1]
                            nodes[rhino+1,3] = z
                            rhino += 2
                        nodes[rhino,0] = rhino+1
                        nodes[rhino,1] = (x+1)*size[0]+x*gap[0]
                        nodes[rhino,2] = y*size[1]+(y-1)*gap[1]
                        nodes[rhino,3] = z
                        rhino += 1
                        #---------------------------------------#
                        nodes[rhino,0] = rhino+1
                        nodes[rhino,1] = 0
                        nodes[rhino,2] = y*size[1]+y*gap[1]
                        nodes[rhino,3] = z
                        rhino += 1
                        for x in range(1,l,1):
                            nodes[rhino,0] = rhino+1
                            nodes[rhino,1] = x*size[0]+(x-1)*gap[0]
                            nodes[rhino,2] = y*size[1]+y*gap[1]
                            nodes[rhino,3] = z
                            nodes[rhino+1,0] = rhino+2
                            nodes[rhino+1,1] = x*size[0]+x*gap[0]
                            nodes[rhino+1,2] = y*size[1]+y*gap[1]
                            nodes[rhino+1,3] = z
                            rhino += 2
                        nodes[rhino,0] = rhino+1
                        nodes[rhino,1] = (x+1)*size[0]+x*gap[0]
                        nodes[rhino,2] = y*size[1]+y*gap[1]
                        nodes[rhino,3] = z
                        rhino += 1
                    #-----------------------------------------#
                    nodes[rhino,0] = rhino+1
                    nodes[rhino,1] = 0
                    nodes[rhino,2] = (y+1)*size[1]+y*gap[1]
                    nodes[rhino,3] = z
                    rhino += 1
                    for x in range(1,l,1):
                        nodes[rhino,0] = rhino+1
                        nodes[rhino,1] = x*size[0]+(x-1)*gap[0]
                        nodes[rhino,2] = (y+1)*size[1]+y*gap[1]
                        nodes[rhino,3] = z
                        nodes[rhino+1,0] = rhino+2
                        nodes[rhino+1,1] = x*size[0]+x*gap[0]
                        nodes[rhino+1,2] = (y+1)*size[1]+y*gap[1]
                        nodes[rhino+1,3] = z
                        rhino += 2
                    nodes[rhino,0] = rhino+1
                    nodes[rhino,1] = (x+1)*size[0]+x*gap[0]
                    nodes[rhino,2] = (y+1)*size[1]+y*gap[1]
                    nodes[rhino,3] = z
                    rhino += 1
                #defining elements  
                ele_num = l*t*h
                element = np.zeros(shape=(ele_num,9))
                zebra = 0
                for k in range(0,h,1):
                    for m in range(0,t,1):
                        for n in range(0,l,1):
                            element[zebra,0] = zebra+1
                            element[zebra,1] = 2*n + m*(3*l+4) + k*base_node_num + 1
                            element[zebra,2] = 2*n + m*(3*l+4) + k*base_node_num + 2
                            element[zebra,3] = 2*n + m*(3*l+4) + k*base_node_num + 2*l + 2
                            element[zebra,4] = 2*n + m*(3*l+4) + k*base_node_num + 2*l + 1
                            element[zebra,5] = 2*n + m*(3*l+4) + (k+1)*base_node_num + 1
                            element[zebra,6] = 2*n + m*(3*l+4) + (k+1)*base_node_num + 2
                            element[zebra,7] = 2*n + m*(3*l+4) + (k+1)*base_node_num + 2*l + 2
                            element[zebra,8] = 2*n + m*(3*l+4) + (k+1)*base_node_num + 2*l + 1
                            zebra += 1                         
                #end of linear gap
            #-----------------------------------------------------------------#
            #Plotting of nodes
            if dispM == True:
                from mpl_toolkits.mplot3d import Axes3D
                import matplotlib.pyplot as plt
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                tiger = -1
                for c in ('r','b'):            
                    for lion in range(len(nodes[:,0])/(h+1)):
                        tiger += 1
                        ax.scatter(nodes[:,1], nodes[:,2], nodes[:,3], c='b', marker='*')
                        ax.text(nodes[tiger,1],nodes[tiger,2],nodes[tiger,3], '%s' % (str(nodes[tiger,0])), size=10, 
                                zorder=1, color=c)                
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                plt.show()
            #end of linear
        #---------------------------------------------------------------------#
        else:
            from hex_quad_mesher import hex_mesh_quad
            nodes, element = hex_mesh_quad(option, size, grid, gap, gap_drct, dispM)   
            #end of quad
        #---------------------------------------------------------------------#                            
        #end of option 3
    #-------------------------------------------------------------------------#
    return nodes, element #, el_center





