import numpy as np
import pylab as pl

def updateallexcept(index_pointer,node_index):
    for ind1 in range(nNodes):
        if (node_index!=ind1):
            x_plot[index_pointer,ind1]=x_plot[index_pointer-1,ind1]
            y_plot[index_pointer,ind1]=y_plot[index_pointer-1,ind1]



nNodes=10
a=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/manet-routing-compare.mob',delimiter=' ',dtype=None)
a_shape=a.shape


time=[]
node=[]
pos =[]
vel =[]
#print time
#--------------------------------------------------------------
# Splitting the data into time, node, postion and velocity
#--------------------------------------------------------------
for row_index in range(0,a_shape[0]):
    #for col_index in range(0,a_shape[1])
    time.append(a[row_index,0].split('='))
    node.append(a[row_index,1].split('='))
    pos.append(a[row_index,2].split('='))
    vel.append(a[row_index,3].split('='))

#print (node,pos)
#------------------------------------------------------
# position and velocity parsing
#------------------------------------------------------
## First filter out the strings vel,pos, and node
pos_temp=[x[1] for x in pos]
vel_temp=[x[1] for x in vel]
node_temp=[x[1] for x in node]
time_temp=[x[1] for x in time]

len1= len(pos_temp)

pos1=np.zeros((len1,3))
vel1=np.zeros((len1,3))
## Parse the pos and vel data into x,y, and z values
for index in range(len(pos_temp)):
    pos_str=''.join(pos_temp[index])
    pos1[index,:]=np.fromstring(pos_str,dtype=None,sep=":")
    vel_str=''.join(vel_temp[index])
    vel1[index,:]=np.fromstring(vel_str,dtype=None,sep=":")

x_val=pos1[:,0]
y_val=pos1[:,1]

x_plot=np.zeros((len1,nNodes))
y_plot=np.zeros((len1,nNodes))

x_plot[0,:]=x_val[:nNodes]
y_plot[0,:]=y_val[:nNodes]

index=0
index_plot=0
x_plot[0,node_temp[0]]=x_val[0]
y_plot[0,node_temp[0]]=y_val[0]
# Updation of the position with time
# if time is different, we update all the positions. Position of currentnode is updated first. Then, the remaining node positions are copied to this row using updateallexcept(.,.) function
while index < len1-1:
    if (time_temp[index+1]>time_temp[index]):
        # update index_plot pointer
        index_plot=index_plot+1
        x_plot[index_plot,node_temp[index+1]]=x_val[index+1]
        y_plot[index_plot,node_temp[index+1]]=y_val[index+1]
    #   updateallexcept(current_row_pointer,nodelocation_which_neednotbe_updated)
        updateallexcept(index_plot,node_temp[index+1])
    
    elif time_temp[index+1]==time_temp[index]:
        #if (node_temp[index+1]!=node_temp[index])|(index<nNodes):
            x_plot[index_plot,node_temp[index+1]]=x_val[index+1]
            y_plot[index_plot,node_temp[index+1]]=y_val[index+1]
    index=index+1

#print index_plot

#print x_plot[index_plot-5:index_plot+1,:]
#-----------------------------------------------------
# Plotting
#-----------------------------------------------------
n=[x for x in range(nNodes)]
#print n
pl.ion()
fig,ax=pl.subplots()
pl.axis([0, 600, 0, 600])
for node_index in range(index_plot):
    for i, txt in enumerate(n):
        pl.annotate(txt,(x_plot[node_index, i],y_plot[node_index, i]))
    #        print (i,txt)

    pl.title('Plot of position of the nodes')
# make axis labels
    pl.xlabel('x axis')
    pl.ylabel('y axis')
# set axis limits
    pl.xlim(0.0, 1000.0)
    pl.ylim(0.0, 1000.)
    pl.plot(x_plot[node_index,0:3],y_plot[node_index,0:3])
    pl.scatter(x_plot[node_index],y_plot[node_index])
    pl.show()
    pl.pause(.22)
# show the plot on the screen
    pl.clf()



#print vel1[35:38]




