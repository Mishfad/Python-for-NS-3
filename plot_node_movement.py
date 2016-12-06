import numpy as np
import pylab as pl

############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def config_manet():
    	manet_config=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/manet_config.txt',delimiter=':',dtype=str)
    #	manet_config=np.genfromtxt('/Users/mishfadsv/Documents/NS-3/ns-allinone-3.25/ns-3.25/manet_config.txt',delimiter=':',dtype=str)
	#print "manet config:\n",manet_config
	nNodes=int(manet_config[0,1])
	nSpeed=int(manet_config[1,1])
	xRange=int(manet_config[2,1])
	yRange=int(manet_config[3,1])
	destn_node=int(manet_config[4,1])
	src_node=int(manet_config[5,1])

	return nNodes,nSpeed,xRange,yRange,destn_node,src_node

############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



def updateallexcept(x_plot,y_plot,index_pointer,node_index):
    for ind1 in range(nNodes):
        if (node_index!=ind1):
            x_plot[index_pointer,ind1]=x_plot[index_pointer-1,ind1]
            y_plot[index_pointer,ind1]=y_plot[index_pointer-1,ind1]

#-----------------------------------------------------------------------------



################################################################################################
	# Position extraction
################################################################################################
def GetMobility():
    	mob=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/fifth_mobility.txt',delimiter=',',dtype=str)
    #mob=np.genfromtxt('/Users/mishfadsv/Documents/NS-3/ns-allinone-3.25/ns-3.25/fifth_mobility.txt',delimiter=',',dtype=str)

	mob_shape=mob.shape
	#print mob[:3]
#----------------------------------------------------------------------------------------
# Splitting the mobility data into time, node, x and y positions
#----------------------------------------------------------------------------------------
	time_mob=[]
	node=[]
	xpos =[]
	ypos =[]
	for row_index in range(0,mob_shape[0]):
	    time_mob.append(float(mob[row_index,0]))
	    node.append(mob[row_index,1].split(':'))
	    xpos.append(mob[row_index,2].split(':'))
	    ypos.append(mob[row_index,3].split(':'))
#------------------------------------------------------
# x and y position and the corresponding node parsing
#------------------------------------------------------
# First filter out the strings vel,pos, and node
	x_mob=[float(x[1]) for x in xpos]
	y_mob=[float(x[1]) for x in ypos]
	node_mob=[int(x[1]) for x in node]

	print "Nodes",node_mob[:11]
	print "Position x and y\n",(x_mob[:3],y_mob[:3])
	len_mob= len(x_mob)
	x_plot=np.zeros((len_mob,nNodes))
	y_plot=np.zeros((len_mob,nNodes))

	time_plot=[]
	x_plot[0,:]=x_mob[:nNodes]
	y_plot[0,:]=y_mob[:nNodes]
	print time_mob[:3]
	index=0
	index_plot=0
	x_plot[0,node_mob[0]]=x_mob[0]
	y_plot[0,node_mob[0]]=y_mob[0]
	time_plot.append(time_mob[0])

# Updation of the position with time
# if time is different, we update all the positions. Position of currentnode is updated first. Then, the remaining node positions are copied to this row using updateallexcept(.,.) function
	for index in range(len_mob-1):
		if (time_mob[index+1]>time_mob[index]):
	# update index_plot pointer
			index_plot+=1
			time_plot.append(time_mob[index+1])
			x_plot[index_plot,node_mob[index+1]]=x_mob[index+1]
			y_plot[index_plot,node_mob[index+1]]=y_mob[index+1]
	# updateallexcept(current_row_pointer,nodelocation_which_neednotbe_updated)
			updateallexcept(x_plot,y_plot,index_plot,node_mob[index+1])
	# if time of position updation are equal
	  	elif time_mob[index+1]==time_mob[index]:
		#if (node_temp[index+1]!=node_temp[index])|(index<nNodes):
		    x_plot[index_plot,node_mob[index+1]]=x_mob[index+1]
		    y_plot[index_plot,node_mob[index+1]]=y_mob[index+1]

	return time_plot,x_plot,y_plot
# end of GetMobility()
################################################################################################

##############################################################################################
# Printing the neighbors
##############################################################################################
def print_neighbors(neighbor_nodes_time,time,x_plot,y_plot):
	index2=0
#	print "time:",time
	while time>neighbor_nodes_time[index2+1]:	
#		print neighbor_nodes[index2,0,0]
		index2+=1
#	print neighbor_nodes[index2-1,0,0]
	str2=[]
	for index_node1 in range(nNodes):
#		str1=['Node '+str(index_node1)]
		str1=[]
		for index_neighb1 in range(nNodes):
			if (neighbor_nodes[index2,index_node1,index_neighb1]==1):
				str1.append(index_neighb1)
#index_neighb1-1 because
		str2.append(str1)
#	print index2
#	print time,str2
	for i, txt in enumerate(n):
		str_annot=str(str2[i])
#		print i,str_annot
	        pl.annotate(str_annot,(x_plot[i]+10,y_plot[i]+xRange/20))
#	pl.pause(5)


# end of print_neighbors
#########################################################################################
# PLOTTING THE DATA
#########################################################################################
def ManetPlot():
	index_mob=0
	index_route=00
	plotting_time=0

	pl.ion()
	fig,ax=pl.subplots()
	for index_mob in range(len(time_plot)):
# plot the nodes with the positions given by index_mob of x_plot and yplot
		pl.scatter(x_plot[index_mob],y_plot[index_mob],s=100,c='g')
#		print x_plot[index_mob],y_plot[index_mob]
		for i, txt in enumerate(n):
		        pl.annotate(txt,(x_plot[index_mob, i]+10,y_plot[index_mob, i]+10))
		pl.xlabel('x axis')
		pl.ylabel('y axis')
	# set axis limits
		pl.xlim(0.0, xRange)
		pl.ylim(0.0, yRange)
		pl.title("Position updation at "+str(time_plot[index_mob]))
#		print time_plot[index_mob],route_table_time[index_route],ntemp,route_table[index_route,1:3]

#		print_neighbors(neighbor_nodes_time,time_neighb,x_plot[index_mob,:],y_plot[index_mob,:])
		pl.show()
		pl.pause(.0005)
	# show the plot on the screen
		pl.clf()



############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# START OF THE MAIN PROGRAM
############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

nNodes,nSpeed,xRange,yRange,destn_node,src_node=config_manet()

################################################################################################
# Find the location	
time_plot,x_plot,y_plot=GetMobility()
#print "Time in seconds after parsing: ",(time_plot[:5])

#### time_plot stores the time in seconds corresponding to the position values in x_plot and y_plot

#----------------------------------------------------------------------------------------------
# Finding the NEIGHBORS
#----------------------------------------------------------------------------------------------
#neighbor_nodes,neighbor_nodes_time=FindNeighbors()
#----------------------------------------------------------------------------------------------
# 	Plotting
#----------------------------------------------------------------------------------------------
n=[x for x in range(nNodes)]

ManetPlot()




