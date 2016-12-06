import numpy as np
import pylab as pl

############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def config_manet():
	manet_config=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/manet_config.txt',delimiter=':',dtype=str)
	print "manet config:\n",manet_config
	nNodes=int(manet_config[0,1])
	nSpeed=int(manet_config[1,1])
	xRange=int(manet_config[2,1])
	yRange=int(manet_config[3,1])
	destn_node=int(manet_config[4,1])
	src_node=int(manet_config[5,1])

	return nNodes,nSpeed,xRange,yRange,destn_node,src_node

############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$



def updateallexcept(index_pointer,node_index):
    for ind1 in range(nNodes):
        if (node_index!=ind1):
            x_plot[index_pointer,ind1]=x_plot[index_pointer-1,ind1]
            y_plot[index_pointer,ind1]=y_plot[index_pointer-1,ind1]

#-----------------------------------------------------------------------------

# if the packet exists in the route table, return the row index. If does not exist, return -5
def packet_search(packetId,index_table):
	for ind1 in range(index_table):
		if(route_table[ind1,2]==packetId):
			return ind1
	return -5
#-----------------------------------------------------------------------------
# route_status=> -1-no path exists, 0-exists with intermediate nodes, and 1- direct connection

#Each row in route table corresponds to the route at a time. col1 gives time, col2 shows the route status, i.e., if the route exists or not. col 3 shows the packet number if route exists; if status=-1, packet=-1, row is truncated and move to the next row. 
#if route exists, that is, status=0, col 4 onwards store the route details (forwarding node details).
#If the prev row corresponds to status=-1, new row is updated only if there exists a route in the current time, obtained from route_status variable.

def UpdateRouteTable():
	col_track=np.zeros(route_size[0],dtype=int)
	index=0
	i_row=0
	i_col=2
#initialization
	route_table[0,0]=0	# time
	route_table[0,1]=-1	# status
	route_table[0,2]=-1	# packet
#	route_table[0,3]=-2	# route end
	discard=0
	print node[:25]
	print route_size
# 2 pointers are used. row_pointer(i_row) to update the row and col_pointer(i_col) to update col. i_col starts at 3. route_table[:,0] stores the time, route_table[:,1] stores the route status and route_table[:,2] stores packet index

# if current row in route table and the current entry in the route status corresponds to route does not exist, simply skip it as current route table entry shows no route and so while plotting no route will be plotted

	for index in range(route_size[0]):
#		print index
		if(route_status[index]==-1)&(route_table[i_row,1]==-1):
			continue
#if any one of them is not -1, First check if the packetId is equal to the current packetId. IF not, search if the packet already exists in the route table or not (to increase the speed))
# packet_search(a,b) a-> packet index, b-> current table row index
		if packet[index]!=route_table[i_row,2]:
			index_packet=packet_search(packet[index],i_row)
		else:
			index_packet=i_row
# if packetId doesn't exist in the route table
# if packetId is less than the previous row of route table, discard the packet. update discard counter to calculate the no:of discarded packets
		if(index_packet==-5):
#			if(packet[index]<route_table[i_row,2]):
#				discard+=1
#				continue
# else, means if packetId is a new entry and is greater than the existing ones, increment i_row and update time, status,packetId and the node.
#			else:
				i_row+=1
				route_table[i_row,0]=time_route[index]
				route_table_time[i_row]=time_route[index]
				route_table[i_row,1]=route_status[index]
				route_table[i_row,2]=packet[index]
				i_col=3
				route_table[i_row,i_col]=node[index]
				col_track[i_row]=i_col
# if packet exists in the route table, increment the col_track of the row and store the current node in the next col.
		else:
			col_track[index_packet]+=1
			route_table[index_packet,col_track[index_packet]]=node[index]
	return i_row,col_track
# return the total number of rows in route_table			
# end of UpdateRouteTable
#-----------------------------------------------------------------------------



#-----------------------------------------------------------------------------
# INCOMPLETE.... Not used as of now
def AddSourceToRow(ind1):
	ind2=3
	while route_table[ind1,ind2]!=0:
		ind2+=1
#	ind2+=1
	route_table[ind1,ind2+1]=-2
	while ind2>2:
		route_table[ind1,ind2+1]=route_table[ind1,ind2]
		ind2-=1
	

#-----------------------------------------------------------------------------
########## May need to EDIT LATER. INCOMPLETE...

def TerminateRouteTable(len_route_table,col_track,src_node):
	for ind1 in range(len_route_table):
# if direct connection between source and destn exist
#		if (route_table[ind1,1]==1):
#			route_table[ind1,col_track[ind1]+1]=-2
		if(route_table[ind1,1]==-1):
			route_table[ind1,3]=-2
		elif(route_table[ind1,1]==0):
#			if(route_table[ind1,3]!=destn_node)|(route_table[ind1,col_track[ind1]]!=src_node):
			if(route_table[ind1,3]!=src_node):
				route_table[ind1,1]=-1


#print route[:3]


################################################################################################
# Finding the NEIGHBOURS
################################################################################################

def FindNeighbours():
	broad=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/fifth_broadcast.txt',delimiter=',',dtype=str)

	#print "Broadcast size: ",broad.shape
	broad_row,broad_col=broad.shape
	print "Broadcast data:\n",broad[:3,:]


	time_broad=[]
	context_broad=[]
	sour_broad=[]
	dest_broad=[]

	for row_index in range(broad_row):
	    #for col_index in range(0,a_shape[1])
		time_broad.append(float(broad[row_index,0]))
		context_broad.append(broad[row_index,1].split('/'))
		sour_broad.append(broad[row_index,2].split('.'))
		dest_broad.append(broad[row_index,3].split('.'))

	#print time_broad[:3]
	#print sour_broad[:5]
	#print dest_broad[:3]

	#------------------------------------------------------
	#	Find the current,source, and destn nodes
	#------------------------------------------------------
	broadcast_rxg_node=[int(x[2]) for x in context_broad]
	broadcasting_node=[int(x[4])-1 for x in sour_broad]

	print "Broadcast rxg node: ",broadcast_rxg_node[:10]
	print "Broadcasting node ",broadcasting_node[:5]
	#------------------------------------------------------
	#	Create neighbor node list 
	#	neighbor_node[a,b,c] a-time b-nodes c-neighbors
	#------------------------------------------------------

	# Initialization of the loop
	neighbor_nodes=np.zeros((len(time_broad),nNodes,nNodes),dtype=int)
	index_broadcast=broadcasting_node[0]	# To traverse along 2nd dim, i e., the node index
	index_rxg=broadcast_rxg_node[0]		# To traverse along broadcasr_rxg_node array
	#index=1
	neighbor_count=0	# To traverse along 3rd dim, i e., the neighbors
	index_time=1
	index1=0	# to traverse along the 1st dim
	neighbor_nodes[0,broadcasting_node[0],broadcast_rxg_node[0]]=1
	neighbor_nodes[0,0,0]=int(time_broad[0])
	updating_count=0

	#print broadcasting_node[199]
	#print len(broadcasting_node)

	while index_time <len(time_broad)-1:
		# check if the current broadcst node is same as prev broadcst node. If not,
		neighbor_nodes[index1,broadcasting_node[index_time],broadcast_rxg_node[index_time]]=1
		index_time+=1
	#	print index_time
		if (broadcasting_node[index_time]!=broadcasting_node[index_time-1]):
			updating_count+=1
		if updating_count==nNodes:
			index1+=1
			updating_count=0
			neighbor_nodes[index1,0,0]=int(time_broad[index_time])

	#print "No:of rows of broadcast entries:",index1
	#print "Neighbour nodes:\n",neighbor_nodes[:2,:,:]
	return neighbor_nodes

#### end of FindNeighbours

############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# START OF THE MAIN PROGRAM
############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

nNodes,nSpeed,xRange,yRange,destn_node,src_node=config_manet()

route=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/first_routing.txt',delimiter=',',dtype=str)

route_size=route.shape

route_table=-2*np.ones((route_size[0],16),dtype=int)
route_table_time=np.zeros((route_size[0]))



################################################################################################
# INITIAL SPLITTING OF THE DATA READ FROM THE FILE INTO PROPER COLUMNS
################################################################################################

#-----------------------------------------------------------------------------------------
# Splitting the route data into time, current node, destination, source  adressess and packet tag
#-----------------------------------------------------------------------------------------

time_route=[]
route_status=[]
node=[]
dest_addr =[]
source_addr =[]
packet=[]


for row_index in range(0,route_size[0]):
    time_route.append(float(route[row_index,0]))
    route_status.append(route[row_index,1].split(':'))
    node.append(route[row_index,2].split(':'))
    dest_addr.append(route[row_index,3].split(':'))
    source_addr.append(route[row_index,4].split(':'))
    packet.append(route[row_index,5].split(':'))
#print time_route[:3]
#print route_status[:3]
#print node[:3]
#print dest_addr1[:3]
#print source_addr1[:3]
#print packet1[:3]
################################################################################################
# INITIAL PARSING OF THE COLUMN ENTRIES INTO
################################################################################################

#------------------------------------------------------
# parsing route data to separate tag and data of node addresses and packet
#------------------------------------------------------
## First filter out the strings and address
route_status=[int(x[1]) for x in route_status]
node=[int(x[1]) for x in node]
dest_addr=[x[1] for x in dest_addr]
source_addr=[x[1] for x in source_addr]
packet=[int(x[1]) for x in packet]

#print time[:3]
#print route_status[:3]
#print node[:3]
#print dest_addr[:3]
#print source_addr[:3]
#print "Packet id:\n",packet[:3]

#len_route= len(time_route)
#print "Len1=",len_route

#------------------------------------------------------
# Separate the address to get node id
#------------------------------------------------------

#print "Before splitting; forwarding node address:\n",node_addr[:3]
#node_addr  = [x.split('.') for x in node_addr  ]
dest_addr  = [x.split('.') for x in dest_addr  ]
source_addr= [x.split('.') for x in source_addr]

#print "Route status:\n",route_status[:15]
#print "After splitting; forwarding nodes:\n",node[:15]
#print "After splitting; Destinatin node address:\n",dest_addr[:3]
#print "After splitting; source node address:\n" ,source_addr[:3]

#forw_node=[int(x[3])-1 for x in node_addr]
dest_node=[int(x[3])-1 for x in dest_addr]
sour_node=[int(x[3])-1 for x in source_addr]
#print "Time in seconds:\n",time_route[:9]
#print "Route status:\n",route_status[:9]
#print "Probed nodes are:\n",node[:9]
#print "Destin nodes are:\n",dest_node[:9]
#print "Source nodes are:\n",sour_node[:9]
#print "Packet id:\n",packet[:9]

#print "Len1=",len1


# route_status=> -1-no path exists, 0-exists with intermediate nodes, and 1- direct connection

#Each row in route table corresponds to the route at a time. col1 gives time, col2 shows the route status, i.e., if the route exists or not. col 3 shows the packet number if route exists; if status=-1, packet=-1, row is truncated and move to the next row. 
#if route exists, that is, status=0, col 4 onwards store the route details (forwarding node details).
#If the prev row corresponds to status=-1, new row is updated only if there exists a route in the current time, obtained from route_status variable.

len_route,track_col=UpdateRouteTable()

#print "Route table\n",route_table[:25,:]
print "Final value of i_row=",len_route
print "track col",track_col[:10]

# make route status of invalid route with -1
TerminateRouteTable(len_route,track_col,src_node)

print "Route table",route_table[:25,:]
print "Route table time",route_table_time[:25]

#len_route=i_row+1
#print "Source is always",sour_node[0]
#print "Destination is always",dest_node[0]
#print node_route[:10,:]

################################################################################################
# MOBILITY DATA
################################################################################################

mob=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/fifth_mobility.txt',delimiter=',',dtype=str)

mob_shape=mob.shape
print mob[:3]
#----------------------------------------------------------------------------------------
# Splitting the mobility data into time, node, x and y positions
#----------------------------------------------------------------------------------------

time_mob=[]
node=[]
xpos =[]
ypos =[]
for row_index in range(0,mob_shape[0]):
    #for col_index in range(0,a_shape[1])
    time_mob.append(float(mob[row_index,0]))
    node.append(mob[row_index,1].split(':'))
    xpos.append(mob[row_index,2].split(':'))
    ypos.append(mob[row_index,3].split(':'))

#print "Mobility change time:\n",time_mob[:3]
#print node[:5]
#print ypos[:5]

#------------------------------------------------------
# x and y position and the corresponding node parsing
#------------------------------------------------------
## First filter out the strings vel,pos, and node
x_mob=[float(x[1]) for x in xpos]
y_mob=[float(x[1]) for x in ypos]
node_mob=[int(x[1]) for x in node]
#time_temp=[x[1] for x in time_mob]

#print node_mob[:11]
#print "Position x and y\n",(x_mob[:3],y_mob[:3])


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
    #   updateallexcept(current_row_pointer,nodelocation_which_neednotbe_updated)
        	updateallexcept(index_plot,node_mob[index+1])
    
  	elif time_mob[index+1]==time_mob[index]:
        #if (node_temp[index+1]!=node_temp[index])|(index<nNodes):
            x_plot[index_plot,node_mob[index+1]]=x_mob[index+1]
            y_plot[index_plot,node_mob[index+1]]=y_mob[index+1]

#print index_plot
#print "Time:\n",time_plot[:5]

#time_mob=[np.fromstring(x,dtype=float,sep="n") for x in time_plot]
#time_mob=[float(x)/1000000000 for x in time_mob]

print "Time in seconds after parsing: ",(time_plot[:5])
len_mob=len(time_plot)
#print "X_plot",x_plot[index_plot-5:index_plot+1,:]

#### time_plot stores the time in seconds corresponding to the position values in x_plot and y_plot

################################################################################################
# Finding the NEIGHBOURS
################################################################################################

neighbor_nodes=FindNeighbours()

##############################################################################################
# Combining routing and mobility
##############################################################################################
def print_neighbors(time,x_plot,y_plot):
	index2=0
#	print "time:",time
	while time>neighbor_nodes[index2,0,0]:	
#		print neighbor_nodes[index2,0,0]
		index2+=1
#	print neighbor_nodes[index2-1,0,0]
	str2=[]
	for index_node1 in range(nNodes):
#		str1=['Node '+str(index_node1)]
		str1=[]
		for index_neighb1 in range(nNodes):
			if (neighbor_nodes[index2-1,index_node1,index_neighb1]==1):
				str1.append(index_neighb1)
		str2.append(str1)
#	print index2
#	print str2[1]
	for i, txt in enumerate(n):
		str_annot=str(str2[i])
#		print str_annot
	        pl.annotate(str_annot,(x_plot[i]+10,y_plot[i]+50))
# end of print_neighbors
#########################################################################################

def ManetPlot():
	
	index_mob=0
	index_route=0
	plotting_time=0

	pl.ion()
	fig,ax=pl.subplots()
	while (index_route<len_route)|(index_mob<index_plot):
# plot the nodes with the positions given by index_mob of x_plot and yplot
		pl.scatter(x_plot[index_mob],y_plot[index_mob],s=100,c='g')
		for i, txt in enumerate(n):
		        pl.annotate(txt,(x_plot[index_mob, i]+10,y_plot[index_mob, i]+10))
		pl.xlabel('x axis')
		pl.ylabel('y axis')
	# set axis limits
		pl.xlim(0.0, xRange)
		pl.ylim(0.0, yRange)

		ntemp=[]
		for index_temp in range(3,track_col[index_route]+1):
			ntemp.append(route_table[index_route,index_temp])


#------------------------------------------------------------------------------------------
# if the route_table_time is lesser than time plot, update the route
		if (time_plot[index_mob]>route_table_time[index_route]):
			pl.title("Route updation at "+str(plotting_time)+" Packet:"+str(route_table[index_route,2]))
#			print "Route updation at "+str(index_mob)+" "+str(time_plot[index_mob])
			plotting_time=route_table_time[index_route]
			time_neighb=route_table_time[index_route]
			index_route+=1
		elif (time_plot[index_mob]<route_table_time[index_route]):
			pl.title("Position updation at "+str(time_plot[index_mob])+" Packet:"+str(route_table[index_route,2]))
			plotting_time=time_plot[index_mob]
			time_neighb=time_plot[index_mob]
#			print "Route updation at "+str(index_mob)+" "+str(time_plot[index_mob])
			index_mob+=1
		print time_plot[index_mob],route_table_time[index_route-1],ntemp,route_table[index_route-1,1:3]

#-------------------------------------------------------------------------------------------			
		if (route_table[index_route-1,1]!=-1):
			pl.plot(x_plot[index_mob,ntemp],y_plot[index_mob,ntemp])
			pl.scatter(x_plot[index_mob,ntemp],y_plot[index_mob,ntemp],s=100,c='r')
		else:
			pl.title("Route doesn't exist at "+str(plotting_time)+" Packet:"+str(route_table[index_route-1,2]))
#-------------------------------------------------------------------------------------------			
		print_neighbors(time_neighb,x_plot[index_mob,:],y_plot[index_mob,:])

		pl.show()
		pl.pause(.005)
	# show the plot on the screen
		pl.clf()



#########################################################################################
# 	Plotting
#########################################################################################


#print "lens",len_route,len_mob
#print "routing matrix:\n ",route_table[:4,:]


#node_final[index_final,1:]=node_route[index_route,:]
#print "node ",node_final[index_final,1]
#print n
#col_route=3
#while route_table[1,col_route]!=-2:
#	col_route+=1
#print col_route
#ntemp=[]
#print ntemp
#for index_temp in range(3,col_route):
#	ntemp.append(int(route_table[1,index_temp]))
#ntemp.append(int(dest_node[0]))
#print "Node route: ",ntemp

#print index_mob
#print x_plot[index_mob+1,:]

print "len mob is",len_mob

## change this to FOR LOOP
n=[x for x in range(nNodes)]
ManetPlot()




