import numpy as np
import pylab as pl
import matplotlib.patches as mpatches
import operator
############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def config_manet():
#      	manet_config=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/manet_config.txt',delimiter=':',dtype=str)
    	manet_config=np.genfromtxt('/Users/mishfadsv/Documents/NS-3/ns-allinone-3.25/ns-3.25/manet_config.txt',delimiter=':',dtype=str)
	print "manet config:\n",manet_config
	nNodes=int(manet_config[0,1])
	nSpeed=int(manet_config[1,1])
	xRange=int(manet_config[2,1])
	yRange=int(manet_config[3,1])
	destn_node=int(manet_config[4,1])
	src_node=int(manet_config[5,1])
	nPackets=int(manet_config[6,1])

	return nNodes,nSpeed,xRange,yRange,destn_node,src_node,nPackets

############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


def updateallexcept(x_plot,y_plot,index_pointer,node_index):
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
#-----------------------------------------------------------------------------

# route_status=> -2-no path exists, 0-exists with intermediate nodes, and 1- direct connection
#Each row in route table corresponds to the route at a time. col0 gives time, col1 shows the route status, i.e., if the route exists or not. col2 shows the packet number if route exists; if status=-1, packet=-1, row is truncated and move to the next row. 
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
	#print node[:25]
	#print route_size
# 2 pointers are used. row_pointer(i_row) to update the row and col_pointer(i_col) to update col. i_col starts at 3. route_table[:,0] stores the time, route_table[:,1] stores the route status and route_table[:,2] stores packet index

# if current row in route table and the current entry in the route status corresponds to route does not exist, simply skip it as current route table entry shows no route and so while plotting no route will be plotted

	for index in range(route_size[0]-1):
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
			if (col_track[index_packet]<nNodes):
#			print packet[index_packet],index_packet,col_track[index_packet]
				route_table[index_packet,col_track[index_packet]]=node[index]
#	print "packet 527: ",(i_row,packet_search(527,i_row))
#	print route_table[101,2]
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
#		elif(route_table[ind1,1]==0):
#			if(route_table[ind1,3]!=destn_node)|(route_table[ind1,col_track[ind1]]!=src_node):
#			if(route_table[ind1,3]!=src_node):
#				route_table[ind1,1]=-1


#print route[:3]

def UpdateRemBroadcastNodes(neighbor_nodes,neighbor_nodes_time,row_index,broadcasting_node_ptr,broad_updation_time):
    neighb_outdate_thresh=50	# 2 seconds
    for ind1 in range(nNodes):
	for ind2 in range(nNodes):
	        if (ind1!=broadcasting_node_ptr)&(neighbor_nodes_time[row_index]<broad_updation_time[ind1]+neighb_outdate_thresh): # check if the prev neighbor info is outdated
			neighbor_nodes[row_index,ind1,ind2]=neighbor_nodes[row_index-1,ind1,ind2]
			



################################################################################################
# Finding the NEIGHBORS
################################################################################################
def FindNeighbors():
#    broad=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/aodv_routing_table.txt',delimiter=',',dtype=str)
    broad=np.genfromtxt('/Users/mishfadsv/Documents/NS-3/ns-allinone-3.25/ns-3.25/aodv_routing_table.txt',delimiter=',',dtype=str)
	#f = open('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/aodv_routing.txt', 'r')
	#route=f.read()
    broad_row,broad_col=broad.shape
#    print broad_row,broad_col
	# remove the last column
    broad=broad[:,:4]
#    print broad[:3]

    time_broad=[]
    curr_broad=[]
    gate_broad=[]
    dest_broad=[]
	# split the entries into label and the entry
    for row_index in range(broad_row):
	    #for col_index in range(0,a_shape[1])
		time_broad.append(float(broad[row_index,0]))
		curr_broad.append((broad[row_index,1]).split(':'))
		dest_broad.append(broad[row_index,2].split(':'))
		gate_broad.append(broad[row_index,3].split(':'))

#	print time_broad[:3]
#	print curr_broad[:5]
#	print dest_broad[:3]

	#------------------------------------------------------
	#	Find the current,Destn, and Gateway nodes. Gateway nodes are the neighbor nodes
	#------------------------------------------------------
    curr_broad=[int(x[1]) for x in curr_broad]
    dest_broad=[(x[1].split('.')) for x in dest_broad]
    gate_broad=[(x[1].split('.')) for x in gate_broad]
#	print "Destn:",dest_broad[:5]

    dest_broad=[int(x[3])-1 for x in dest_broad]
    gate_broad=[int(x[3])-1 for x in gate_broad]
#    print "Time:", time_broad[:5]
#    print "Curr broad:", curr_broad[:5]
#    print "dest_broad:", dest_broad[:5]
#    print "gate_broad:", gate_broad[:15]

#------------------------------------------------------
#	Create neighbor node list 
#	neighbor_node[a,b,c] a-time_index (time is in time_broad) b-nodes c-neighbors
# for b and c, nodes=location. i.e., node0 is [:,0,:]
#------------------------------------------------------

	# Initialization of the loop
    neighbor_nodes=np.zeros((len(time_broad),nNodes,nNodes),dtype=int)
    neighbor_nodes_time=np.zeros(len(time_broad))
#	broad_updation_time=np.zeros(nNodes)

    index_curr_node=curr_broad[0]	# To traverse along 2nd dim, i e., the node index
    index_neighb=gate_broad[0]		# To traverse along gate_broad array
	#index=1
    neighbor_count=0	# To traverse along 3rd dim, i e., the neighbors
#	index_time=1
    index1=0	# to traverse along the 1st dim
    if (gate_broad[0]!=254):
		neighbor_nodes[0,curr_broad[0],gate_broad[0]]=1
    neighbor_nodes_time[0]=(time_broad[0])
	
	#print broadcasting_node[199]
	#print len(broadcasting_node)

    for index_time in range(1,len(time_broad)):
	# check if the current broadcst node is same as prev broadcst node. If not,
	# update all the rows other than the row corresponding to the current broadcasting node
	#	print index_time
        if (time_broad[index_time]!=time_broad[index_time-1]):
# UpdateRemBroadcastNodes(a,b,c,d,e)
# a-the matrix neighbor_nodes,	b- array neighbor_nodes_time, c-time index of 1st dim
# d-2nd dim 
#			UpdateRemBroadcastNodes(neighbor_nodes,neighbor_nodes_time,index1,curr_broad[index_time-1],broad_updation_time)
            index1+=1
            neighbor_nodes_time[index1]=(time_broad[index_time])
# skip the broadcast and local host entries from the routing table
        if((gate_broad[index_time]!=254)&(gate_broad[index_time-1]!=254)&(gate_broad[index_time]!=101)):
#			print gate_broad[index_time],time_broad[index_time]
            neighbor_nodes[index1,curr_broad[index_time],gate_broad[index_time]]=1
#			broad_updation_time[broadcasting_node[index_time]]=neighbor_nodes_time[row_index]

	#print "No:of rows of broadcast entries:",index1
	#print "Neighbor nodes:\n",neighbor_nodes[7:10,:,:]

    return neighbor_nodes,neighbor_nodes_time
#### end of FindNeighbors

################################################################################################
	# Position extraction
################################################################################################
def GetMobility():
    #mob=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/fifth_mobility.txt',delimiter=',',dtype=str)
    mob=np.genfromtxt('/Users/mishfadsv/Documents/NS-3/ns-allinone-3.25/ns-3.25/fifth_mobility.txt',delimiter=',',dtype=str)

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

#    print node_mob[:11]
#    print "Position x and y\n",(x_mob[:3],y_mob[:3])
    len_mob= len(x_mob)
    x_plot=np.zeros((len_mob,nNodes))
    y_plot=np.zeros((len_mob,nNodes))

    time_plot=[]
    x_plot[0,:]=x_mob[:nNodes]
    y_plot[0,:]=y_mob[:nNodes]
        #print time_mob[:3]
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
#    print time_plot[:5]
    return time_plot,x_plot,y_plot
# end of GetMobility()
################################################################################################
def IsNodeinTemp(ntemp,node):
	for index in range(len(ntemp)):
		if (ntemp[index]==node)&(node!=destn_node)&(node!=src_node):
			return 1
	return 0
	
			
	
##############################################################################################
# Printing the neighbors
##############################################################################################
def print_neighbors(neighbor_nodes_time,time,x_plot,y_plot,ntemp):
	index2=0
#	print "ntemp: ",ntemp
#	print "time:",time
	while time>neighbor_nodes_time[index2+1]:	
#		print neighbor_nodes[index2,0,0]
		index2+=1
	indexCache=0
	while time>cache_time[indexCache+1]:	
#		print neighbor_nodes[index2,0,0]
		indexCache+=1

#	print neighbor_nodes[index2-1,0,0]
	str2=[]
	for index_node1 in range(nNodes):
#		str1=['Node '+str(index_node1)]
		str1=[]
		for index_neighb1 in range(nNodes):
			if (neighbor_nodes[index2,index_node1,index_neighb1]==1)&(index_neighb1!=src_node)&(index_neighb1!=destn_node):
				str1.append(index_neighb1)
# checks if this node is in the route, not destination
		if ((IsNodeinTemp(ntemp,index_node1))&(index_node1!=src_node)):
#			print "Node: "+str(index_node1)+" ntemp: "+str(ntemp)+str(str1)
			pl.scatter(x_plot[str1],y_plot[str1],s=(cache_size[indexCache,str1]+1)*node_size,c='g')
#index_neighb1-1 because
		str2.append(str1)
#	print index2
#	print time,str2
	for i, txt in enumerate(n):
		str_annot=str(str2[i])
#		print i,str_annot
#	        pl.annotate(str_annot,(x_plot[i]+10,y_plot[i]+xRange/20))
#		pl.scatter(x_plot[i],y_plot[i],s=100,c='m')
#	print "ntemp: ",ntemp
#	for i in range(1,len(ntemp)):
#		if ntemp[i]!=dest_node:
#			pl.scatter(x_plot[ntemp[i]],y_plot[ntemp[i]],s=100,c='m')
#	pl.pause(5)
# end of print_neighbors

#########################################################################################
# SORTING THE TABLE
#########################################################################################

def sort_table(table, col=0):
    return sorted(table, key=operator.itemgetter(col))

#########################################################################################
# MAPPING PACKET ID TO SEQUENCE NUMBER
#########################################################################################
def ConvPackIdToSeqNum():
    #packtoseq=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/packetidToSeq.txt',delimiter=',',dtype=str)
    packtoseq=np.genfromtxt('/Users/mishfadsv/Documents/NS-3/ns-allinone-3.25/ns-3.25/packetidToSeq.txt',delimiter=',',dtype=str)
    
    packtoseq_shape=packtoseq.shape
    #print packtoseq_shape,packtoseq[:26]
    packet_id=[]
    seq_num=[]
    for row_index in range(0,packtoseq_shape[0]):
	    packet_id.append(packtoseq[row_index,0].split(':'))
	    seq_num.append(packtoseq[row_index,1].split(':'))

    packet_id=[int(x[1]) for x in packet_id]
    seq_num=[int(x[1]) for x in seq_num]
    packet_num=[int((x-1)/536) for x in seq_num]
#    print "packet id: ",packet_id[:25]
#    print "sequence num: ",seq_num[:25]
#    print "packet num: ",packet_num[:25]
	# CREATING THE TABLE FOR SORTING ACCORDING TO THE SEQUENCE NUMBER
    mytable=[(packet_id[ind],seq_num[ind],(seq_num[ind]-1)/536) for ind in range(len(packet_id))]
#    print mytable
#    print "sorted one"
    table=sort_table(mytable, 1)
#    for row in table:
#        print row
#    print table
    packet_id = [x[0] for x in table]
    seq_num   = [x[1] for x in table]
#    print packet_id
#    print "seq_num",seq_num
#    print "seq: ",[(int(x)-1)/536 for x in seq_num]
#    row_packetid=a
    RouteTablewithSeq=[]
    RouteTablewithSeq_time=[]
    track_col_seq=[]
    row_packetid=0
    for row_index in range(0,len(seq_num)-1):
#	print packet_id[row_packetid],len_route
	row_RT=packet_search(packet_id[row_index],len_route)
#	print "packet id:",packet_id[row_packetid]
#        print "row: ",row_RT
#	print "route_table packetid, row_packetid",(packet_id[row_index],route_table[row_RT,2])
        if (packet_id[row_index]==route_table[row_RT,2]):
#            print packet_id[row_packetid],seq_num[row_packetid],(seq_num[row_packetid]-1)/536
            RouteTablewithSeq.append(route_table[row_RT,:])
#	    print row_packetid,len(seq_num),len(RouteTablewithSeq)
#	    print RouteTablewithSeq[row_packetid][2],(seq_num[row_packetid]-1)/536
            RouteTablewithSeq[row_packetid][2]=(seq_num[row_packetid]-1)/536
            RouteTablewithSeq_time.append(route_table_time[row_RT])
            track_col_seq.append(track_col[row_RT])
            row_packetid=row_packetid+1
#    print "abc",[x[2] for x in RouteTablewithSeq[4:7]]
#    print "No of sequences: ",row_packetid
#    print RouteTablewithSeq[3][2]
    
    return RouteTablewithSeq,RouteTablewithSeq_time,track_col_seq ,row_packetid

#########################################################################################
# PLOTTING THE DATA BASED ON RouteTablewithSeq
#########################################################################################
def ManetPlotwithSeq():
    index_mob=0
    index_route=0
    plotting_time=0
    plot_index=0
#    print [x[2] for x in RouteTablewithSeq]
        
#    fig.set_tight_layout(True)


# Query the figure's on-screen size and DPI. Note that when saving the figure to
# a file, we need to provide a DPI for that separately.
#    print('fig size: {0} DPI, size in inches {1}'.format(fig.get_dpi(), fig.get_size_inches()))
    pl.ion()

#    while (index_route<len(track_col_seq)-2)|(index_mob<len(time_plot)):
    while (index_route<len(track_col_seq)-2):
	while (time_plot[index_mob]<RouteTablewithSeq_time[index_route]):
		index_mob+=1
            # plot the nodes with the positions given by index_mob of x_plot and yplot
	fig,ax=pl.subplots()
	node_dia=(cache_size[index_route,n]+1)*node_size
	node_dia[src_node]=0
        ax.scatter(x_plot[index_mob],y_plot[index_mob],s=node_dia,c='m')
#        for i, txt in enumerate(n):
#            pl.annotate(txt,(x_plot[index_mob, i]+10,y_plot[index_mob, i]+10))
#        pl.annotate("Source",(x_plot[index_mob, src_node]+10,y_plot[index_mob, src_node]-20))
#        pl.annotate("Dest",(x_plot[index_mob, destn_node]+10,y_plot[index_mob, destn_node]-20))
        #pl.xlabel('x axis')
        #pl.ylabel('y axis')
            # set axis limits
        pl.xlim(0.0, xRange)
        pl.ylim(0.0, yRange)
        ntemp=[]
        for index_temp in range(3,track_col_seq[index_route]+1):
            ntemp.append(RouteTablewithSeq[index_route][index_temp])

        #print RouteTablewithSeq_time[index_route],time_plot[index_mob]
        if (RouteTablewithSeq_time[index_route-1]>time_plot[index_mob]):
            time_neighb=time_plot[index_mob]
        else:
            time_neighb=RouteTablewithSeq_time[index_route-1]

#-------------------------------------------------------------------------------------------
        if (RouteTablewithSeq[index_route][1]!=-1):
	    print_neighbors(neighbor_nodes_time,time_neighb,x_plot[index_mob,:],y_plot[index_mob,:],ntemp)
#------------ Draw the route
            pl.scatter(x_plot[index_mob,ntemp[1:len(ntemp)]],y_plot[index_mob,ntemp[1:len(ntemp)]],s=(cache_size[index_route,ntemp[1:len(ntemp)]]+1)*node_size,c='r')
### source and destn
            pl.scatter(x_plot[index_mob,src_node],y_plot[index_mob,src_node],s=100,c='b')
            pl.scatter(x_plot[index_mob,destn_node],y_plot[index_mob,destn_node],s=(cache_size[index_route,destn_node]+1)*node_size,c='k')
	    ntemp.append(src_node)
            pl.plot(x_plot[index_mob,ntemp],y_plot[index_mob,ntemp],c='b')
                
        else:
            pl.title("Route doesn't exist at "+str(plotting_time)+" Packet:"+str(RouteTablewithSeq[index_route-1][2]))

        if (time_plot[index_mob]>RouteTablewithSeq_time[index_route]):
            pl.title("Route updation at "+str(plotting_time)+" Packet:"+str(RouteTablewithSeq[index_route][2])+' Caching probability, p='+str(p))
                        #			print "Route updation at "+str(index_mob)+" "+str(time_plot[index_mob])
            plotting_time=RouteTablewithSeq_time[index_route]
            index_route+=1
#        elif (time_plot[index_mob]<RouteTablewithSeq_time[index_route]):
#            pl.title("Position updation at "+str(time_plot[index_mob])+" Packet:"+str(RouteTablewithSeq[index_route][2]))
#            plotting_time=time_plot[index_mob]
                        #			print "Route updation at "+str(index_mob)+" "+str(time_plot[index_mob])
#            index_mob+=1
        else:
            print "Time equal"
            index_route+=1

#-------------------------------------------------------------------------------------------			
#	pl.scatter(x_plot[index_mob,ntemp],y_plot[index_mob,ntemp],s=(cache_size[index_route,ntemp]+1)*node_size,c='r')
#	pl.scatter(x_plot[index_mob,src_node],y_plot[index_mob,src_node],s=100,c='b')
#	pl.scatter(x_plot[index_mob,destn_node],y_plot[index_mob,destn_node],s=(cache_size[index_route,destn_node]+1)*node_size,c='k')
#	pl.plot(x_plot[index_mob,ntemp],y_plot[index_mob,ntemp],c='b')
	blue = mpatches.Patch(color='blue', label='Source')
	black = mpatches.Patch(color='k', label='Destination')
	red = mpatches.Patch(color='red', label='Forwarding nodes')
	neighbor = mpatches.Patch(color='g', label='Neighbor nodes')
	inactive = mpatches.Patch(color='m', label='Inactive nodes')
	prob = mpatches.Patch(color='w',label='Caching probability, p='+str(p))
	pl.legend(handles=[blue,black,red,neighbor,inactive])
#	pl.legend(handles=[magentha_patch])
	ax.set_yticklabels([])
	ax.set_xticklabels([])
#	pl.axis('off')
	pl.savefig("Route_1."+str(plot_index)+".png")
	plot_index+=1
#        pl.show()
        pl.pause(.005)
        # show the plot on the screen
        pl.clf()


#########################################################################################
# Update CACHE OF THE NEIGHBORS
#########################################################################################
def UpdateCacheofNeighbor(CachingTimeIndex,node,p):
	index=0
#	Traverse to the time just before the caching time to get the neighbor entries in RT at the time just before CachingTime
	while cache_time[CachingTimeIndex]>neighbor_nodes_time[index+1]:	
		index+=1
#	str1=[]
	for node_neighb in range(nNodes):
# Check if the given node_neighb is a neighbor of node. If yes, cache it with prob p
		if ((neighbor_nodes[index,node,node_neighb])&(np.random.uniform()<p)):
			cache_matrix[CachingTimeIndex,node_neighb,RouteTablewithSeq[CachingTimeIndex][2]]=1
#			print "inside if in cache neighbor"
#			print CachingTimeIndex,node_neighb,RouteTablewithSeq[CachingTimeIndex][2]



#########################################################################################
# CACHE THE PACKETS
#########################################################################################
def cache_packets(p):
#	cache_matrix[0,RouteTablewithSeq[0][:],0]=1
	for time_index in range(1,len(RouteTablewithSeq_time)):
#		print "RouteTablewithSeq:",RouteTablewithSeq[time_index]
		cache_time[time_index]=RouteTablewithSeq_time[time_index]
#		print "cache time:",cache_time[time_index]
		cache_matrix[time_index,:,:]=cache_matrix[time_index-1,:,:]
	        for index_temp in range(3,track_col_seq[time_index]+1):
#			print "node:",RouteTablewithSeq[time_index][index_temp],destn_node
			if (RouteTablewithSeq[time_index][index_temp]==destn_node):
				cache_matrix[time_index,RouteTablewithSeq[time_index][index_temp],RouteTablewithSeq[time_index][2]]=1
#				print "updating destination cache"
			elif(RouteTablewithSeq[time_index][index_temp]!=destn_node)&(np.random.uniform()<p)&(RouteTablewithSeq[time_index][index_temp]!=src_node):
				cache_matrix[time_index,RouteTablewithSeq[time_index][index_temp],RouteTablewithSeq[time_index][2]]=1
#				print "inside the if"
			if (RouteTablewithSeq[time_index][index_temp]!=destn_node):
				UpdateCacheofNeighbor(time_index,RouteTablewithSeq[time_index][index_temp],p)
		for node_index in range(nNodes):
			cache_size[time_index,node_index]=np.sum(cache_matrix[time_index,node_index,:])
#	print "cached_size\n",cache_size
#	return cache_size


############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# START OF THE MAIN PROGRAM
############$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
node_size=50
nNodes,nSpeed,xRange,yRange,destn_node,src_node,nPackets=config_manet()
print nPackets

route=np.genfromtxt('/Users/mishfadsv/Documents/NS-3/ns-allinone-3.25/ns-3.25/first_routing.txt',delimiter=',',dtype=str)
#route=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/first_routing.txt',delimiter=',',dtype=str)

route_size=route.shape
route_table=-2*np.ones((route_size[0],nNodes),dtype=int)
route_table_time=np.zeros((route_size[0]))

#print "routetable size: ",route_table.shape

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

#------------------------------------------------------
# Separate the address to get node id
#------------------------------------------------------
dest_addr  = [x.split('.') for x in dest_addr  ]
source_addr= [x.split('.') for x in source_addr]

dest_node=[int(x[3])-1 for x in dest_addr]
sour_node=[int(x[3])-1 for x in source_addr]

################################################################################################
# Update the route table
len_route,track_col=UpdateRouteTable()

#print "Route table\n",route_table[75:110,:]
#print "Final value of i_row=",len_route
#print "track col",track_col[:10]

# replace the route status of invalid route with -1
TerminateRouteTable(len_route,track_col,src_node)

#print "Route table",route_table[:25,:]
#print "Route table time",route_table_time[:25]

################################################################################################
# Find the location	
time_plot,x_plot,y_plot=GetMobility()
#print "Time in seconds after parsing: ",(time_plot[:5])

#### time_plot stores the time in seconds corresponding to the position values in x_plot and y_plot

#----------------------------------------------------------------------------------------------
# Finding the NEIGHBORS
#----------------------------------------------------------------------------------------------
neighbor_nodes,neighbor_nodes_time=FindNeighbors()
#----------------------------------------------------------------------------------------------
# 	Plotting
#----------------------------------------------------------------------------------------------
n=[x for x in range(nNodes)]

RouteTablewithSeq,RouteTablewithSeq_time,track_col_seq ,len_seq_num=ConvPackIdToSeqNum()
print "RouteTablewithSeq",len(RouteTablewithSeq)
cache_matrix=np.zeros((len(RouteTablewithSeq_time),nNodes,nPackets))
cache_time=np.zeros(len(RouteTablewithSeq_time))
cache_size=np.zeros((len(RouteTablewithSeq_time),nNodes))

#cache_packets(probability p)
p=0.4
cache_packets(p)
#print cache_size
ManetPlotwithSeq()


