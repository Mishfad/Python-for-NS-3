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

################################################################################################
# Finding the NEIGHBORS
################################################################################################
def FindNeighbors():
	broad=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/aodv_routing_table.txt',delimiter=',',dtype=str)
	#f = open('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/aodv_routing.txt', 'r')
	#route=f.read()

	broad_row,broad_col=broad.shape
	print broad_row,broad_col
	# remove the last column
	broad=broad[:,:4]
	print broad[:3]

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
	print "Time:", time_broad[:5]
	print "Curr broad:", curr_broad[:5]
	print "dest_broad:", dest_broad[:5]
	print "gate_broad:", gate_broad[:15]

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
		if((gate_broad[index_time]!=254)|(gate_broad[index_time-1]!=254)):
			neighbor_nodes[index1,curr_broad[index_time],gate_broad[index_time]]=1
#			broad_updation_time[broadcasting_node[index_time]]=neighbor_nodes_time[row_index]
		else:
			index_time+=1

	#print "No:of rows of broadcast entries:",index1
	#print "Neighbor nodes:\n",neighbor_nodes[7:10,:,:]

	return neighbor_nodes,neighbor_nodes_time

nNodes,nSpeed,xRange,yRange,destn_node,src_node=config_manet()
neighbor_nodes=FindNeighbors()

