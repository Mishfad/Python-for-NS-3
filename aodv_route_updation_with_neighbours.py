import numpy as np
import pylab as pl

def updateallexcept(index_pointer,node_index):
    for ind1 in range(nNodes):
        if (node_index!=ind1):
            x_plot[index_pointer,ind1]=x_plot[index_pointer-1,ind1]
            y_plot[index_pointer,ind1]=y_plot[index_pointer-1,ind1]



nNodes=15
route=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/first_routing.txt',delimiter=',',dtype=str)

route_shape=route.shape

################################################################################################
# INITIAL SPLITTING OF THE DATA READ FROM THE FILE INTO PROPER COLUMNS
################################################################################################

#-----------------------------------------------------------------------------------------
# Splitting the route data into time, current node, destination, source  adressess and packet tag
#-----------------------------------------------------------------------------------------

time_route=[]
node_addr1=[]
dest_addr1 =[]
source_addr1 =[]
packet1=[]


for row_index in range(0,route_shape[0]):
    time_route.append(float(route[row_index,0]))
    node_addr1.append(route[row_index,1].split(':'))
    dest_addr1.append(route[row_index,2].split(':'))
    source_addr1.append(route[row_index,3].split(':'))
    packet1.append(route[row_index,4].split(':'))

#print time_route[:3]
#print node_addr1[:3]
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
node_addr=[x[1] for x in node_addr1]
dest_addr=[x[1] for x in dest_addr1]
source_addr=[x[1] for x in source_addr1]
packet=[int(x[1]) for x in packet1]

#print time[:3]
#print node_addr[:3]
#print dest_addr[:3]
#print source_addr[:3]
#print "Packet id:\n",packet[:3]

len_route= len(time_route)
#print "Len1=",len1

#------------------------------------------------------
# Separate the address to get node id
#------------------------------------------------------

#print "Before splitting; forwarding node address:\n",node_addr[:3]
node_addr  = [x.split('.') for x in node_addr  ]
dest_addr  = [x.split('.') for x in dest_addr  ]
source_addr= [x.split('.') for x in source_addr]


#print "After splitting; forwarding node address:\n",node_addr[:3]
#print "After splitting; Destinatin node address:\n",dest_addr[:3]
#print "After splitting; source node address:\n" ,source_addr[:3]

forw_node=[int(x[3])-1 for x in node_addr]
dest_node=[int(x[3])-1 for x in dest_addr]
sour_node=[int(x[3])-1 for x in source_addr]
#print "Time in seconds:\n",time_route[:9]
#print "Forwdg nodes are:\n",forw_node[:9]
#print "Destin nodes are:\n",dest_node[:9]
#print "Source nodes are:\n",sour_node[:9]
#print "Packet id:\n",packet[:9]

#print "Len1=",len1

node_route=np.zeros((len_route,8))
index=1
i_row=0
i_col=2

#initialization
node_route[i_row,0]=time_route[0]
node_route[i_row,1]=forw_node[0]
# 2 pointers are used. row_pointer(i_row) to update the row and col_pointer(i_col) to update col. i_col starts at 2. node[.,0] stores the time, node[.,1] stores the 


while index<len_route:
	if (packet[index]==packet[index-1]):
		node_route[i_row,i_col]=forw_node[index]
		index=index+1
		i_col=i_col+1

# if packet indices are different, increment the row_pointer and set col_pointer to 2
	else:
		node_route[i_row,i_col]=-1	# -1 shows the end of the row; used later to traverse through the row
		i_col=2
		i_row=i_row+1
		node_route[i_row,0]=time_route[index]		
		node_route[i_row,1]=forw_node[index]
		index+=1
		
print "Final value of i_row=",i_row
len_route=i_row+1
#print "Source is always",sour_node[0]
#print "Destination is always",dest_node[0]
print node_route[:10,:]


################################################################################################
# MOBILITY DATA
################################################################################################

mob=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/fifth_mobility.txt',delimiter=',',dtype=str)

mob_shape=mob.shape
print mob[:3]
#----------------------------------------------------------------------------------------
# Splitting the mobility data into time, node, postion and velocity
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

print "Mobility change time:\n",time_mob[:3]
print node[:5]
print ypos[:5]

#------------------------------------------------------
# position and velocity parsing
#------------------------------------------------------
## First filter out the strings vel,pos, and node
x_mob=[float(x[1]) for x in xpos]
y_mob=[float(x[1]) for x in ypos]
node_mob=[int(x[1]) for x in node]
#time_temp=[x[1] for x in time_mob]

print node_mob[:11]
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
while index < len_mob-1:
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
	index=index+1

print index_plot
print "Time:\n",time_plot[:5]

#time_mob=[np.fromstring(x,dtype=float,sep="n") for x in time_plot]
#time_mob=[float(x)/1000000000 for x in time_mob]

print "Time in seconds after parsing: ",(time_plot[:5])
len_mob=len(time_plot)
#print x_plot[index_plot-5:index_plot+1,:]

################################################################################################
# Finding the NEIGHBOURS
################################################################################################


broad=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/fifth_broadcast.txt',delimiter=',',dtype=str)

print broad.shape
broad_row,broad_col=broad.shape
print broad[:3,:]


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

print time_broad[:3]
print sour_broad[:5]
print dest_broad[:3]

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
print len(broadcasting_node)

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

print index1
print neighbor_nodes[:2,:,:]

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
		str1=['Node '+str(index_node1)]
		for index_neighb1 in range(nNodes):
			if (neighbor_nodes[index2-1,index_node1,index_neighb1]==1):
				str1.append(index_neighb1)
		str2.append(str1)
#	print index2
#	print str2[1]
	for i, txt in enumerate(n):
		str_annot=str(str2[i])
#		print str_annot
	        pl.annotate(str_annot,(x_plot[i]+10,y_plot[i]+10))
# end of print_neighbors
#########################################################################################



#########################################################################################
# 	Plotting
#########################################################################################


index_route=1
index_mob=0
index_final=0
print "lens",len_route,len_mob
print "routing matrix:\n ",node_route[:4,:]


#node_final[index_final,1:]=node_route[index_route,:]
#print "node ",node_final[index_final,1]
n=[x for x in range(nNodes)]
#print n
col_route=1
while node_route[0,col_route]!=-1:
	col_route+=1
#print col_route
ntemp=[int(sour_node[0])]
#print ntemp
for index_temp in range(1,col_route):
	ntemp.append(int(node_route[0,index_temp]))
ntemp.append(int(dest_node[0]))
print "Node route: ",ntemp

print index_mob
print x_plot[index_mob+1,:]


pl.ion()
fig,ax=pl.subplots()
print "len mob is",len_mob
while (index_route<len_route)|(index_mob<index_plot):
#	print  time_plot[index_mob+1],node_route[index_route,0]
	if (time_plot[index_mob+1]>node_route[index_route,0]):
	# route updation
		col_route=1
		while node_route[index_route,col_route]!=-1:
#			route_plot_indices[col]=node_route[index_route,col_route]
			col_route+=1
		pl.title("Route updation at "+str(node_route[index_route,0]))
#		print "Route updation at "+str(index_mob)+" "+str(time_plot[index_mob])
		index_route+=1
		time_neighb=node_route[index_route-1,0]
		continue

	elif (time_plot[index_mob+1]<node_route[index_route,0]):
	# position updation
		index_mob+=1
#		print (index_mob,time_plot[index_mob])
		pl.title("Position updation at "+str(time_plot[index_mob]))
#		print "Position updation at "+str(time_plot[index_mob])
		time_neighb=time_plot[index_mob]

	else:
		print "Time equal"
		index_mob+=1

	pl.scatter(x_plot[index_mob],y_plot[index_mob],s=100,c='g')
	for i, txt in enumerate(n):
	        pl.annotate(txt,(x_plot[index_mob, i]+10,y_plot[index_mob, i]+10))
# make axis labels
	pl.xlabel('x axis')
	pl.ylabel('y axis')
# set axis limits
	pl.xlim(0.0, 1500.0)
	pl.ylim(0.0, 1500.)
	ntemp=[int(sour_node[0])]
	for index_temp in range(1,col_route):
		ntemp.append(node_route[index_route,index_temp])
	ntemp.append(int(dest_node[0]))

	pl.plot(x_plot[index_mob,(ntemp[:col_route+2])],y_plot[index_mob,(ntemp[:col_route+2])])
	pl.scatter(x_plot[index_mob,(ntemp[:col_route+2])],y_plot[index_mob,(ntemp[:col_route+2])],s=100,c='r')
#	print_neighbors(time_neighb,x_plot[index_mob,:],y_plot[index_mob,:])
	pl.show()
	pl.pause(.00000000000000000000000000000000001)
	# show the plot on the screen
	pl.clf()


#################################################################################################
#################################################################################################
#################################################################################################


while (index_route<len_route)|(index_mob<index_plot):
#	print  time_plot[index_mob+1],node_route[index_route,0]
	if (time_plot[index_mob+1]>route_table_time[index_route]):
	# route updation
#		col_route=3
#		while route_table[index_route,col_route]!=-2:
#			route_plot_indices[col]=node_route[index_route,col_route]
#			col_route+=1
		pl.title("Route updation at "+str(route_table_time[index_route]))
#		print "Route updation at "+str(index_mob)+" "+str(time_plot[index_mob])
		index_route+=1
		time_neighb=route_table_time[index_route-1]
#		continue

	elif (time_plot[index_mob+1]<route_table_time[index_route]):
	# position updation
		index_mob+=1
#		print (index_mob,time_plot[index_mob])
		pl.title("Position updation at "+str(time_plot[index_mob]))
#		print "Position updation at "+str(time_plot[index_mob])
		time_neighb=time_plot[index_mob]

	else:
		print "Time equal"
		index_mob+=1

	pl.scatter(x_plot[index_mob],y_plot[index_mob],s=100,c='g')
	for i, txt in enumerate(n):
	        pl.annotate(txt,(x_plot[index_mob, i]+10,y_plot[index_mob, i]+10))
# make axis labels
	pl.xlabel('x axis')
	pl.ylabel('y axis')
# set axis limits
	pl.xlim(0.0, xRange)
	pl.ylim(0.0, yRange)
	ntemp=[]
	for index_temp in range(3,track_col[index_route-1]+1):
		ntemp.append(route_table[index_route-1,index_temp])
	print ntemp

	if (route_table[index_route-1,1]!=-1):
		pl.plot(x_plot[index_mob,(ntemp)],y_plot[index_mob,(ntemp)])
		pl.scatter(x_plot[index_mob,(ntemp)],y_plot[index_mob,(ntemp)],s=100,c='r')


	pl.title("Position updation at "+str(time_plot[index_mob])+" Packet:"+str(route_table[index_route-1,2]))
#	print_neighbors(time_neighb,x_plot[index_mob,:],y_plot[index_mob,:])
	pl.show()
	pl.pause(.05)
	# show the plot on the screen
	pl.clf()


