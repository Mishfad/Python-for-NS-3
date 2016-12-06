import numpy as np
import pylab as pl

def updateallexcept(index_pointer,node_index):
    for ind1 in range(nNodes):
        if (node_index!=ind1):
            x_plot[index_pointer,ind1]=x_plot[index_pointer-1,ind1]
            y_plot[index_pointer,ind1]=y_plot[index_pointer-1,ind1]



nNodes=5
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

mob=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/manet-routing-compare.mob',delimiter=' ',dtype=None)

mob_shape=mob.shape

#----------------------------------------------------------------------------------------
# Splitting the mobility data into time, node, postion and velocity
#----------------------------------------------------------------------------------------

time_mob=[]
node=[]
pos =[]
vel =[]
for row_index in range(0,mob_shape[0]):
    #for col_index in range(0,a_shape[1])
    time_mob.append(mob[row_index,0].split('='))
    node.append(mob[row_index,1].split('='))
    pos.append(mob[row_index,2].split('='))
    vel.append(mob[row_index,3].split('='))

#print "Mobility change time:\n",time_mob[:3]

#------------------------------------------------------
# position and velocity parsing
#------------------------------------------------------
## First filter out the strings vel,pos, and node
pos_temp=[x[1] for x in pos]
vel_temp=[x[1] for x in vel]
node_temp=[int(x[1]) for x in node]
time_temp=[x[1] for x in time_mob]

#print "Position x:y:z=\n",pos_temp[:3]


len_mob= len(pos_temp)

pos1=np.zeros((len_mob,3))
vel1=np.zeros((len_mob,3))
## Parse the pos and vel data into x,y, and z values
for index in range(len(pos_temp)):
    pos_str=''.join(pos_temp[index])
    pos1[index,:]=np.fromstring(pos_str,dtype=None,sep=":")
    vel_str=''.join(vel_temp[index])
    vel1[index,:]=np.fromstring(vel_str,dtype=None,sep=":")

print "Position x y z=\n",pos1[:3,:]

x_val=pos1[:,0]	# x axis
y_val=pos1[:,1]	# y axis

#print x_val[:4]

x_plot=np.zeros((len_mob,nNodes))
y_plot=np.zeros((len_mob,nNodes))
time_plot=[]

x_plot[0,:]=x_val[:nNodes]
y_plot[0,:]=y_val[:nNodes]

print node_temp[:3]

index=0
index_plot=0
x_plot[0,node_temp[0]]=x_val[0]
y_plot[0,node_temp[0]]=y_val[0]
time_plot.append(time_temp[0])
# Updation of the position with time
# if time is different, we update all the positions. Position of currentnode is updated first. Then, the remaining node positions are copied to this row using updateallexcept(.,.) function
while index < len_mob-1:
    if (time_temp[index+1]>time_temp[index]):
        # update index_plot pointer
        index_plot=index_plot+1
        x_plot[index_plot,node_temp[index+1]]=x_val[index+1]
        y_plot[index_plot,node_temp[index+1]]=y_val[index+1]
        time_plot.append(time_temp[index+1])
    #   updateallexcept(current_row_pointer,nodelocation_which_neednotbe_updated)
        updateallexcept(index_plot,node_temp[index+1])
    
    elif time_temp[index+1]==time_temp[index]:
        #if (node_temp[index+1]!=node_temp[index])|(index<nNodes):
            x_plot[index_plot,node_temp[index+1]]=x_val[index+1]
            y_plot[index_plot,node_temp[index+1]]=y_val[index+1]
    index=index+1

print index_plot
print "Time:\n",time_plot[:5]

time_mob=[np.fromstring(x,dtype=float,sep="n") for x in time_plot]
time_mob=[float(x)/1000000000 for x in time_mob]

print "Time in seconds after parsing: ",(time_mob)
len_mob=len(time_mob)
#print x_plot[index_plot-5:index_plot+1,:]

##############################################################################################
# Combining both
##############################################################################################

index_route=1
index_mob=0
index_final=0
print "lens",len_route,len_mob
print "Time route: ",node_route[:4,0]


#node_final[index_final,1:]=node_route[index_route,:]
#print "node ",node_final[index_final,1]
n=[x for x in range(nNodes)]
#print n
col_route=1
while node_route[0,col_route]!=-1:
	col_route+=1
#print col_route
ntemp=[3]
for index_temp in range(1,col_route):
	ntemp.append(int(node_route[index_route,index_temp]))
ntemp.append(0)
print "Node route: ",ntemp

#print index_mob
print x_plot[index_mob+1,:]



#########################################################################################
# 	Plotting
#########################################################################################


pl.ion()
fig,ax=pl.subplots()

while (index_route<len_route)|(index_mob<len_mob):
	if (time_mob[index_mob+1]>node_route[index_route,0]):
	# route updation
		col_route=1
		while node_route[index_route,col_route]!=-1:
#			route_plot_indices[col]=node_route[index_route,col_route]
			col_route+=1
		pl.title("Route updation at "+str(node_route[index_route,0]))
#		print "Route updation at "+str(index_mob)+" "+str(time_plot[index_mob])
		index_route+=1

	elif (time_mob[index_mob+1]<node_route[index_route,0]):
	# position updation
		index_mob+=1
		pl.title("Position updation at "+str(time_plot[index_mob]))
#		print "Position updation at "+str(time_plot[index_mob])

	else:
		print "TIme equal"

	pl.scatter(x_plot[index_mob],y_plot[index_mob],s=20)
	for i, txt in enumerate(n):
	        pl.annotate(txt,(x_plot[index_mob, i],y_plot[index_mob, i]))
# make axis labels
	pl.xlabel('x axis')
	pl.ylabel('y axis')
# set axis limits
	pl.xlim(0.0, 1000.0)
	pl.ylim(0.0, 1000.)
	ntemp=[3]
	for index_temp in range(1,col_route):
		ntemp.append(node_route[index_route,index_temp])
	ntemp.append(0)

	pl.plot(x_plot[index_mob,(ntemp[:col_route+2])],y_plot[index_mob,(ntemp[:col_route+2])])
	pl.show()
	pl.pause(.000001)
	# show the plot on the screen
	pl.clf()


