import numpy as np
import pylab as pl

def updateallexcept(index_pointer,node_index):
    for ind1 in range(nNodes):
        if (node_index!=ind1):
            x_plot[index_pointer,ind1]=x_plot[index_pointer-1,ind1]
            y_plot[index_pointer,ind1]=y_plot[index_pointer-1,ind1]



nNodes=10
a=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/first_routing.txt',delimiter=',',dtype=str)
a_shape=a.shape

#print a.shape
#print a[:3]

time=[]
node_addr1=[]
dest_addr1 =[]
source_addr1 =[]
packet1=[]

#print time
#--------------------------------------------------------------
# Splitting the data into time, current node, destination, source and packet tag
#--------------------------------------------------------------
for row_index in range(0,a_shape[0]):
    time.append(float(a[row_index,0]))
    node_addr1.append(a[row_index,1].split(':'))
    dest_addr1.append(a[row_index,2].split(':'))
    source_addr1.append(a[row_index,3].split(':'))
    packet1.append(a[row_index,4].split(':'))

print time[:3]
#print node_addr1[:3]
#print dest_addr1[:3]
#print source_addr1[:3]
#print packet1[:3]


#print (node,pos)
#------------------------------------------------------
# parsing to separate tag and data
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

len1= len(time)
#print "Len1=",len1

#------------------------------------------------------
# Separate the address to get node id
#------------------------------------------------------

print "Before splitting; forwarding node address:\n",node_addr[:3]
node_addr  = [x.split('.') for x in node_addr  ]
dest_addr  = [x.split('.') for x in dest_addr  ]
source_addr= [x.split('.') for x in source_addr]


print "After splitting; forwarding node address:\n",node_addr[:3]
#print "After splitting; Destinatin node address:\n",dest_addr[:3]
#print "After splitting; source node address:\n" ,source_addr[:3]

forw_node=[int(x[3])-1 for x in node_addr]
dest_node=[int(x[3])-1 for x in dest_addr]
sour_node=[int(x[3])-1 for x in source_addr]
print "Time in seconds:\n",time[:9]
print "Forwdg nodes are:\n",forw_node[:9]
print "Destin nodes are:\n",dest_node[:9]
print "Source nodes are:\n",sour_node[:9]
print "Packet id:\n",packet[:9]

print "Len1=",len1

node=np.zeros((len1,8))
index=1
i_row=0
i_col=2

#initialization
node[i_row,0]=time[0]
node[i_row,1]=forw_node[0]
# 2 pointers are used. row_pointer(i_row) to update the row and col_pointer(i_col) to update col. i_col starts at 2. node[.,0] stores the time, node[.,1] stores the 


while index<len1:
	if (packet[index]==packet[index-1]):
		node[i_row,i_col]=forw_node[index]
		index=index+1
		i_col=i_col+1

# if packet indices are different, increment the row_pointer and set col_pointer to 2
	else:
		node[i_row,i_col]=-1	# -1 shows the end of the row; used later to traverse through the row
		i_col=2
		i_row=i_row+1
		node[i_row,0]=time[index]		
		node[i_row,1]=forw_node[index]
		index+=1
		
print "Final value of i_row=",i_row
print "Source is always",sour_node[0]
print "Destination is always",dest_node[0]
print node[:100,:]












#-----------------------------------------------------
# Plotting
#-----------------------------------------------------


