import numpy as np
packtoseq=np.genfromtxt('/home/mishfad/Documents/NS-3/ns-allinone-3.25/ns-3.25/packetidToSeq.txt',delimiter=',',dtype=str)

print packtoseq[:5]
