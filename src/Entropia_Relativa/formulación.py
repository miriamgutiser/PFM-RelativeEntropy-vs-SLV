import numpy as np

#Entropia relativa:
def relative_entropy(p,q):
    D = 0
    v = len(q)

    for r in range(v):
        D = D + p[r] * np.log(p[r] / q[r])   
    return D
     
p = (0.3, 0.3, 0.4)        
q = (1/3, 1/3, 1/3) 
print(relative_entropy(p, q))



