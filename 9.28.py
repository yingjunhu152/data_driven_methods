#Program1 computing SVD
import numpy as np
X=np.random.rand(5,3)
U,S,VT=np.linalg.svd(X,full_matrices=True)#Full SVD
Uhat,Shat,VThat=np.linalg.svd(X,full_matrices=False) #Economy SVD

#Program2 use SVD to compress image
import matplotlib.pyplot as plt
from matplotlib.image import imread
image_path='C:/Users/asdfg/Pictures/Cyberpunk 2077/head.png'
A=imread(image_path)
X=np.mean(A,-1)

img=plt.imshow(X)
plt.show()
#take SVD
U,S,VT=np.linalg.svd(X,full_matrices=False)
S=np.diag(S)
#Approximate matrix with truncated SVD for various ranks r
for r in [5,20,100]:
    Xapprox=U[:,:r]@S[0:r,:r]@VT[:r,:]
    img = plt.imshow(Xapprox)
    plt.show()

#Program3 least-aquares fit of noisy data
x=3
a=np.arange(-2,2,0.25)
a=a.reshape(-1,1)
b=x*a+np.random.randn(*a.shape)
plt.plot(a,x*a) #true relationship
plt.plot(a,b,'x')
#compute the least-squares approximation with SVD
U,S,VT=np.linalg.svd(a,full_matrices=False)
xtilde=VT.T@np.linalg.inv(np.diag(S))@U.T@b          #least-square fit
plt.plot(a,xtilde*a)
plt.show()

#Program4 pca on noisy cloud of data
#generate noisy cloud of data
xC=np.array([2,1])      #center of data
sig=np.array([2,0.5])   #principal axes
theta=np.pi/3           #rotate cloud by pi/3
R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
nPoints=10000
X=R@np.diag(sig)@np.random.randn(2,nPoints)+np.diag(xC)@np.ones(2,nPoints)
Xavg=np.mean(X,axis=1)
B=X-np.tile(Xavg,(nPoints,1)).T
#SVD and PCA
U,S,VT=np.linalg.svd(B/np.sqrt(nPoints),full_matrices=False)
theta=2*np.pi*np.arange(0,1,0.01)
