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

'''#Program4 pca on noisy cloud of data
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
theta=2*np.pi*np.arange(0,1,0.01)'''

#Program5 compare different threshold approaches on noisy low rank data
t=np.arange(-3,3,0.01)
Utrue=np.array([np.cos(17*t)*np.exp(-t**2),np.sin(11*t)]).T
Strue=np.array([[2,0],[0,0.5]])
Vtrue=np.array([np.sin(5*t)*np.exp(-t**2),np.cos(13*t)]).T

X=Utrue@Strue@Vtrue.T
plt.imshow(X)
plt.show()
#contaminate signal with noise
sigma=1
Xnoisy=X+sigma*np.random.randn(*X.shape)
plt.imshow(Xnoisy)
plt.show()
#truncate using optimal hard threshold
U,S,VT=np.linalg.svd(X,full_matrices=False)
N=Xnoisy.shape[0]
cutoff=(4/np.sqrt(3)*np.sqrt(N)*sigma)#hard threshold
r=np.max(np.where(S>cutoff))
Xclean=U[:,:(r+1)]@np.diag(S[:r+1])@VT[:(r+1),:]
plt.imshow(Xclean)
plt.show()
cds=np.cumsum(S)/np.sum(S)
r90=np.min(np.where(cds>0.9))
X90=U[:,:(r90+1)]@np.diag(S[:r90+1])@VT[:(r90+1),:]
plt.imshow(X90)
plt.show()


#Program randomized SVD alogrithm
def rSVD(X,r,q,p):
    #sample column space of X with P 
    ny=X.shape[1]
    P=np.random.randn(ny,r+p)
    Z=X@P
    for k in range(q):
        Z=X@(X.T@Z)
    Q,R=np.linalg.qr(Z,mode='reduced')
    #compute svd on Y=Q.T@X
    Y=Q.T@X#Y is the projection of X on its column space
    UY,S,VT=np.linalg.svd(Y,full_matrices=False)
    U=Q@UY
    return U,S,VT