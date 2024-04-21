# Nume studenti: Trandafir Victor si Bara Andrei
# DVS: înlăturarea de zgomot din imagini alb-negre
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio.v2 as imageio

#Conversie în alb-negru
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

Img =  mpimg.imread('imagine.png')

img_gray= rgb2gray(Img)

#Bidiagonalizarea Golub-Kahan si alg. DVS aferent (incomplet, din suport curs)
def JQ(A):
    m = A.shape[0]
    n = A.shape[1]

    if(m < n):
        m, n = n, m

    S = np.zeros(n)

    U = np.eye(m)
    V = np.eye(n)

    for k in range (n):
        #Calcul reflector Householder
        norm = 0
        for i in range(k, m):
            norm += A[i][k]**2

        norm = np.sqrt(norm)

        sigma = np.sign(A[k][k]) * norm

        u = U[k, :]
        u[k] = A[k][k] + sigma

        for i in range (k + 1, m):
            u[i] = A[i][k]
        
        beta = u[k] * sigma
        #Aplicare reflector Householder
        A[k][k] = -sigma

        for j in range (k + 1, n):
            tau = 0
            for i in range(k, m):
                tau += u[i] * A[i][j]

            tau /= beta
            
            for i in range(k, m):
                A[i][j] -= tau * u[i]

        for i in range (m):
            tau = 0
            for k in range(k, m):
                tau += U[i][j] * u[j]

            tau /= beta

            for j in range (k, m):
                U[i][j] -= tau * u[j]

        if k < n - 2:
            #Calcul reflector Householder
            norm = 0
            for i in range(k + 1, n):
                norm += A[k][j]**2

            norm = np.sqrt(norm)

            sigma = np.sign(A[k][k + 1]) * norm
            v = V[:, k + 1]
            v[k + 1] = A[k][k + 1] + sigma

            for j in range (k + 2, n):
                v[j] = A[k][j]
            
            gamma = v[k + 1] * sigma
            #Aplicare reflector Householder
            A[k][k + 1] = -sigma
            
            for i in range(k + 1, m):
                tau = 0
                for j in range (k + 1, n):
                    tau += A[i][j] * v[j]

                tau /= gamma

                for j in range(k + 1, n):
                    A[i][j] -= tau * v[j]
                
    for i in range (n - 1):
        S[i] = A[i][i]

    S[n - 1] = A[n - 1][n - 1]

    return U, S, V, n, m

#Ordonarea valorilor si vectorilor singulari
def Ord(U, S, V, n):
    ord = 0
    i = -1
    while ord == 0 & i < n - 1:
        ord = 1

        for j in range(n - 2, i, -1):
            if(S[j] < S[j + 1]):

                S[[j, j + 1]] = S[[j + 1, j]]

                U[:, [j, j + 1]] = U[:, [j + 1, j]]
                V[:, [j, j + 1]] = V[:, [j + 1, j]]

                ord = 0
        
        i += 1
   
    return U, S, V

def DVS(A):
    U, S, V, n, m = JQ(A)

    for j in range (n):
        if S[j] != 0:
            V[: , j] *= np.sign(S[j])

        S[j] = abs(S[j])
    
    U, S, V = Ord(U, S, V, n)
    return U, S, V

#Scriere alternativă a alg. DVS bazat pe Golub-Kahan
def golub_kahan(A):
    m, n = A.shape
    U = np.copy(A)
    V = np.eye(n)
    for i in range(min(m, n)):
        x = U[i:, i]
        e = np.zeros(len(x))
        e[0] = np.linalg.norm(x)
        v = np.sign(x[0]) * np.linalg.norm(x) * e - x
        v = v / np.linalg.norm(v)
        U[i:, i:] -= 2 * np.outer(v, np.dot(v, U[i:, i:]))
        V[i:, i:] -= 2 * np.outer(v, np.dot(v, V[i:, i:]))
        if i < m - 1:
            x = U[i, i+1:]
            e = np.zeros(len(x))
            e[0] = np.linalg.norm(x)
            v = np.sign(x[0]) * np.linalg.norm(x) * e - x
            v = v / np.linalg.norm(v)
            U[i:, i+1:] -= 2 * np.outer(np.dot(U[i:, i+1:], v), v)
    B = U.dot(V.T)
    return B, U, V

#Adaugare zgomot alb Gaussian cu diferite deviații standard
def adaugaNoise(X, s):
    noise = np.random.normal(0, s, X.shape)
    X_noisy = X + noise
    return X_noisy

img_noised1 = adaugaNoise(img_gray, 0.05)
img_noised2 = adaugaNoise(img_gray, 0.1)
img_noised3 = adaugaNoise(img_gray, 0.15)
img_noised4 = adaugaNoise(img_gray, 0.25)

img_noisedforDVS = adaugaNoise(img_gray, 0.05)

U, S, V = DVS(img_noisedforDVS)
img_denoised1 = U@np.diag(S)@V
mpimg.imsave('denoisedGK1.png', img_denoised1, cmap=plt.get_cmap('gray'))

S, U, V = golub_kahan(img_noised1)
img_denoised1 = U@S@V
mpimg.imsave('denoisedGK2.png', img_denoised1, cmap=plt.get_cmap('gray'))

def RMSE(U, S, V, p, original):
    RMSE=np.zeros(p)
    for k in range(p):
        RMSE[k] = 0
        reconstruit = U[:,:k] @ np.diag(S[:k]) @ V[:k,:]
        RMSE[k] += np.mean(abs(reconstruit - original)**2)
        RMSE[k] = np.sqrt(RMSE[k])
    return RMSE

U1, S1, V1 = np.linalg.svd(img_noised1)
p = S1.size
RMSE1 = RMSE(U1, S1, V1, p, img_gray)

U2, S2, V2 = np.linalg.svd(img_noised2)
RMSE2 = RMSE(U2, S2, V2, p, img_gray)

U3, S3, V3 = np.linalg.svd(img_noised3)
RMSE3 = RMSE(U3, S3, V3, p, img_gray)

U4, S4, V4 = np.linalg.svd(img_noised4)
RMSE4 = RMSE(U4, S4, V4, p, img_gray)

plt.plot(np.linspace(0, p, p),RMSE1)
plt.plot(np.linspace(0, p, p),RMSE2)
plt.plot(np.linspace(0, p, p),RMSE3)
plt.plot(np.linspace(0, p, p),RMSE4)
plt.title("RMSE v aproximarea imaginii la un anumit rang") 
plt.ylabel('RMSE')
plt.xlabel('Rang')
plt.legend(["\u03C3 = 5%","\u03C3 = 10%","\u03C3 = 15%","\u03C3 = 25%"])
plt.show()

plt.figure(1)

plt.subplot(1,2,1)
plt.tight_layout(pad=1.0)
plt.subplots_adjust(left=0.1, right = 0.9, wspace=1, hspace=1)
plt.imshow(img_noised1, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.title(f"Cu zgomot")

plt.subplot(1,2,2)
plt.tight_layout(pad=1.0)
plt.subplots_adjust(left=0.1, right = 0.9, wspace=1, hspace=1)
img_denoised1 = U1[:,:np.argmin(RMSE1)] @ np.diag(S1[:np.argmin(RMSE1)]) @ V1[:np.argmin(RMSE1),:]
plt.imshow(img_denoised1, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.title(f"Fără zgomot")

plt.suptitle("\u03C3 = 5% Zgomot")

plt.show()

plt.figure(2)

plt.subplot(1,2,1)
plt.tight_layout(pad=1.0)
plt.subplots_adjust(left=0.1, right = 0.9, wspace=1, hspace=1)
plt.imshow(img_noised2, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.title(f"Cu zgomot")

plt.subplot(1,2,2)
plt.tight_layout(pad=1.0)
plt.subplots_adjust(left=0.1, right = 0.9, wspace=1, hspace=1)
img_denoised1 = U2[:,:np.argmin(RMSE2)] @ np.diag(S2[:np.argmin(RMSE2)]) @ V2[:np.argmin(RMSE2),:]
plt.imshow(img_denoised1, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.title(f"Fără zgomot")

plt.suptitle("\u03C3 = 10% Zgomot")

plt.show()

plt.figure(3)

plt.subplot(1,2,1)
plt.tight_layout(pad=1.0)
plt.subplots_adjust(left=0.1, right = 0.9, wspace=1, hspace=1)
plt.imshow(img_noised3, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.title(f"Cu zgomot")

plt.subplot(1,2,2)
plt.tight_layout(pad=1.0)
plt.subplots_adjust(left=0.1, right = 0.9, wspace=1, hspace=1)
img_denoised1 = U3[:,:np.argmin(RMSE3)] @ np.diag(S3[:np.argmin(RMSE3)]) @ V3[:np.argmin(RMSE3),:]
plt.imshow(img_denoised1, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.title(f"Fără zgomot")

plt.suptitle("\u03C3 = 15% Zgomot")

plt.show()

plt.figure(4)

plt.subplot(1,2,1)
plt.tight_layout(pad=1.0)
plt.subplots_adjust(left=0.1, right = 0.9, wspace=1, hspace=1)
plt.imshow(img_noised4, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.title(f"Cu zgomot")

plt.subplot(1,2,2)
plt.tight_layout(pad=1.0)
plt.subplots_adjust(left=0.1, right = 0.9, wspace=1, hspace=1)
img_denoised1 = U4[:,:np.argmin(RMSE4)] @ np.diag(S4[:np.argmin(RMSE4)]) @ V4[:np.argmin(RMSE4),:]
plt.imshow(img_denoised1, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.title(f"Fără zgomot")

plt.suptitle("\u03C3 = 25% Zgomot")

plt.show()
