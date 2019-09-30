import numpy as np
import time
def ipca_unrestricted(Zts, Zt_cov, x_t, K, T, init = False, tol = 1e-6):
    L = Zts.shape[1]
#    N = Zts.shape[0]
    # initialization
    Gamma_0 = np.random.rand(L, 1+K)
    Gamma_1 = np.array(Gamma_0, copy=True)
    Gamma_a = Gamma_0[:,0]
    Gamma_b = Gamma_0[:,1:]
    F_0 = np.random.rand(K, T)
    F_1 = np.array(F_0, copy=True)
    tol = 1e-6
    start_time = time.time()
    max_iter = 1001
    max_abs_delta_F = np.zeros(max_iter)
    max_abs_delta_Gamma = np.zeros(max_iter)
    for iter in range(max_iter):
        # update F
        for i_t in range(T-1):
            #Zt = Zts[:,:,i_t]
            mat = Gamma_b.T.dot(Zt_cov[:,:,i_t]).dot(Gamma_b)
            F_1[:,i_t+1] = np.linalg.inv(mat).dot(Gamma_b.T).dot(x_t[:,i_t] - Zt_cov[:,:,i_t].dot(Gamma_a) ) #.dot(Zt.T).dot(rs[:,i_t+1]-Zt.dot(Gamma_a))
        # update Gamma_b
        tilde_F_1 = np.concatenate((np.ones((1,T)), F_1), axis = 0) # (1+K, T)
        mat_1 = np.zeros((L*(1+K),L*(1+K)))
        mat_2 = np.zeros(L*(1+K))
        for i_t in range(T-1):
            cur_F = tilde_F_1[:,i_t+1].reshape(-1,1)
            mat_1 += np.kron(Zt_cov[:,:,i_t], (cur_F.dot(cur_F.T)) ) # mat_0.T.dot(mat_0)
            mat_2 += np.kron(x_t[:,i_t], cur_F.flatten()) #mat_0.T.dot(rs[:,i_t+1])
            #Zt = Zts[:,:,i_t]
            #mat_0 = np.kron(Zt, tilde_F_1[:,i_t+1].T)
            #mat_1 += mat_0.T.dot(mat_0)
            #mat_2 += mat_0.T.dot(rs[:,i_t+1])
        Gamma_1 = np.linalg.inv(mat_1).dot(mat_2).reshape(L,1+K)
        Gamma_a = Gamma_1[:,0]
        Gamma_b = Gamma_1[:,1:]

#        Gamma_1[:,0] = Gamma_a

        max_abs_delta_F[iter] = np.sum(np.abs(F_0[:,1:]-F_1[:,1:])) / ((T-1) * K)
        max_abs_delta_Gamma[iter] = np.sum(np.abs(Gamma_0-Gamma_1)) / (L * (1+K))
        if max_abs_delta_F[iter] <= tol and max_abs_delta_Gamma[iter] <= tol:
            print('iter', iter, '\tmax_abs_delta_F', max_abs_delta_F[iter],
                  '\tmax_abs_delta_Gamma', max_abs_delta_Gamma[iter])
            break
        if not iter % 100: 
            print('iter', iter, '\tmax_abs_delta_F', max_abs_delta_F[iter],
                  '\tmax_abs_delta_Gamma', max_abs_delta_Gamma[iter])
        Gamma_0 = np.array(Gamma_1, copy=True)
        F_0 = np.array(F_1, copy=True)

        iter += 1
    end_time = time.time()
    # normalization
    U,S,V = np.linalg.svd(Gamma_1[:,1:], full_matrices=False)
    Gamma_b = Gamma_1[:,1:].dot(V.T).dot(np.diag(1/S))
    Gamma_a = Gamma_a-Gamma_b.dot(np.linalg.inv(Gamma_b.T.dot(Gamma_b))).dot(Gamma_b.T).dot(Gamma_a)
    F_1 = np.diag(S).dot(V).dot(F_1)


    print("Elapsed time", end_time - start_time)
    return F_1, Gamma_b, Gamma_a, max_abs_delta_F, max_abs_delta_Gamma
