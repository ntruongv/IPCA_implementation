import numpy as np
import time
def ipca_restricted(Zts, Zt_cov, x_t, K, T, init = False, tol = 1e-5):
    np.random.seed(4134132)
    L = Zts.shape[1]
    N = Zts.shape[0]
    # initialization
    if True:
        mat = np.zeros((L,L))
        for i_t in range(T-1):
            #Zt = Zts[:,:,i_t]
            mat0 = x_t[:,i_t] #Zt.T.dot(rs[:,i_t+1])
            mat += mat0.dot(mat0.T)/(Zts.shape[0])**2
        _, eig_vec = np.linalg.eigh(mat)
        Gamma_b_0 = eig_vec[:,np.arange(-1,-K-1,-1)]
    else: Gamma_b_0 = np.random.rand(L,K)

    # Gamma_b_0 = np.array(Gamma_b_init, copy=True)
    Gamma_b_1 = np.array(Gamma_b_0, copy=True)
    F_0 = np.random.rand(K,T)
    F_1 = np.array(F_0, copy=True)
    iter = 0
    # tol = 1e-3
    start_time = time.time()
    max_iter = 1001
    max_abs_delta_F = np.zeros(max_iter)
    max_abs_delta_Gamma_b = np.zeros(max_iter)
    cond_F = np.zeros(max_iter)
    cond_Gamma_b = np.zeros(max_iter)
    cond_threshold = 1e5
    while iter < max_iter:
    # while iter < 1:
        # update F
        for i_t in range(T-1):
            #Zt = Zts[:,:,i_t]
            mat = Gamma_b_0.T.dot(Zt_cov[:,:,i_t]).dot(Gamma_b_0)
            F_1[:,i_t+1] = np.linalg.inv(mat).dot(Gamma_b_0.T).dot(x_t[:,i_t])
            temp = np.linalg.cond(mat)
            cond_F[iter] += temp
            if temp >= cond_threshold:
                print("*** WARNING: det >=", cond_threshold, ". Instability in F")
        cond_F[iter] /= (T-1)

        # update Gamma_b
        mat_1 = np.zeros((L*K,L*K))
        mat_2 = np.zeros(L*K)
        for i_t in range(T-1):
            #Zt = Zts[:,:,i_t]
            #mat_0 = np.kron(Zt, F_1[:,i_t+1].T)
            cur_F = F_1[:,i_t+1].reshape(-1,1)
            mat_1 += np.kron(Zt_cov[:,:,i_t], (cur_F.dot(cur_F.T)) ) # mat_0.T.dot(mat_0)
            mat_2 += np.kron(x_t[:,i_t], cur_F.flatten()) #mat_0.T.dot(rs[:,i_t+1])
        Gamma_b_1 = np.linalg.inv(mat_1).dot(mat_2).reshape(L,K)
        cond_Gamma_b[iter] = np.linalg.cond(mat_1)
        # if cond_Gamma_b[iter] >= cond_threshold:
        #     print("*** WARNING: det >=", cond_threshold, ". Instability in Gamma")

        max_abs_delta_F[iter] = np.sum(np.abs(F_0[:,1:]-F_1[:,1:])) / ((T-1) * K)
        max_abs_delta_Gamma_b[iter] = np.sum(np.abs(Gamma_b_0-Gamma_b_1)) / (L * K)
        # max_abs_delta_F[iter] = np.amax(np.abs(F_0[:,1:]-F_1[:,1:]))
        # max_abs_delta_Gamma_b[iter] = np.amax(np.abs(Gamma_b_0-Gamma_b_1))

        if max_abs_delta_F[iter] <= tol and max_abs_delta_Gamma_b[iter] <= tol:
            # print('iter', iter, '\tmax_abs_delta_F', max_abs_delta_F[iter],
            #       '\tmax_abs_delta_Gamma_b', max_abs_delta_Gamma_b[iter])
            break
        # if not iter % 100:
        #     print('iter', iter, '\tmax_abs_delta_F', max_abs_delta_F[iter],
        #           '\tmax_abs_delta_Gamma_b', max_abs_delta_Gamma_b[iter])
        Gamma_b_0 = np.array(Gamma_b_1, copy=True)
        F_0 = np.array(F_1, copy=True)

        iter += 1
    end_time = time.time()

    # normalization
    # idx = np.argsort(np.diag(F_1[:,1:].dot(F_1[:,1:].T)))
    # F_1 = F_1[idx,:]
    U,S,V = np.linalg.svd(Gamma_b_1, full_matrices=False)
    Gamma_b_1 = Gamma_b_1.dot(V.T).dot(np.diag(1/S))
    F_1 = np.diag(S).dot(V).dot(F_1)
    U2, S2, V2 = np.linalg.svd(F_1.dot(F_1.T))
    F_1 = (U2.T).dot(F_1)
    Gamma_b_1 = Gamma_b_1.dot(U2)

    # print("Elapsed time", end_time - start_time)
    return F_1, Gamma_b_1, max_abs_delta_F, max_abs_delta_Gamma_b, cond_F, cond_Gamma_b
