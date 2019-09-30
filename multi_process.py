from multiprocessing import Pool
import time

class solve_for_Gammba_b:
    def __init__(self, Zts, F, rs):
        self.Zts = Zts
        self.F = F
        self.rs = rs
    def __call__(self, i_t):
        Zt = self.Zts[:,:,i_t]
        mat_0 = np.kron(Zt, self.F[:,i_t+1].T)
        mat_1 = mat_0.T.dot(mat_0)
        mat_2 = mat_0.T.dot(rs[:,i_t+1])
        return (mat_1, mat_2)

class par_helper:
    def __init__(self, N_thread=4):
        print("Initializing pool N_thread =", N_thread)
        self.pool = Pool(N_thread)
    def __call__(self, Zts, F, rs, idx):
        return self.pool.map(solve_for_Gammba_b(Zts, F, rs), idx)
    def __del__(self):
        print("Deleting pool N_thread =", N_thread)
        self.pool.close()

# test
N_repeat = 10
N_thread_list = range(1,6)
exec_times = [0]*len(N_thread_list)
for i in range(len(N_thread_list)):
    N_thread = N_thread_list[i]
    ph = par_helper(N_thread)
    for j in range(N_repeat):
        start_time = time.time()
        ph(Zts, F_1, rs, range(T-1));
        end_time = time.time()
        exec_times[i] += end_time - start_time
    exec_times[i] /= N_repeat
    del ph
    print("N_thread", N_thread, "time", exec_times[i])
