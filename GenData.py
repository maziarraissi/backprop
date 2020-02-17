import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

if __name__ == "__main__": 
    
    N = 100
    noise = 0.0
    
    # Generate Training Data   
    def f(x):
        return x*np.sin(4.0*np.pi*x)
    
    lb = -1.0*np.ones((1,1))
    ub =  1.0*np.ones((1,1))
    
    X_train = lb + (ub-lb)*lhs(1, N)
    Y_train = f(X_train) + noise*np.random.randn(N,1)
    
    N_test = 1000
    X_test = lb + (ub-lb)*np.linspace(0,1,N_test)[:,None]
    Y_test = f(X_test)

    plt.figure()
    plt.plot(X_train,Y_train,'kx')
    plt.plot(X_test,Y_test,'b-')
                
    np.savetxt('training_data.csv', np.concatenate((X_train,Y_train),axis=1))
    np.savetxt('test_data.csv', np.concatenate((X_test,Y_test),axis=1))