import numpy as np

'''  Defines a simple random walk, each step is choosen uniformly on [a,b]
- k is the number of steps
- a is a float 
- b is a float
- [a,b] must be a valid interval
'''

def urw(k,a=-1,b=+1):
    assert a < b, "[a,b] must be a valid interaval, a<b"

    x = np.random.uniform(low=a, high=b, size=k)
    rw = sum(x)
    return np.abs(rw)

if __name__=="__main__":
    ''' Example of usage '''
    k = 15
    print(urw(k))
