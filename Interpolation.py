import math
import numpy as np


# Orthogonal polynomial fitting schema (returns array of alpha coefficients and orthogonal polynomial for x_i)
def orthogonalPolynomialFit(m,x,f):
    n = len(x) - 1
    u = np.zeros([m+1,n+2])
    s = np.zeros([n+1]) # 
    g = np.zeros([n+1]) # <x * u_k|u_k> / <u_k|u_k>
    h = np.zeros([n+1]) # <x * u_k-1|u_k-1> / <u_k-1|u_k-1>

    # Check and fix the order of the curve
    if m > n:
        m = n
        print("The highest power n is {}, adjusted m to order {}".format(n,m))

    # Set up zeroth order polynomial

    for i in range(0,n+1):
        u[0][i] = 1
        stmp = u[0][i] * u[0][i]
        s[0] += stmp
        g[0] += x[i]*stmp
        u[0][n+1] += u[0][i]*f[i]
    
    g[0] = g[0]/s[0]
    u[0][n+1] = u[0][n+1]/s[0]

    # Set up the first-order polynomial
    for i in range(0,n+1):
        u[1][i] = x[i]*u[0][i]-g[0]*u[0][i]
        s[1] += u[1][i]*u[1][i]
        g[1] += x[i]*u[1][i]*u[1][i]
        h[1] += x[i]*u[1][i]*u[0][i]
        u[1][n+1] += u[1][i]*f[i]
    
    g[1] = g[1]/s[1]
    h[1] = h[1]/s[0]
    u[1][n+1] = u[1][n+1]/s[1]

    # Obtain the higher-order polynomials recursively
    if m >= 2:
        for i in range(1,m): # java code has a range of 1,m but when I run it my fit dun goofs
            for j in range(0,n+1):
                u[i+1][j] = x[j]*u[i][j]-g[i]*u[i][j]-h[i]*u[i-1][j]
                s[i+1] += u[i+1][j]*u[i+1][j]
                g[i+1] += x[j]*u[i+1][j]*u[i+1][j]
                h[i+1] += x[j]*u[i+1][j]*u[i][j]
                u[i+1][n+1] += u[i+1][j]*f[j]
            
            g[i+1] = g[i+1]/s[i+1]
            h[i+1] = h[i+1]/s[i]
            u[i+1][n+1] = u[i+1][n+1]/s[i+1]

    return u

# first order polynomial least squares fit (returns coefficients of polynomial fit)
def leastSquaresFit(x,f):
    n = len(x) - 1
    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    for i in range(0,n+1):
        c1 += x[i]
        c2 += x[i]*x[i]
        c3 += f[i]
        c4 += f[i]*x[i]

    g = c1*c1-c2*(n+1)
    a0 = (c1*c4-c2*c3)/g
    a1 = (c1*c3-c4*(n+1))/g

    return a0, a1

# natural spline interpolation algorithm. Use this when you have data and you need to approximate the values in between the given points (returns approximated points)
class Spline:
    def __init__(self): # empty init
        pass
    # Reads the data file and outputs two numpy arrays
    def readFile(self,datafile='xy.data'):
        f = open(datafile, "r")
        non_empty_lines =[line.strip("\n") for line in f if line != "\n" and len(line) != 0]
        if non_empty_lines[0] == 'x,y': # ignore the column names in the file and carry on 
            non_empty_lines.pop(0)
        n = len(non_empty_lines) # set the n automatically
        # create zero spline approximation arrays
        xi = np.zeros([n])
        fi = np.zeros([n])
        # read in datapoints xi and fi
        # iterate trhough the data, split the text up, and make the arrays for spline interpolator
        i = 0
        for line in non_empty_lines:
            line = line.split(',')
            xi[i], fi[i] = float(line[0]), float(line[1])
            i+=1
        # return the data
        return xi, fi

    # Creates a spline fit to the data 
    def fit(self,m=100,xi=np.zeros([10]),fi=np.zeros([10]),a_0=0,a_n=0):
        n = len(xi) - 1
        p2 = np.zeros([n+1]) # second derivative of cubics
        p2 = self.cubicSpline(x=xi,f=fi) # coefficients of each polynomial
        p2[0] = a_0
        p2[n] = a_n

        # Find the approximation of the function
        h = (xi[n] - xi[0])/m # distance between points
        x = xi[0] # starting x
        

        a = np.zeros([m]) # approximation
        x1 = np.zeros([m]) # fitted x array

        for i in range(1,m):
            x+=h # step x by h
            # Find the interval where x resides
            k = 0 # k = 0 for counter start
            dx = x-xi[0]
            while dx>0: # iterate until point is found
                k+=1
                dx = x-xi[k]
            k-=1
            
            # Find the value of the function f(x)
            dx = xi[k+1]-xi[k] # steps
            alpha = p2[k+1]/(6*dx) # coef 1
            beta = -p2[k]/(6*dx) # coef 2 
            gamma = fi[k+1]/dx - dx*p2[k+1]/6 # coef 3 
            eta = dx*p2[k]/6 - fi[k]/dx # coef 4 

            # approximated cubic polynomial
            f = alpha*(x-xi[k])*(x-xi[k])*(x-xi[k]) + beta*(x-xi[k+1])*(x-xi[k+1])*(x-xi[k+1]) + gamma*(x-xi[k]) + eta*(x-xi[k+1])
            #print(x,f)
            # append the fucntion value for each value of x that is a step of the cubic spline algorithm
            a[i] = f
            x1[i] = x
            

        return x1,a
    # Method to perform the cubic spline approximation
    def cubicSpline(self,x,f):
        n = len(x)-1
        p = np.zeros([n+1])
        d = np.zeros([n-1])
        b = np.zeros([n-1])
        c = np.zeros([n-1])
        g = np.zeros([n])
        h = np.zeros([n])

        # Assign the intervals and function differences

        for i in range(n):
            h[i] = x[i+1]-x[i] 
            g[i] = f[i+1]-f[i]
        
        # Evaluate the coefficient matrix elements

        for i in range(0,n-1):
            d[i] = 2*(h[i+1]+h[i])
            b[i] = 6*(g[i+1]/h[i+1]-g[i]/h[i])
            c[i] = h[i+1]


        # Obtain second order derivatives
        g = self.tridiagonalLinearEq(d, c, c, b)
        for i in range(1,n):
            p[i] = g[i-1]
        
        return p # return the 2nd order derivatives for each polynomial segment

    # performs the LU decomposition to solve for the spline's p values
    def tridiagonalLinearEq(self,d, e, c, b):
        m = len(b)
        w = np.zeros([m])
        y = np.zeros([m])
        z = np.zeros([m])
        v = np.zeros([m-1])
        t = np.zeros([m-1])

        # Evaluate the elements in the LU decomposition
        w[0] = d[0]
        v[0] = c[0]
        t[0] = e[0]/w[0]
        
        for i in range(1,m-1):
            w[i]  = d[i]-v[i-1]*t[i-1]
            v[i]  = c[i]
            t[i]  = e[i]/w[i]

        w[m-1]  = d[m-1]-v[m-2]*t[m-2]


        # Forward substitution to get y
        y[0] = b[0]/w[0]
        for i in range(1,m):
            y[i] = (b[i] - v[i-1]*y[i-1])/w[i]
        
        # Backward substitution to obtain z
        z[m-1] = y[m-1]
        i=m-2
        while i >=0: # potential source of error ----------------
            z[i] = y[i]-t[i]*z[i+1]
            i-=1
        
        return z

# uniform radnom number generator (returns float)
def ranf():
    global seed
    a = 16807
    c = 2147483647
    q = 127773
    r = 2836
    cd = c
    h = seed/q
    l = seed%q
    t = a*l - r*h
    seed = t if t > 0 else c+t
    return seed/cd

# random number generator that follows a gaussian distribution (returns array with two values)
def randg():
    x = np.zeros([2])
    r1 = -math.log(1-ranf())
    r2 = 2*math.pi*ranf()
    r1 = math.sqrt(2*r1)
    x[0] = r1*math.cos(r2)
    x[1] = r1*math.sin(r2)
    return x