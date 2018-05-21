import operator,functools,copy,random,math,cmath

class Matrix:
    #%%Initialization and Representation
    
    def __init__(self,M,N):
        """initializes instance of class; attributes include entries as rows, # of rows, and # of columns"""
        self.rows=[[0]*N for n in range(M)]
        self.m=M
        self.n=N
        
    def __getitem__(self, idx):
        """returns the rows of given matrix"""
        return self.rows[idx]
 
    def __setitem__(self, idx, row):
        """sets the rows of given matrix and reassigns instance attributes"""
        self.rows[idx] = row
        self.m=len(self.rows)
        self.n=len(row)
   
    def __repr__(self):
        """list representation of given matrix"""
        return str(self.rows)
    
    def __str__(self):
        """string representation of given matrix with each row being a newline"""
        return '\n'.join([" ".join([str(item) for item in row]) for row in self.rows])+'\n'
    #%% Boolean Operations
    
    def __bool__(self):
        """compares given matrix to the null matrix"""
        return self!=Matrix(self.m,self.n)
    
    def __eq__(self,other):
        """compares rows of given matrix to determine equality"""
        return self.rows==other.rows
    
    def __ne__(self,other):
        """compares rows of given matrix to determine inequality"""
        return self.rows!=other.rows
    
    #%% Basic Operations            
    def __add__(self,other):
        """adds entries of each row of given matrices"""
        m,n=self.m,self.n
        if m==other.m and n==other.n: 
            matrix=Matrix(m,n)
            matrix.rows=[list(map(operator.add,self.rows[i],other.rows[i])) for i in range(self.m)]
            return matrix
        else:
            raise Exception("Matrix addition of unidentical dimensions")
    
    def __iadd__(self,other):
        return self+other
    
    def __sub__(self, other):
        """subtracts entries of each row of given matrices"""
        m,n=self.m,self.n
        if m==other.m and n==other.n:
            matrix=Matrix(m,n)
            matrix.rows=[list(map(operator.sub,self.rows[i],other.rows[i])) for i in range(self.m)]
            return matrix
        else:
            raise Exception("Matrix subtraction of unidentical dimensions")
    
    def __isub(self,other):
        return self-other
    
    def __mul__(self,other):
        """performs non-associative multiplication of given matrix with another matrix or scalar"""
        M,N=self.m,other.n
        matrix=Matrix(M,N)
        if type(self)!=type(other):
            matrix.rows=list([[other*entry for entry in row] for row in self.rows])
            return matrix
        elif self.n==other.m:
            matrix.rows=[[functools.reduce(lambda x,y:x+y,list(map(operator.mul,self_row,other_row))) for other_row in other.Transpose().rows] for self_row in self.rows]
            return matrix
        else:
            raise Exception("Matrix multiplication of invalid dimensions")
    
    def __imul__(self,other):
        return self*other
    
    def __pow__(self,exp):
        m,n=self.m,self.n
        if self.m==self.n:
            if n==0:
                return Matrix.Identity(m,n)
            matrix=copy.deepcopy(self)
            for n in range(1,exp):
                matrix*=matrix
            return matrix
        else:
            raise Exception("Matrix powers of non-square matrix")
    
    def __reversed__(self):
        """reverses all rows and columns of given matrix"""
        m,n=self.m,self.n
        matrix=Matrix(m,n)
        matrix.rows=[row[::-1] for row in self.rows[::-1]]
        return matrix
    
    #%% Matrix Specific Operations
    def Dimensions(self):
        """returns # of rows and # of columns of given matrix"""
        return self.m,self.n
       
    def Transpose(self):
        """transposes (exchanges rows and columns) of given matrix"""
        m,n=self.n,self.m
        matrix=Matrix(m,n)
        matrix.rows=list(map(list,zip(*self.rows)))
        return matrix
    
    def Conjugate(self):
        """returns the conjugate matrix of a given matrix"""
        m,n=self.n,self.m
        matrix=Matrix(m,n)
        matrix.rows=[[item.real-item.imag*1j for item in row] for row in self.rows]
        return matrix
    
    def Hermitian(self):
        """returns the Hermitian of a given matrix"""
        return self.Conjugat().Transpose()
    
    def Gram_Schmidt(self):
        """returns an orthogonormal matrix for the column space"""
        m,n=self.m,self.n
        basis=Matrix(n,m)
        MT=self.Transpose()
        first=Vector(MT.rows[0])
        basis.rows[0]=list(first*(1/first.norm()))
        for i in range(1,n):
            v=Vector(MT.rows[i])
            v_orthogonal=copy.deepcopy(v)
            for j in range(i):
                u=Vector(basis.rows[j])
                v_orthogonal-=u*(u.dot(v)/u.norm())
            v_normalized=v_orthogonal/v_orthogonal.norm()
            basis.rows[i]=list(v_normalized)
        return basis.Transpose()

    def Index(self,item):
        """finds index of item, if the item is not found returns (-1,-1)"""
        for i in range(self.m):
            try:
                idx=self.rows[i].index(item)
                return i,idx
            except:
                pass
        return -1,-1
    
    def Count(self,item):
        """finds the count of a particular item; this is especially useful for determing sparse/dense matrices"""
        count=0
        for row in self.rows:
            for entry in row:
                if entry==item:
                    count+=1
        return count
    
    def Entry(self,idx1,idx2,item):
        """modifies entry at the given column and row indices"""
        self.rows[idx1][idx2]=item

    def ExchangeRows(self,idx1,idx2):
        """exchanges rows at the given indices"""
        self.rows[idx1],self.rows[idx2]=self.rows[idx2],self.rows[idx1]
    
    def ExchangeCols(self,idx1,idx2):
        """exchanges columns at the given indices"""
        for row in self.rows:
            row[idx1],row[idx2]=row[idx2],row[idx1]
    
    def Identity(M,N):
        """produces identity matrix of dimensions M by N"""
        I=Matrix(M,N)
        for idx in range(M):
            I.Entry(idx,idx,1)
        return I

    def Random(M,N,lower=0,upper=10):
        """produces a random matrix of dimensions M by N within specified bounds"""
        R=Matrix(M,N)
        for idx in range(M):
            R[idx]=[random.random()*(upper-lower)+(lower) for n in range(N)]
        return R
    
    def Permute(self):
        """ """
    #%% Matrix Factorizations and Solvers
    
    def LU_Decomposition(self):
        """decomposes matrix into the lower and upper triangle forms"""
        U=copy.deepcopy(self)
        L=Matrix.Identity(U.m,U.n)
        
        for i in range(U.m):
            for j in range(i):
                c=-U.rows[i][j]/U.rows[j][j]
                R=[c*entry for entry in U.rows[j]]
                U[i]=list(map(operator.add,U[i],R))
                L.Entry(i,j,-c)
        return L,U
    
#Update QR using householder reflections  

    def QR_Decomposition(self):
        """decomposes matrix into the orthonormal basis and upper triangular form"""
        Q=self.Gram_Schmidt()
        R=Q.Transpose()*self
        return Q,R
    
    def LDLT_Decomposition(self):
        """decomposes symmetric matrix by utilizing the property that such matrices have equivalent transposes and inverses"""
        LT=copy.deepcopy(self)
        D=Matrix.Identity(LT.m,LT.n)
        for i in range(LT.m):
            for j in range(i):
                c=-LT.rows[i][j]/LT.rows[j][j]
                R=[c*entry for entry in LT.rows[j]]
                LT[i]=list(map(operator.add,LT[i],R))
            d=LT[i][i]
            LT[i]=[entry/d for entry in LT[i]]
            D[i]=[entry*d for entry in D[i]]
        return LT.Transpose(),D,LT
    
    def Cholesky_Decomposition(self):
        """"""
    def Eigenvalue_Decomposition(self):
        """Use Householder reflections"""
    def QS_Decomposition(self):
        """"""
    def SVD_Decomposition(self):
        """"""
    
    def Determinant(L,U):
        """finds the determinant of a matrix, by utilizing LU Decomposition"""
        det=1
        for idx in range(L.m):
            det*=L[idx][idx]*U[idx][idx]
        return det
    
    def Solve(L,U,b):
        """"solves the equation Ax=b by utilizing LU matrices"""
        
        def ForwardSubstitution(L,b):
            """solves by examining the lower triangular matrix"""
            y=[]
            for i in range(L.m):
                row=L.rows[i]
                current=b[i][0]
                for j in range(i):
                    current+=-row[j]*y[j]
                current/=row[i]
                y.append(current)
            return y
        
        def BackSubstitution(U,y):
            """solves by examining the upper triangular matrix"""
            x=[]
            for i in range(U.m):
                row=U.rows[U.m-i-1]
                current=y[-1-i]
                for j in range(len(x)):
                    current+=-row[U.n-j-1]*x[j]
                current/=row[-1-i]
                x.append(current)
            return x[::-1]
        
        return BackSubstitution(U,ForwardSubstitution(L,b))
    
    def Inverse(self):
        """finds the inverse of a matrix through Gauss-Jordan Method """
        
        def ForwardStep():
            """performs Gaussian Elimination to augment the Identity Matrix through forward substitution"""
            A=copy.deepcopy(self)
            B=Matrix.Identity(A.m,A.n)
            for i in range(A.m):
                for j in range(i):
                    c=-A.rows[i][j]/A.rows[j][j]
                    R=[c*entry for entry in A.rows[j]]
                    S=[c*entry for entry in B.rows[j]]
                    A[i]=list(map(operator.add,A[i],R))
                    B[i]=list(map(operator.add,B[i],S))
            return A,B
        
        def BackwardStep(A,B):
            """performs portion of Gauss-Jordan Elimination to augment the Identity Matrix through back substitution"""
            for i in range(A.m):
                idx=-i-1
                A_pivot=A.rows[idx][idx]
                if A_pivot!=1:
                    B.rows[idx]=[entry/A_pivot for entry in B.rows[idx]]
                    A.rows[idx]=[entry/A_pivot for entry in A.rows[idx]]
                
                for j in range(i):
                    c=-1*A.rows[idx][-j-1]/A.rows[-j-1][-j-1]
                    A.rows[idx][-j-1]=0
                    B.rows[idx]=list(map(operator.add,B.rows[idx],map(lambda x: c*x,B.rows[-j-1])))
            return B
        A,B=ForwardStep()
        return BackwardStep(A,B)
    
    #%% Read and Write Matrices
    
    def Write(self,filename):
        """writes the given matrix as a text file with the given name"""
        file=open(filename+".txt","w")
        file.write(str(self.rows))
        file.close()
        print ("Done.")
    
    def Read(filename):
        """reads a text file as a matrix with the given name"""
        file=open(filename+".txt","r")
        contents=eval(file.read())
        matrix=Matrix(len(contents),len(contents[0]))
        matrix.rows=[row for row in contents]
        print("Done.")
        return matrix

class Vector:
    
    #%% Initialization and Representation
    def __init__(self,row=None,length=None):
        """initializes vector from a given list; if no list is specified, length is a required argument to produce the null vector"""
        if row!=None:
            self.entries=row
            self.length=len(row)
        else:
            self.entries=[0 for i in range(length)]
            self.length=length
            
    def __repr__(self):
        """list representation of a given vector"""
        return str(self.entries)
    
    def __getitem__(self,idx):
        """returns specific element of vector at the given index"""
        return self.entries[idx]
    
    def __setitem__(self,idx,item):
        """sets vector element to item at the specificied index"""
        self.entries[idx]=item

    def __add__(self,other):
        """adds each element of vectors, producing a new vector"""
        return Vector([self.entries[i]+other.entries[i] for i in range(self.length)])
    
    def __sub__(self,other):
        """subtracts each element of vectors, producing a new vector"""
        return Vector([self.entries[i]-other.entries[i] for i in range(self.length)])
    
    def __isub__(self,other):
        return self-other
    
    def __mul__(self,other):
        """multiplies vector by scalar; does not obey the principle of associativity"""
        if type(other)==type(self):
            raise Exception("cannot multiply two vectors together; can only multiply by scalar")
        else:
            return Vector([other*self.entries[i] for i in range(self.length)])
        
    def __truediv__(self,other):
        """divides vector by scalar; does not obey the principle of associativity"""
        if type(other)==type(self):
            raise Exception("cannot divide two vectors together; can only divide by scalar")
        else:
            return Vector([self.entries[i]/other for i in range(self.length)])
        
    def dot(self,other):
        """returns the standard inner product between two vectors"""
        return sum([self.entries[i]*other.entries[i] for i in range(self.length)])
    
    def norm(self):
        """returns the standard Euclidean norm of a vector"""
        return math.sqrt(self.dot(self))
    

    
def Polynomial_Least_Squares(x,y,order=1):
    """performs least square fit in the form 'a0*x**0+a1*x**1+a2*x**2...' for a given order (degree)"""
    A=Matrix(len(x),order+1)
    A.rows=[[entry**n for n in range(order+1)] for entry in x]
    b=Matrix(len(y),1)
    b.rows=[[entry] for entry in y]
    AT=A.Transpose()
    c=(AT*A).Inverse()*AT*b
    y_predict=lambda x:sum([c.rows[n][0]*x**n for n in range(order+1)])
    y_average=sum(y)/len(y)
    RSS=0
    Syy=0
    for i in range(len(x)):
        RSS+=(y[i]-y_predict(x[i]))**2
        Syy+=(y_average-y[i])**2
    r=(1-RSS/Syy)
    return y_predict,r,c

def Exponential1_Least_Squares(x,y):
    """performs least square fit in the form 'y=a*e**x'"""
    Y=[math.log(entry) for entry in y]
    y_predict,r,c=Polynomial_Least_Squares(x,Y)
    c.rows[0][0]=math.exp(c.rows[0][0])
    y=lambda x: c.rows[0][0]*math.exp(c.rows[1][0]*x)
    return y,r,c

def Exponential2_Least_Squares(x,y):
    """performs least square fit in the form 'y=a*b**x'"""
    Y=[math.log(entry) for entry in y]
    y_predict,r,c=Polynomial_Least_Squares(x,Y)
    c.rows[0][0]=math.exp(c.rows[0][0])
    c.rows[1][0]=math.exp(c.rows[1][0])
    y=lambda x: c.rows[0][0]*math.pow(c.rows[1][0],x)
    return y,r,c

def Logarithmic_Least_Squares(x,y):
    """performs least square fit in the form 'y=a+b*ln(x)'"""
    X=[math.log(entry) for entry in x]
    y_predict,r,c=Polynomial_Least_Squares(X,y)
    y=lambda x: c.rows[0][0]+c.rows[1][0]*math.log(x)
    return y,r,c

def Power_Least_Squares(x,y):
    """performs least square fit in the form 'y=a*x**b'"""
    X=[math.log(entry) for entry in x]
    Y=[math.log(entry) for entry in y]
    y_predict,r,c=Polynomial_Least_Squares(X,Y)
    y=lambda x: math.exp(c.rows[0][0])*math.exp(x)*math.exp(c.rows[0][0])
    return y,r,c
    
def Trignometric_Least_Squares(x,y):
    """performs least square fit in the form 'y=a+b*cos(x)'"""
    X=[math.cos(entry) for entry in x]
    y_predict,r,c=Polynomial_Least_Squares(X,y)
    y=lambda x: c.rows[0][0]+c.rows[1][0]*math.cos(x)
    return y,r,c

def Saturation_Least_Squares(x,y):
    """performs least square fit in the form '1/y=b/a*1/x+1/a'"""
    Y=[1/entry for entry in y]
    X=[1/entry for entry in x]
    y_predict,r,c=Polynomial_Least_Squares(X,Y)
    A=1/c.rows[0][0]
    B=c.rows[1][0]/c.rows[0][0]
    c.rows[0][0],c.rows[1][0]=A,B
    y=lambda x: A+B*x
    return y,r,c