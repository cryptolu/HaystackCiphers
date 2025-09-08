####################################Defining the code correcion scheme####################################

q=2  #Finite Field size
r=5  #Number of randomness
n=5  #Plaintext size
s=50 #s>=n+r, Encoded plaintext size, s-(n+r) is the redundancy
A=MatrixSpace(GF(q),n+r,s).random_element() #Matrix A of the Error Correction Code
PCM=A.right_kernel_matrix() #Parity Check Matrix of A: if c in Fq^s is an element of the code, then PCM*c=0

#Encode: Fq^n->Fq^s, the encoding function of the Error Correction Code
def Encode(p):
    Rnd=VectorSpace(GF(q),r).random_element()
    x=vector(GF(q),list(p)+list(Rnd))*A
    return(x)

#Decode: We did note define the real decode function that would consist of retrieving the corresponding plaintext
#        of a cipher text if it exists, or return false otherwise. Instead, we simply verify that a given ciphertext
#        is a element of the code or not.
#        If Decode(F)=True for F a forged ciphertext, then the forgery attack succeeded
def Decode(x):
    return(PCM*vector(GF(q),x)==0) #If True then x is a codeword, else not and we return bot


####################################Defining the Haystack-Code cipher:####################################

ShuffleElement=Permutations(range(s),s).random_element() #Lookup table shuffling function
#computing the reverse shuffle function:
UnshuffleElement=[]
for i in range(s):
    j=0
    while ShuffleElement[j]!=i:
        j+=1
        UnshuffleElement.append(j)
#We have for any input i UnshuffleElement[ShuffleElement[i]]=ShuffleElement[UnshuffleElement[i]]=i

#The Shuffle function of the Haystack cipher
def Shuffle(x):
    x=list(x)
    c=[0 for _ in range(len(x))]
    for i in range(len(x)):
        c[ShuffleElement[i]]=x[i]
    return(c)

#The Unshuffle function of the Haystack cipher, we have Shuffle(Unshuffle(a))=Unshuffle(Shuffle(a))=a
def Unshuffle(c):
    c=list(c)
    x=[0 for _ in range(len(c))]
    for i in range(len(c)):
        x[UnshuffleElement[i]]=c[i]
    return(c)

#The function Ek of the Haystack-Code cipher
def EncCode(p):
    x=Encode(p)
    c=Shuffle(x)
    return(x)

#The Decoding function that returns if a given ciphertext is correct or not.
def DecCode(c):
    x=Unshuffle(c)
    p=Decode(x)
    return(p)

#The CCA1 function that an attacker can call as much as he wants to receive the ciphertext of a random plaintext
def RandEncQuery():
    p=VectorSpace(GF(q),n).random_element()
    return(EncCode(p))

#Verification of the validity of the Haystack-Code cipher
for _ in range(100):
    assert(DecCode(RandEncQuery()))


####################################Linear Fault Attack forgery on the CCA1 game:####################################

#A function that returns the number of zeroes of a given list
def numOfNonZeroes(L):
    ctr=0
    for l in L:
        if l!=0:
            ctr+=1
    return ctr

#The core LFR function that perform an algebraic fault on a ciphertext
def OneLinearFault(ListIndexes, Forgery, X):

    #Choose a random index to perform the fault on, that hasn't been already faulted
    #Verify that the vector is not linearly independent from the other elements of the matrix
    rcol=randrange(len(ListIndexes))
    while numOfNonZeroes(X.solve_right(X[:,rcol]))==1:
        rcol=randrange(len(ListIndexes))
    ListIndexes.pop(rcol)

    #Compute the vectors that sums to the selected vector at index rcol
    LinEq=X.solve_right(X[:,rcol])

    #Compute the algebraic fault
    temp=GF(q)(0)
    for i in range(s):
        temp+=(LinEq[i]*Forgery[i])[0]

    #Fault the ciphertext
    ForgeryFinal=list(Forgery)
    ForgeryFinal[rcol]=temp

    return(ListIndexes, ForgeryFinal)

#The Linear Fault Recoding (LFR) attack presentedn in the paper.
#T: number of traces, the more the less chances we avoid false positives, T is in O(s) queries and O(s^omega) time complexity
#Notice that we do not use the information of the matrix A of the Error Correction Code to perform our attack.
def LinearFaultRecoding(T=s+20):

    #In the CCA1 model, we can ask as many random encryption queries as we want before asking for the decryption
    Queries=[]
    for _ in range(T):
        Queries.append(RandEncQuery())

    #We store all of these ciphertexts in a matrix X
    X=Matrix(GF(q),[list(Queries[i]) for i in range(T)])

    #We select one of the ciphertext as a basis to construct our forgery
    Forgery=list((X[0,:])[0])
    ListIndexes=list(range(s))

    #We perform algebraic fault until finding a valid ciphertext that was not previously queried
    #Usually, one linear fault is enough
    while len(ListIndexes)>0:
        (ListIndexes,Forgery)=OneLinearFault(ListIndexes, Forgery, X)
        # print("Fault %d: " % (s-len(ListIndexes)), end="")
        # for e in Forgery:
        #     print(e,end="")
        # print()

        if (Forgery not in Queries):
            if DecCode(vector(GF(q),Forgery)):
                return(Forgery)
    return(False)

#Verify that we can perform 100 successfull forgery attacks
Successful=True
for _ in range(100):
    Successful= Successful and LinearFaultRecoding()
assert Successful
print("LFR is successful.")
