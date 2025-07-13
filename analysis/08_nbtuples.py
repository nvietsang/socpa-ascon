import pycryptosat as cs
import sys

def upperbound(sSat,vesId,size_word,hwmax,startextra):
    sSat.add_clause([-(vesId[0]), startextra])
    for j in range (2,hwmax+1):
        sSat.add_clause([-(startextra+j-1)])
    for i in range(2,size_word):
        sSat.add_clause([-(vesId[i-1]),
                         startextra+(hwmax)*(i-1)])
        sSat.add_clause([-(startextra+(hwmax)*(i-2)),
                         startextra+(hwmax)*(i-1)])
        for j in range(2,hwmax+1):
            sSat.add_clause([-(vesId[i-1]), 
                             -(startextra+hwmax*(i-2)+j-2), 
                             startextra+(i-1)*hwmax+j-1])
            sSat.add_clause([-(startextra+hwmax*(i-2)+j-1), 
                             startextra+(i-1)*hwmax+j-1])
        sSat.add_clause([-(vesId[i-1]), 
                         -(startextra+hwmax*(i-2)+hwmax-1)])
    sSat.add_clause([-(vesId[size_word-1]), 
                     -(startextra+hwmax*(size_word-2)+hwmax-1)])
    startextra+=(size_word-1)*hwmax

if __name__ == "__main__":
    # Arguments
    # Run `python3 64 36 45 23` for k0
    # Run `python3 64 3 25 24`  for k1
    n  = int(sys.argv[1]) # n = 64 indexes
    s1 = int(sys.argv[2]) # 1st shift
    s2 = int(sys.argv[3]) # 2nd shift
    ub = int(sys.argv[4]) # upper bound for number of tuples

    list_set=[]
    for i in range(n):
        list_set.append([i,(i+s1)%n,(i+s2)%n])
    list_set_to_have=list(range(n))

    sSat=cs.Solver()

    #Each t subset of variable should appear 
    for i in list_set_to_have: 
        clause_presence=[]
        for j in list_set:
            if i in j:
                clause_presence += [list_set.index(j)+1]
        sSat.add_clause(clause_presence)        
    upperbound(sSat,
               list(range(1,len(list_set)+1)),
               len(list_set),
               ub,
               len(list_set)+1)

    satq,solution=sSat.solve()
    print(satq)
    if(satq):
        veritab = [False] * 64
        for i in range(1,len(list_set)+1):
            if solution[i]:
                for idx in list_set[i-1]:
                    veritab[idx] = True
        assert sum(veritab) == 64
        for i in range(1,len(list_set)+1):
            if solution[i]:
                print(list_set[i-1])
        

