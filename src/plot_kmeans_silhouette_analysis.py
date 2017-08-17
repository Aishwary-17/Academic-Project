for i in range(1000):
    ctr=0
    n=i
    while n!=1:
        if n%2==0:
            n=n/2
        else:
            n=3*n+1
        ctr=ctr+1
    print ctr