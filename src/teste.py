global x
x = 5


def vv():
    global x
    print(x)
    
    x = x +1


vv()
print(x)
