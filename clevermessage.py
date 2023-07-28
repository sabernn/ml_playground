
# def GuessWho(list):
#     for i in list:
#         print(chr(i), end="")
# print("We are ",end="")
# GuessWho([84,67,78,65,33])


# print(", and we are looking for ", end="")
# GuessWho([89,79,85,33])


def Enigma(x):
    y = -26/3*x**3 +\
    40*x**2 -\
    145/3*x +\
    84
    return int(y)

def LostArk(x):
    y = -37/3*x**3 +\
    45*x**2 -\
    128/3*x +\
    89
    return int(y)

# -37/3*A9^3+45*A9^2-128/3*A9+89

def GuessWho(list):
    for i in list:
        print(chr(i), end="")
print("We are ",end="")
GuessWho([Enigma(x) for x \
in range(4)])
print(", and we are looking for \
    ", end="")
GuessWho([LostArk(x) for x in range(4)])
# Unearth Enigma and LostArk in our booth!