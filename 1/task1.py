from random import randint

a = [randint(0, 100) for _ in range(25)]
b = list(filter(lambda x: x % 2 == 0, a))

print(a, b, sum(b), sep='\n')

