import numpy as np 
import matplotlib.pyplot as plt

a = np.array([3,4])
b = np.array([7,2])

print(a)
print(type(a))
print(len(a))

print(a[0])
print(a[1])

sum1 = a + b

sum2 = np.add(a,b)

minus = a -b

multiple = a * b

divide = a / b

scalarMultiple = np.dot(3, a)

vectorMultiple = np.dot(a, b) 

# size of a vector
vectorSize1 = (a**2 + b**2) ** (1/2)

vectorSize2 = np.linalg.norm(a)

# draw a vector
plt.quiver(3,4, angles='xy', scale_units='xy', scale=1)
plt.xlim(0,5)
plt.ylim(0,5)
plt.text(3,4,r'$\vec{u}$', size=25)


# draw a, b, c = a+b
c = np.add(a,b)
plt.quiver(a[0], a[1], angles='xy', scale_units='xy', scale=1)
plt.quiver(b[0], b[1], angles='xy', scale_units='xy', scale=1)
plt.quiver(c[0], c[1], angles='xy', scale_units='xy', scale=1, color='r')

plt.xlim(0,11)
plt.ylim(0,7)

plt.axvline(x=0)
plt.axhline(x=0)