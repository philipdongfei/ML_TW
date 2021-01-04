import numpy as np

a = np.random.randn(3,4)
np.savetxt('sample.txt', a)

b = np.loadtxt('sample.txt')

print(a)

print(b)

np.savetxt('sample1.csv', a)
c = np.loadtxt('sample1.csv')

print(c)

np.savetxt('sample2.txt', a, delimiter=',')

d = np.loadtxt('sample2.txt', delimiter=',')
print(d)

np.savetxt('sample3.txt', a, fmt='%.2e')

np.savetxt('sample4.txt', a, fmt='%.2f')
np.savetxt('sample5.txt', a, fmt='%.3f',
           header='this is a header', footer='this is a footer')
np.savetxt('sample6.txt', a, fmt='%.3f',
           header='this is a header', footer='this is a footer',
           comments='>>>')

e = np.loadtxt('sample6.txt', comments='>>>')
print(e)
print(np.loadtxt('sample4.txt', usecols=(0,2)))
print(np.loadtxt('sample4.txt',skiprows=1))

print(np.loadtxt('foo.csv', dtype=[('col1', 'i8'),
                                   ('col2', 'S10'),
                                   ('col3', 'f8'),
                                   ('col4', 'S10')]))

print(np.loadtxt('foo.csv', dtype=[('col1', 'i8'),
                                   ('col2', 'S10'),
                                   ('col3', 'f8'),
                                   ('col4', 'S10')],unpack=True))


age, gender, tall, driver_license = np.loadtxt('foo.csv', dtype=[('col1', 'i8'),
                                   ('col2', 'S10'),
                                   ('col3', 'f8'),
                                   ('col4', 'S10')],unpack=True)
print(age)
print(gender)
print(tall)
print(driver_license)

def driver_license(str):
    if str == b'Yes' : return 1
    else : return -1

def gender(str):
    if str == b'male' : return 1
    else : return -1

print(np.loadtxt('foo.csv', converters={1: gender,
                                        3: driver_license}))





