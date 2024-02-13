import numpy as np

fname = 'C:/Users/danko/Desktop/L2C_python/Server/DATA_LOG.txt'
f1 = open(fname, 'r')

data = f1.readlines()

f1.close()

formated_data = list()

for a in data:
    tmp = a.replace('[', '', -1)
    tmp = tmp.replace(']', '', -1)
    tmp = tmp.replace(' ', '', -1)
    formated_data.append(tmp.split(','))

x = list()
for a in formated_data:
    x.append(np.array(a).astype(float))

y = list()
for a in x:
    # y.append([int(a[0]), int(a[1]), a[2], int(a[3]), a[4:].astype(int).tolist()])
    y.append([int(a[0]), int(a[1]), a[2], int(a[3]), int(a[11]), int(a[13]), int(a[14])])


f2 = open('C:/Users/danko/Desktop/L2C_python/Server/log1.csv', 'a')
for a in y:
    f2.write(f"{a}\n")
f2.close()
