import numpy as np
import sys

file_name = 'emb_deepwalk/email/week/output_week_' + sys.argv[1] + '.emb'
file = open(file_name,'r')
result = []
first_line = file.readline()
tmp = first_line.split(' ')
print(tmp[0])
for i in range(int(tmp[0])):
	result.append([])
for line in file:
	line = line.strip('\n')
	item = line.split(' ')
	if (len(item) >3):
		result[int(item[0])-1] = item [1:]

result = np.array(result)
print(len(result[0]))
print(len(result))
name = 'npy_deepwalk/email_week/email_week_' + sys.argv[1]

np.save(name, result)
