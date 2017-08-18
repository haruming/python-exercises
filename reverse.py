from functools import reduce

#list
def reverse1(str):
	a = list(str)
	a.reverse()
	return (''.join(a))

#slice, define the step to be -1
def reverse2(str):
	return (str[::-1])

#reduce and lambda
def reverse3(str):
	return reduce(lambda x, y : y + x, str)

#more complex(naive basic one)
def reverse4(str):
	strTemp = ''
	l = len(str)-1
	while l >= 0:
		strTemp += str[l]
		l -=1
	return strTemp

test = 'yangweiming'
print(reverse1(test),reverse2(test),reverse3(test),reverse4(test))