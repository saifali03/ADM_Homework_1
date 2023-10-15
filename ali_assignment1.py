import numpy as np
import textwrap
from datetime import datetime
import calendar
from collections import deque
from collections import OrderedDict
from collections import namedtuple
from collections import defaultdict
from collections import Counter
import math
import os
import random
import re
import sys

# Python If-Else

if __name__ == '__main__':
    n = int(input().strip())
if n % 2 == 0 and (n >= 2 and n <= 5):
    print("Not Weird")
elif n % 2 == 0 and (n >= 20 and n <= 20):
    print("Weird")
elif n % 2 == 0 and (n >= 20):
    print("Not Weird")
else:
    print("Weird")

# Arithmetic Operators

if __name__ == '__main__':
    a = int(input())
    b = int(input())

    print(a+b)
    print(a-b)
    print(a*b)

# Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)

# Loops
if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i**2)

# Write a function

def is_leap(year):
    leap = False
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                leap = True
        else:
            leap = True
    return leap

# Print Function

if __name__ == '__main__':
    n = int(input())
    for i in range(1, n+1):
        print(i, end='')

# Say "Hello, World!" With Python
print("Hello, World!")

# List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    print([[i, j, k] for i in range(x + 1) for j in range(y + 1)
          for k in range(z + 1) if i + j + k != n])

# Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    theset = set(map(int, input().split()))
    sortedarr = sorted(list(theset), reverse=True)
    print(sortedarr[1])

# Nested Lists

if __name__ == '__main__':
    records = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        records.append([name, score])
    second_highest = sorted(set([score for name, score in records]))[1]
    names = (
        sorted([name for name, score in records if score == second_highest]))
    [print(name) for name in names]

# Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()

# Lists

if __name__ == '__main__':
    N = int(input())
    thelist = []
    for _ in range(N):
        command = list(input().strip().split())
        if command[0] == "insert":
            thelist.insert(int(command[1]), int(command[2]))
        elif command[0] == "print":
            print(thelist)
        elif command[0] == "remove":
            thelist.remove(int(command[1]))
        elif command[0] == "append":
            thelist.append(int(command[1]))
        elif command[0] == "sort":
            thelist.sort()
        elif command[0] == "pop":
            thelist.pop()
        elif command[0] == "reverse":
            thelist.reverse()
# Tuples

N = int(input())
t = tuple(map(int, input().split()))
print(hash(t))

# sWAP cASE


def swap_case(s):
    newstr = ''.join([char.lower() if char.isupper()
                     else char.upper() for char in s])
    return newstr

# What's Your Name?


def print_full_name(first, last):
    print(f'Hello {first} {last}! You just delved into python.')

# Mutations


def mutate_string(string, position, character):
    l = list(string)
    l[int(position)] = character
    s = ''.join(l)
    return s

# String Split and Join


def split_and_join(line):
    words = list(line.split())
    newstr = "-".join(words)
    return newstr

# Find a string


def count_substring(main_string, substring):
    total = 0
    for i in range(len(main_string)):
        if main_string[i:len(main_string)].startswith(substring):
            total += 1
    return total

# String Validators


if __name__ == '__main__':
    s = input()

    if any(char.isalnum() for char in s):
        print("True")
    else:
        print("False")

    if any(char.isalpha() for char in s):
        print("True")
    else:
        print("False")

    if any(char.isdigit() for char in s):
        print("True")
    else:
        print("False")

    if any(char.islower() for char in s):
        print("True")
    else:
        print("False")

    if any(char.isupper() for char in s):
        print("True")
    else:
        print("False")

# Text Wrap


def wrap(string, max_width):
    newstr = textwrap.fill(string, max_width)
    return newstr

# String Formatting


def print_formatted(number):
    width = len(bin(number)[2:])
    for i in range(1, number+1):
        deci = str(i)
        octa = oct(i)[2:]
        hexa = hex(i)[2:].upper()
        bina = bin(i)[2:]
        print(deci.rjust(width), octa.rjust(width),
              hexa.rjust(width), bina.rjust(width))

# Capitalize!


def solve(s):
    words = s.split(sep=" ")
    for i in range(len(words)):
        words[i] = words[i].capitalize()
    newstr = " ".join(words)
    return newstr

# Merge the Tools!


def merge_the_tools(string, k):
    for i in range(0, len(string), k):
        substring = string[i:i + k]
        unique_chars = []
        for char in substring:
            if char not in unique_chars:
                unique_chars.append(char)
        print(''.join(unique_chars))

# The Minion Game


def minion_game(string):
    vowels = "AEIOU"
    length = len(string)
    kevin_score = 0
    stuart_score = 0

    for i in range(length):
        if string[i] in vowels:
            kevin_score += length - i
        else:
            stuart_score += length - i
 # ---
    if kevin_score > stuart_score:
        print("Kevin", kevin_score)
    elif kevin_score < stuart_score:
        print("Stuart", stuart_score)
    else:
        print("Draw")

# Introduction to Sets


def average(array):
    theset = set(array)
    meann = round(sum(theset)/len(theset), 3)
    return meann

# No Idea!


happiness = 0
firstline = list(map(int, input().split()))
n = firstline[0]
m = firstline[1]
Arr = list(map(int, input().split()))
setA = set(map(int, input().split()))
setB = set(map(int, input().split()))

for elem in Arr:
    if elem in setA:
        happiness += 1
    if elem in setB:
        happiness -= 1
print(happiness)

# Symmetric Difference

M = int(input())
setA = set(map(int, input().split()))
N = int(input())
setB = set(map(int, input().split()))
symdiff = setA.symmetric_difference(setB)
difflist = sorted(list(symdiff))
for elem in difflist:
    print(elem)

# Set .union() Operation

M = int(input())
setA = set(map(int, input().split()))
N = int(input())
setB = set(map(int, input().split()))
UNION = setA.union(setB)
print(len(UNION))

# Set .intersection() Operation

M = int(input())
setA = set(map(int, input().split()))
N = int(input())
setB = set(map(int, input().split()))
INTER = setA.intersection(setB)
print(len(INTER))

# Set .difference() Operation

M = int(input())
setA = set(map(int, input().split()))
N = int(input())
setB = set(map(int, input().split()))
DIFF = setA.difference(setB)
print(len(DIFF))

# Set .symmetric_difference() Operation

M = int(input())
setA = set(map(int, input().split()))
N = int(input())
setB = set(map(int, input().split()))
SYMDIFF = setA.symmetric_difference(setB)
print(len(SYMDIFF))

# Set .add()

countries = set()
for _ in range(int(input())):
    countries.add(str(input()))
print(len(countries))

# Set .discard(), .remove() & .pop()

n = int(input())
mylist = set(map(int, input().split()))

N = int(input())
for _ in range(N):
    command = list(input().split())
    if command[0] == 'pop':
        mylist.pop()
    elif command[0] == 'remove':
        mylist.remove(int(command[1]))
    elif command[0] == 'discard':
        mylist.discard(int(command[1]))

print(sum(mylist))

# Set Mutations

A = int(input())
set_A = set(map(int, input().split()))
N = int(input())
for _ in range(N):
    cmd, *args = input().split()
    setB = set(map(int, input().split()))
    if cmd == 'intersection_update':
        set_A.intersection_update(setB)
    elif cmd == 'update':
        set_A.update(setB)
    elif cmd == 'symmetric_difference_update':
        set_A.symmetric_difference_update(setB)
    elif cmd == 'difference_update':
        set_A.difference_update(setB)

print(sum(set_A))

# Check Subset

T = int(input())
for i in range(T):
    LenA = int(input())
    setA = set(map(int, input().split()))
    LenB = int(input())
    setB = set(map(int, input().split()))

    if len(setA - setB) == 0:
        print("True")
    else:
        print("False")

# Check Strict Superset

setA = set(map(int, input().split()))
N = int(input())
YES = 0
NO = 0
for i in range(N):
    if not setA.issuperset(set(map(int, input().split()))):
        NO += 1
if NO != 0:
    print('False')
else:
    print('True')

# The Captain's Room

K = input()
mylist = input().split()
countdict = Counter(mylist)
for k, v in countdict.items():
    if v == 1:
        print(k)

# collections.Counter()

X = int(input())
sizes = Counter(map(int, input().split()))
nofcustomers = int(input())
amount = 0
for _ in range(nofcustomers):
    size, price = map(int, input().split())
    if sizes[size]:
        amount += price
        sizes[size] -= 1
print(amount)

# DefaultDict Tutorial

A = defaultdict(list)
n, m = map(int, input().split())
for i in range(n):
    key = input()
    A[key].append(str(i + 1))

for j in range(m):
    query = input()
    result = ' '.join(A[query])
    if result:
        print(result)
    else:
        print(-1)

# Collections.namedtuple()

N = int(input())
columns = input().split()
students = namedtuple('students', columns)
total_marks = 0
for _ in range(N):
    MARKS, CLASS, NAME, ID = input().split()
    student = students(MARKS, CLASS, NAME, ID)
    total_marks += int(student.MARKS)
print(round(total_marks/N, 2))

# Collections.OrderedDict()

order = OrderedDict()
listofitems = int(input())
for _ in range(listofitems):
    input_line = input()
    item, price = input_line.rsplit(' ', 1)
    order[item] = order.get(item, 0) + int(price)

for item, price in order.items():
    print(item, price)

# Word Order
n = int(input())
charinputs = []

for i in range(n):
    charinputs.append(input().strip())

counts = Counter(charinputs)
print(len(counts))
print(*counts.values())

# Collections.deque()

N = int(input())
d = deque()
for _ in range(N):
    commands = input().strip().split()
    if (commands[0] == 'append'):
        d.append(commands[1])
    elif (commands[0] == 'pop'):
        d.pop()
    elif (commands[0] == 'popleft'):
        d.popleft()
    elif (commands[0] == 'appendleft'):
        d.appendleft(commands[1])
result = ' '.join(d)
print(result)

# Company Logo

if __name__ == '__main__':
    S = input()
    S = sorted(S)
    freq = Counter(list(S))
    for x, y in freq.most_common(3):
        print(x, y)

# Calendar Module

month, day, year = map(int, input().split())
day_of_week_int = calendar.weekday(year, month, day)
day_of_week_str = calendar.day_name[day_of_week_int].upper()
print(day_of_week_str)

# Time Delta

#!/bin/python3


# Complete the time_delta function below.

def time_delta(t1, t2):
    time_format = '%a %d %b %Y %H:%M:%S %z'
    t1 = datetime.strptime(t1, time_format)
    t2 = datetime.strptime(t2, time_format)
    return str(int(abs((t1-t2).total_seconds())))


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()

# Exceptions

T = int(input())
for _ in range(T):
    try:
        a, b = map(int, input().split())
        print(int(a/b))
    except ZeroDivisionError:
        print("Error Code:"+" integer division or modulo by zero")
    except ValueError as ve:
        print("Error Code:", ve)

# Zipped!

N, X = map(int, input().split())
score = []

for _ in range(X):
    score.append(list(map(float, input().split())))

for stud in zip(*score):
    average_score = round(sum(stud) / len(stud), 1)
    print(average_score)

# Athlete Sort

N_M = input().split(' ')
N = int(N_M[0])
M = int(N_M[1])
table = []
for i in range(N):
    line = input().split()
    table.append([int(x) for x in line])
sort_key = int(input())
table.sort(key=lambda x: x[sort_key])
for row in table:
    print(' '.join(str(x) for x in row))

# ginortS

aninput = input().strip()
lowercase_letters = []
uppercase_letters = []
odd_digits = []
even_digits = []

for x in aninput:
    if x.islower():
        lowercase_letters.append(x)
    elif x.isupper():
        uppercase_letters.append(x)
    elif x.isnumeric():
        if int(x) % 2 == 0:
            even_digits.append(x)
        else:
            odd_digits.append(x)

lowercase_letters.sort()
uppercase_letters.sort()
odd_digits.sort()
even_digits.sort()

sorted_str = ''.join(lowercase_letters +
                     uppercase_letters + odd_digits + even_digits)
print(sorted_str)

# Map and Lambda Function


def cube(x): return x ** 3


def fibonacci(n):
    x, y = 0, 1
    for _ in range(n):
        yield x
        x, y = y, x + y

# Standardize Mobile Number Using Decorators


def wrapper(f):
    def fun(l):
        f(['+91 ' + i[-10:-5] + ' ' + i[-5:] for i in l])
    return fun

# XML 1 - Find the Score. Peeked at somebody's solution to understand
# Not truly my code


def get_attr_number(node):
    a = 0
    for child in node:
        a += (get_attr_number(child))
    return (len(node.attrib) + a)

# Detect Floating Point Number


t = int(input())
for _ in range(t):
    num = input()
    regex = r"^[\.\+\-\d]\d*\.\d+$"
    print(bool(re.match(regex, num)))

# Re.split()
regex_pattern = r"[.,]"

# Group(), Groups() & Groupdict()

regex_repeat = r'([a-zA-Z0-9])((.*\1\1)|(\1))'
search = re.search(regex_repeat, input().strip())
if search != None:
    print(search.group(1))
else:
    print('-1')

# Re.findall() & Re.finditer()

s = input()
vowels = "aeiou"
consonant = "mnbvcxzsdfghjklpuytrwq"
pattern = rf'(?<=[{consonant}])([{vowels}]{{2,}})(?=[{consonant}])'
match = re.findall(pattern, s, flags=re.I)
if len(match) != 0:
    print('\n'.join(match))
else:
    print("-1")

# Re.start() & Re.end()

string = input()
substring = input()

pattern = re.compile(substring)
match = pattern.search(string)

if not match:
    print('(-1, -1)')

while match:
    print('({0}, {1})'.format(match.start(), match.end()-1))
    match = pattern.search(string, match.start() + 1)

# Regex Substitution

n = int(input())
regex = re.compile(r'(?<= )(&&|\|\|)(?= )')


def replace_operator(match):
    if match.group() == '&&':
        return 'and'
    elif match.group() == '||':
        return 'or'


for i in range(n):
    s = input()
    newstr = regex.sub(replace_operator, s)
    print(newstr)

# Validating phone numbers

N = int(input())
for _ in range(N):
    number = input()
    if re.match(r'[789]\d{9}$', number):
        print('YES')
    else:
        print('NO')

# Validating and Parsing Email Addresses

pattern = r'^<[A-Za-z](\w|-|\.)+@[A-Za-z]+\.[A-Za-z]{1,3}>$'
for _ in range(int(input())):
    name, email = input().split(' ')
    if re.match(pattern, email):
        print(name, email)

# Validating UID


def is_valid_uid(uid):
    if len(set(uid)) != 10:
        return False
    if len(re.findall(r'[A-Z]', uid)) < 2:
        return False
    if len(re.findall(r'[0-9]', uid)) < 3:
        return False
    if not re.match(r'^[a-zA-Z0-9]*$', uid):
        return False
    return True


t = int(input())
for _ in range(t):
    uid = input().strip()
    if is_valid_uid(uid):
        print('Valid')
    else:
        print('Invalid')

# Validating Credit Card Numbers


def is_valid_credit_card(credit):
    credit_removed_hyphen = credit.replace('-', '')
    if re.match(r'^(4|5|6)\d{15}$', credit) or re.match(r'^(4|5|6)\d{3}-\d{4}-\d{4}-\d{4}$', credit):
        if not re.search(r'(\d)(-?\1){3}', credit_removed_hyphen):
            return 'Valid'
    return 'Invalid'


n = int(input())
for _ in range(n):
    credit = input().strip()
    result = is_valid_credit_card(credit)
    print(result)

# Decorators 2 - Name Directory


def person_lister(f):
    def inner(people):
        thelist = map(f, sorted(people, key=lambda x: int(x[2])))
        return thelist
    return inner


# Arrays


def arrays(arr):
    newarr = numpy.array(arr[::-1], float)
    return newarr


# Shape and Reshape

theinput = list(map(int, input().strip().split()))
nparr = np.array(theinput)
print(np.reshape(nparr, (3, 3)))

# Transpose and Flatten

n, m = map(int, input().split())
arr = []
for i in range(n):
    row = list(map(int, input().split()))
    arr.append(row)
nparr = numpy.array(arr)
print(numpy.transpose(nparr))
print(nparr.flatten())

# Concatenate

n, m, p = map(int, input().split())
lista1 = [list(map(int, input().split())) for i in range(n)]
lista2 = [list(map(int, input().split())) for i in range(m)]
a1 = numpy.array(lista1)
a2 = numpy.array(lista2)

print(numpy.concatenate((a1, a2)))

# Zeros and Ones

x = tuple(map(int, input().split()))
zeros_array = np.zeros(x, dtype=int)
ones_array = np.ones(x, dtype=int)
print(zeros_array)
print(ones_array)

# Eye and Identity

np.set_printoptions(legacy="1.13")
N, M = map(int, input().split())
matrix = np.eye(N, M)
print(matrix)

# Array Mathematics

N = list(map(int, input().split()))
A, B = [], []
for _ in range(N[0]):
    A.append([int(i) for i in input().strip().split()])
for _ in range(N[0]):
    B.append([int(i) for i in input().strip().split()])
A = np.array(A)
B = np.array(B)
print(A + B)
print(A - B)
print(A * B)
print(A // B)
print(A % B)
print(A ** B)

# Floor, Ceil and Rint

numpy.set_printoptions(legacy='1.13')
theinput = list(map(float, input().strip().split()))
nparr = numpy.array(theinput)
print(numpy.floor(nparr))
print(numpy.ceil(nparr))
print(numpy.rint(nparr))

# Sum and Prod

n = list(map(int, input().split()))
x = []
for i in range(n[0]):
    x.append(list(map(int, input().split())))
nparr = np.array(x, int)
thesum = np.sum(nparr, axis=0)
print(np.prod(thesum))

# Min and Max

N = list(map(int, input().split()))
x = []
for i in range(N[0]):
    x.append(list(map(int, input().split())))
nparr = numpy.array(x, int)
min1 = numpy.min(nparr, axis=1)
print(numpy.max(min1))

# Mean, Var, and Std

n, m = tuple(map(int, input().split()))
A = numpy.array([list(map(int, input().split())) for j in range(n)])
print(numpy.mean(A, axis=1))
print(numpy.var(A, axis=0))
stdnp = numpy.std(A)
print(round(stdnp, 11))

# Dot and Cross

N = int(input())
a = []
b = []
for _ in range(N):
    a.append(list(map(int, input().split())))
for _ in range(N):
    b.append(list(map(int, input().split())))
nparr = numpy.array(a, int)
nparr2 = numpy.array(b, int)
res = numpy.dot(nparr, nparr2)
print(res)

# Inner and Outer

A = list(map(int, input().split()))
B = list(map(int, input().split()))
nparr = numpy.array(A, int)
nparr2 = numpy.array(B, int)
print(numpy.inner(nparr, nparr2))
print(numpy.outer(nparr, nparr2))

# Polynomials

N = list(map(float, input().split()))
x = float(input())

result = numpy.polyval(N, x)
print(result)

# Linear Algebra

N = int(input())
l = []
for _ in range(N):
    l.append(list(map(float, input().strip().split())))
nparr = np.array(l)
nparr = nparr.reshape((N, N))
result = np.linalg.det(nparr)
print(round(result, 2))

# Birthday Cake Candles

def birthdayCakeCandles(candles):
    maxnum = max(candles)
    count = 0
    for i in candles:
        if i == maxnum:
            count = count + 1
    return count


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

# Number Line Jumps

def kangaroo(x1, v1, x2, v2):
    if x2 > x1 and v2 > v1:
        return "NO"
    else:
        if v2-v1 == 0:
            return 'NO'
        else:
            result = (x1-x2) % (v2-v1)
            if result == 0:
                return 'YES'
            else:
                return 'NO'


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

# Viral Advertising

def viralAdvertising(n):
    m = 0
    next = 5
    count = 0
    for i in range(n):
        m = next//2
        count += m
        next = m*3
    return count


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

# Recursive Digit Sum

def superDigit(n, k):
    d = map(int, list(n))
    short = (sum(d))

    def func(c):
        new_s = str(c)
        if len(new_s) == 1:
            return c
        else:
            d = map(int, list(new_s))
            return func(sum(d))
    return func(short * k)


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

# Insertion Sort - Part 1

def insertionSort1(n, arr):
    lastelem = n-1
    a = arr[lastelem]
    arr[lastelem] = arr[lastelem-1]

    for i in range(n):
        lastelem = lastelem - 1
        b = arr[lastelem]
        if lastelem < 0:
            break
        if b < a:
            arr[lastelem+1] = a
            print(*arr, sep=" ")
            break
        else:
            arr[lastelem+1] = arr[lastelem]
            print(*arr, sep=" ")
    if b >= a:
        arr[0] = a
        print(*arr, sep=" ")


if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)
