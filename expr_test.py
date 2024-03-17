
import heapq

q_list = []
heapq.heapify(q_list)

heapq.heappush( q_list, (4, {'name':'d'}))
print(q_list)
heapq.heappush( q_list, (5, {'name':'e'}))
print(q_list)
heapq.heappush( q_list, (3, {'name':'c'}))
print(q_list)
heapq.heappush( q_list, (1, {'name':'a'}))
heapq.heappush( q_list, (2, {'name':'b'}))

while len(q_list)>0:
    data = heapq.heappop(q_list)
    print(data)
print("---")

class Test:
    def __init__(self,value):
        self.value=value
    def __eq__(self,other):
        print(f"__eq__ {self.value} {other}")
        return self.value == other
    def __lt__(self,other):
        print(f"__lt__ {self.value} {other}")
        return self.value < other
    def __le__(self,other):
        print(f"__le__ {self.value} {other}")
        return self.value <= other
    def __gt__(self,other):
        print(f"__gt__ {self.value} {other}")
        return self.value > other
    def __ge__(self,other):
        print(f"__ge__ {self.value} {other}")
        return self.value >= other
    def __bool__(self):
        print(f"__bool__ {self.value}")
        return self.value!=0
    def __int__(self):
        print(f"__int__ {self.value}")
        return int(self.value)
    def __float__(self):
        print(f"__float__ {self.value}")
        return float(self.value)
    def __str__(self):
        print(f"__str__ {self.value}")
        return str(self.value)

T=Test(3)

print( str(T) )
print( int(T) )
print( float(T) )
print( bool(T) )

print( "--BOOL--")
if T:
    pass

print( "--NotBOOL--")
if not T:
    pass

print( "--EQ--")
if 3 == T:
    pass

print( "--NE--")
if 4 != T:
    pass

print( "--LT--")
if T<4:
    pass

print( "--LE--")
if T<=4:
    pass

print( "--GT--")
if T>4:
    pass

print( "--GE--")
if T>=4:
    pass
if 4>=T:
    pass
B=True

if B>False:
    print(">")
else:
    print("!>")
