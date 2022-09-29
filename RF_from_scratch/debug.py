def generate_x(r, c, d):
    x = r
    if c==0:
        return x
    else: # c>0
        for i in range(1,c+1):
            breakpoint()
            if i % 2 == 1:
                x += 2**(d-r+1)-1
            else: # i % 2 == 0
                x += 2**(d-r+1)
        return x

def isleft(n,d,r):
    for i in range(2**r+1):
        if n == generate_x(r, i, d):
            if i%2 == 1:
                return False
            else:
                return True

counter = 4
node_id = 11
node_search = node_id
result = [node_id]
r = counter
while r > 0:
    if isleft(node_search,counter,r):
        print("left")
        node_search = node_search-1
    else:
        print("right")
        node_search = node_search-2
    result.append(node_search)
    r-=1
result.append(r)
