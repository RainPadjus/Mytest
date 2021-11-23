import time
import keyboard
import random

x = False
y = False

tail = [[0,0]]
direction = "right"
add_point = False
i=5
board = [["•" for i in range(10)] for i in range(10)]

def add():
    global tail
    tail.append(tail[-1].copy())

def update_tail():
    global add_point, tail, direction, x, y

    buff=[]
        
    for i in range(len(tail)-1):
        #print(tail[len(tail)-i-1])
        #print(tail[len(tail)-i-2])
        tail[len(tail)-i-1] = tail[len(tail)-i-2].copy()

    #print(tail)
    if direction == "up":
        tail[0][0] = (tail[0][0]-1)%10
    
    if direction == "down":
        tail[0][0] = (tail[0][0]+1)%10

    if direction == "left":
        tail[0][1] = (tail[0][1]-1)%10
    
    if direction == "right":
        tail[0][1] = (tail[0][1]+1)%10
    
    if (tail[0][0] == x) and (tail[0][1] == y):
        x=False
        y=False
        add()
def render():
    global x,y
    board = [["•" for i in range(10)] for i in range(10)]
    for points in tail:
        board[points[0]][points[1]] = "X"
    if x and y: board[x][y] = "O"
    print("\033[F"*30)
    for i in board:
        for j in i:
            print(j,"", end="")
        print("", end= "\n")

def up():
    global direction
    direction = 'up'

def down():
    global direction
    direction = 'down'

def right():
    global direction
    direction = 'right'

def left():
    global direction
    direction = 'left'

def game():
   global x, y, board , direction
   a = time.time()
   while True:
       keyboard.add_hotkey('up', up)
       keyboard.add_hotkey('down', down)
       keyboard.add_hotkey('left', left)
       keyboard.add_hotkey('right', right)
       time.sleep(0.5)
       update_tail()
       render()
       
       if (time.time() - a) >= 10:
           x = random.randint(0, 9)
           y = random.randint(0, 9)
           a = time.time()
game()

keyboard.wait()
