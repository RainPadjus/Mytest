import time
import keyboard
import random

x = False
y = False

game = True
tail = [[0,0]]
direction = "right"
add_point = False
i=5
board = [["•" for i in range(10)] for i in range(10)]

def add():
    global tail
    tail.append(tail[-1].copy())

def update_tail():
    global game, add_point, tail, direction, x, y

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
    
    if tail[0] in tail[1:]:
        game = False
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
    if direction != "down":
        direction = 'up'
    elif direction == 'down' and len(tail)==1:
        direction = 'up'
    else: pass

def down():
    global direction
    if direction != "up":
        direction = 'down'
    elif direction =='up' and len(tail)==1:
        direction = 'down'
    else: pass

def right():
    global direction
    if direction != "left":
        direction = 'right'
    elif direction == 'left' and len(tail)==1:
        direction ='right'
    else: pass

def left():
    global direction
    if direction != "right":
        direction = 'left'
    elif direction == "right" and len(tail)==1:
        direction = 'left'
    else: pass

def game():
   global game, x, y, board , direction
   a = time.time()
   while game:
       keyboard.add_hotkey('up', up)
       keyboard.add_hotkey('down', down)
       keyboard.add_hotkey('left', left)
       keyboard.add_hotkey('right', right)
       time.sleep(0.2)
       update_tail()
       render()
       
       if (time.time() - a) >= 5:
           x = random.randint(0, 9)
           y = random.randint(0, 9)
           a = time.time()
   print("GAME OVER : FINAL SCORE {}".format(len(tail)))
game()

keyboard.wait()
