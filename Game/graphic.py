import time
import keyboard

global x, y
#x = 0
#y = 0
tail = [[0,0], [1,0], [2,0], [3,0]]
direction = "right"
add_point = False

board = [["•" for i in range(10)] for i in range(10)]

def update_tail():
    global add_point, tail, direction

    buff=[]
    if add_point:
        buff= tail[-1].copy()
        
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

    if add_point:
        tail.append(buff)
        print("APPENDED TAIL")
        add_point = False

def render():
    board = [["•" for i in range(10)] for i in range(10)]
    for points in tail:
        board[points[0]][points[1]] = "X"

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
   global board, x, y, direction
   # keyboard.add_hotkey('up', up)
   # keyboard.add_hotkey('down', down)
   # keyboard.add_hotkey('left', left)
   # keyboard.add_hotkey('right', right)
   # keyboard.wait()
   pp=3
   while True:
       keyboard.add_hotkey('up', up)
       keyboard.add_hotkey('down', down)
       keyboard.add_hotkey('left', left)
       keyboard.add_hotkey('right', right)
       time.sleep(0.2)
       print(tail)
       update_tail()
       render()
game()

keyboard.wait()
