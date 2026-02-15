from mydatacollectionapp import MyDataCollectionApp
import serial

mot_serial=serial.Serial('COM9', 9600, timeout=1)

def cbJoyPos(joystickPosition):
    posX, posY = joystickPosition
    
    #car direction
    #if cmd=='w': rcCar.goForward()
    #elif cmd=='q': rcCar.goForwardLeft()
    #elif cmd=='e': rcCar.goForwardRight()
    #elif cmd=='s': rcCar.stop()
    #elif cmd=='x': rcCar.goBackward()
    #elif cmd=='z': rcCar.goBackwardLeft()
    #elif cmd=='c': rcCar.goBackwardRight()
    
    
    command = 's'
    collect_data = 1
    
    #forwarding
    if posY > 0.3:
        if posX > 0.4:
            command = 'e'
    
        elif posX < -0.4:
            command= 'q'
            
        else:
            command = 'w'
    
    #backwarding
    elif posY < -0.3:
        if posX > 0.4:
            command = 'c'
    
        elif posX < -0.4:
            command = 'z'
        
        else:
            command = 'x'
    
    else:
        command = 's'
    
    
    if command == 'w':   #forward
        right, left = 0, 0
    
    elif command == 'q':  #left
        right, left = 1, 0
    
    elif command == 'e': #right
        right, left = 0 , 1
    
    elif command == 's':  #stop
        right, left = 1, 1
    
    else:
        right, left = 1, 1
        
    rl = collect_data << 2 | right << 1 | left
    myDataCollectionApp.setRL(rl)
    
    mot_serial.write(command.encode())

    
v_source="http://192.168.4.1:81/stream"
myDataCollectionApp = MyDataCollectionApp(cbJoyPos= cbJoyPos, v_source=v_source)
myDataCollectionApp.run()


