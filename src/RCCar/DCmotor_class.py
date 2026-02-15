from machine import Pin

class DCMotor:
    def __init__(self, C1, C2):
        self.C1=Pin(C1, Pin.OUT)
        self.C2=Pin(C2, Pin.OUT)
        
    
    def rotateForward(self):
        self.C1.value(0)
        self.C2.value(1)
        
    
    def rotateBackward(self):
        self.C1.value(1)
        self.C2.value(0)
        
    
    def stop(self):
        self.C1.value(0)
        self.C2.value(0)
    
    
    if __name__=='__main__':
        import time
        
        dcMotor = DCMotor(1,0)
        
        dcMotor.rotateForward()
        time.sleep(3)
        
        dcMotor.stop()
        time.sleep(3)
        
        dcMotor.rotateBackward()
        time.sleep(3)
        
        dcMotor.stop()
        
    
