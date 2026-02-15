from  DCmotor_class import DCMotor
from _13_steering_motor import SteeringMotor
import time

class RCCar:
    def __init__(self, dcMotor, steeringMotor):
        self.dcMotor=dcMotor
        self.steeringMotor=steeringMotor
        
    def goForward(self):
        self.dcMotor.rotateForward()
        self.steeringMotor.turnForward()

    def goForwardLeft(self):
        self.dcMotor.rotateForward()
        self.steeringMotor.turnLeft()
    
    def goForwardRight(self):
        self.dcMotor.rotateForward()
        self.steeringMotor.turnRight()
        
    def goBackward(self):
        self.dcMotor.rotateBackward()
        self.steeringMotor.turnForward()
        
    def goBackwardLeft(self):
        self.dcMotor.rotateBackward()
        self.steeringMotor.turnLeft()
        
    def goBackwardRight(self):
        self.dcMotor.rotateBackward()
        self.steeringMotor.turnRight()
        
    def stop(self):
        self.dcMotor.stop()
        self.steeringMotor.turnForward()  

if __name__=='__main__':
    
    dcMotor = DCMotor(1,0)
    steeringMotor = SteeringMotor(7)
    
    rcCar = RCCar(dcMotor, steeringMotor)
    
    rcCar.goForward()
    time.sleep(1)
    
    rcCar.goForwardLeft()
    time.sleep(1)
    
    rcCar.goForwardRight()
    time.sleep(1)
    
    rcCar.stop()
        
