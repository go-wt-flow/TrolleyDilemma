from _12_servo_motor import ServoMotor

class SteeringMotor(ServoMotor):
    def __init__(self, pwm):
        super().__init__(pwm)
        
    
    def turnLeft(self):
        self.setAngle(0)
    
    def turnForward(self):
        self.setAngle(int(90*1.1))

    
    def turnRight(self):
        self.setAngle(180)
    
    
if __name__=='__main__':
    steeringMotor=SteeringMotor(7)
        
    
    while True:
        cmd = input('left:l, middle:m, right:r >> ')
        if cmd =='l':
            steeringMotor.turnLeft()
            
        elif cmd =='m':
            steeringMotor.turnForward()
            
        elif cmd == 'r':
            steeringMotor.turnRight()
