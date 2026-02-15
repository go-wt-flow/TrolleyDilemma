from machine import Pin, PWM

class ServoMotor:
    
    MIN_ANGLE=1600
    MAX_ANGLE=7900
    MID_ANGLE=(MIN_ANGLE+MAX_ANGLE)//2
    DEGREE_1=(MAX_ANGLE-MIN_ANGLE)//180
    
    def __init__(self, pwm):
        self.pwm=PWM(Pin(pwm))
        self.pwm.freq(50)
        
    
    def setDutyCycle(self, position):
        self.pwm.duty_u16(position)
        
    
    def setAngle(self, angle):
        if 0<=int(angle)<=180:
            self.setDutyCycle(self.MIN_ANGLE+self.DEGREE_1*angle)
            
    
    if __name__=='__main__':
        
        import time
        
        servoMotor=ServoMotor(7)
        
        for i in range(3):
            for angle in range(0, 180, 1):
                servoMotor.setAngle(angle)
                time.sleep(0.01)
            
            for angle in range(180, 00, -1):
                servoMotor.setAngle(angle)
                time.sleep(0.01)
            
        
        servoMotor.setAngle(90)
        
        

