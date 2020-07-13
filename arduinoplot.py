import serial 
import matplotlib.pyplot as plt
import numpy as np
from drawnow import *


voltage=[]

ser = serial.Serial('COM3',9600)
plt.ion()
fig=plt.figure()
i=0 
def makeFig(): #Create a function that makes our desired plot
    plt.ylim(0,5)                                 #Set y min and max values
    plt.title('My Live Streaming Sensor Data')      #Plot the title
    plt.grid(True)                                  #Turn the grid on
    plt.ylabel('amplitude')                            #Set ylabels
    plt.plot(voltage, 'ro-', label='Degrees F')       #plot the temperature
    plt.legend(loc='upper left')   
 plt.ylim(93450,93525)

 while True: # While loop that loops forever
    while (ser.inWaiting()==0): #Wait here until there is data
        pass #do nothing
    ser = ser.readline() #read the line of text from the serial port
    dataArray = arduinoString.split(',')   #Split it into an array called dataArray
    temp = float( dataArray[0])            #Convert first element to floating number and put in temp
            #Convert second element to floating number and put in P
    voltage.append(temp)                     #Build our voltage array by appending temp readings
                     #Building our pressure array by appending P readings
    drawnow(makeFig)                       #Call drawnow to update our live graph
    plt.pause(.000001)                     #Pause Briefly. Important to keep drawnow from crashing
    cnt=cnt+1
    if(cnt>50):                            #If you have 50 or more points, delete the first one from the array
        voltage.pop(0)                       #This allows us to just see the last 50 data points
        pressure.pop(0)
