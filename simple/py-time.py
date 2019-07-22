'''
import time
tref = time.time()
count = 0
while time.time()-tref <=5.0:
    #do nothing
    count +=1


print("Done waiting")
'''

import time

def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        print(timeformat, end='\r')
        time.sleep(1)
        t -= 1
    print('Goodbye')
    

countdown(10)


