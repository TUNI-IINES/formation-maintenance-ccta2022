import time

class timeProfiling():
    def __init__(self, label='', min=1000, max=-1000): # all in seconds
        self.label = label
        self.min = min
        self.max = max
        self.ave = 0
        self.n = 0
        self.total = 0

    def startTimer(self):
        self.currentStartTime = time.time()
    
    def stopShowElapsed(self):
        print("Time Elapsed - {} :{}".format( self.label, (time.time() - self.currentStartTime) ))

    def stopTimer(self):
        t_elapsed = time.time() - self.currentStartTime
        if t_elapsed < self.min: self.min = t_elapsed
        if t_elapsed > self.max: self.max = t_elapsed

        self.total += t_elapsed
        self.n += 1
        self.ave = self.total / self.n

        #n = self.n
        #self.ave = (self.ave*n + t_elapsed)/(n+1)
        #self.n += 1



    def printStatus(self):
        print('Comp Stat: {} >> min: {:.2f} ms, ave: {:.2f} ms, max: {:.2f} ms, total: {:.2f} ms'.format( \
                self.label, self.min*1000, self.ave*1000, self.max*1000, self.total*1000 ))


