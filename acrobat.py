from numpy import sin,cos, pi, floor
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

class Acrobat:

    def __init__(self):
        self.th1 = 0
        self.dth1 = 0
        self.th2 = 0
        self.dth2 = 0

        self.xhistory = [[0., 0., 0.]]
        self.yhistory = [[0., 0., 0.]]

    def init(self):
        self.th1 = 0
        self.dth1 = 0
        self.th2 = 0
        self.dth2 = 0

        self.xhistory = [[0., 0., 0.]]
        self.yhistory = [[0., 0., 0.]]

    def get_state(self):
        return [self.th1, self.dth1, self.th2, self.dth2]

    def get_points(self):
        xp2 = sin(self.th1)
        yp2 = -cos(self.th1)
        th3 = self.th1 + self.th2
        xtip = xp2 + sin(th3)
        ytip = yp2 - cos(th3)

        return [0.,0.,xp2,yp2,xtip,ytip]

    def isterminal(self):
        points = self.get_points()
        ytip = points[-1]

        return ytip >= 1

    def do_action(self, F):
        g = 9.8
        t = 0.05
        reward = -1
        for _ in range(4):
            th1 = self.th1
            th2 = self.th2
            dth1 = self.dth1
            dth2 = self.dth2
            
            ph2 = 1*0.5*g*cos(th1 + th2 - pi/2)
            ph1 = -1*1*0.5*dth2**2 * sin(th2) - 2*1*1*0.5*dth2*dth1*sin(th2) + (1*0.5+1*1)*g*cos(th1 - pi/2) + ph2

            d2 = 1*(0.5**2 + 1*0.5*cos(th2)) + 1
            d1 = 1*0.5**2 + 1*(1 + 0.5**2 + 2*1*0.5*cos(th2)) + 2

            ddth2 = 1/(1*0.5**2 + 1 - d2**2 / d1) * (F + d2/d1*ph1 - 1*1*0.5*dth1**2 * sin(th2) - ph2)
            ddth1  = -(d2*ddth2 + ph1)/d1

            self.dth2 += t*ddth2
            self.dth2 = max(-9*pi, min(9*pi, self.dth2))

            self.dth1 += t*ddth1
            self.dth21 = max(-4*pi, min(4*pi, self.dth1))

            self.th2 += t*self.dth2
            self.th2 = self.th2 - floor((self.th2 + pi)/(2*pi)) * 2*pi

            self.th1 += t*self.dth1
            self.th1 = self.th1 - floor((self.th1 + pi)/(2*pi)) * 2*pi

            p = self.get_points()
            self.xhistory.append([p[0], p[2], p[4]])
            self.yhistory.append([p[1], p[3], p[5]])

                

        return reward

    def animate(self, interval=50):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set(xlim=(-2.1, 2.1), ylim=(-2.1, 2.1))

        

        line = ax.plot(self.xhistory[0], self.yhistory[0], color='k', lw=2)[0]
        line2 = ax.plot([-3. ,3], [1,1], "--k")

        def animate(i):
            
            line.set_ydata(self.yhistory[i])
            line.set_xdata(self.xhistory[i])

        anim = FuncAnimation(fig, animate, interval=interval, frames=len(self.xhistory)-1)
        
        anim.save('filename.mp4')
