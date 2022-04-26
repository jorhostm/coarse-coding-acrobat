import enum
from idna import alabel
from numpy import sin,cos, pi, floor
import numpy as np
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
            self.dth1 = max(-4*pi, min(4*pi, self.dth1))

            self.th2 += t*self.dth2
            self.th2 = self.th2 - floor((self.th2 + pi)/(2*pi)) * 2*pi

            self.th1 += t*self.dth1
            self.th1 = self.th1 - floor((self.th1 + pi)/(2*pi)) * 2*pi

            p = self.get_points()
            self.xhistory.append([p[0], p[2], p[4]])
            self.yhistory.append([p[1], p[3], p[5]])

            if self.isterminal():
                break

                

        return reward

    def animate(self, interval=25):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set(xlim=(-2.1, 2.1), ylim=(-2.1, 2.1))

        

        line = ax.plot(self.xhistory[0], self.yhistory[0], color='k', lw=2)[0]
        line2 = ax.plot([-3. ,3], [1,1], "--k")

        def animate(i):
            
            line.set_ydata(self.yhistory[i])
            line.set_xdata(self.xhistory[i])

        anim = FuncAnimation(fig, animate, interval=interval, frames=len(self.xhistory)-1)
        
        anim.save('filename.mp4')

class Tiling:

    def __init__(self, bounds):
        self.bounds = []
        self.dims = 0
        for bound in bounds:
            if bound is None:
                self.bounds.append(None)
            
            else:
                a,b = bound
                self.bounds.append(np.linspace(a,b,5))
                self.dims += 1

        self.til = 0

    def get_num_tiles(self):
        return 6**self.dims
    
    
    def get_index(self, state):
        e = 0
        index = 0
        for i, arr in enumerate(self.bounds):
            if arr is not None:
                k = np.digitize(state[i] , arr)
                index += k * 6 ** e
                e += 1
        
        return index


def init_tilings():
    til = []
    np.random.seed(69)
    for i in range(12):
        lis = []
        lis.append((np.random.uniform(-pi, -0.5*pi) , np.random.uniform(0.5*pi, pi)))
        lis.append((np.random.uniform(-4*pi, -2*pi) , np.random.uniform(2*pi, 4*pi)))
        lis.append((np.random.uniform(-pi, -0.5*pi) , np.random.uniform(0.5*pi, pi)))
        lis.append((np.random.uniform(-9*pi, -4.5*pi) , np.random.uniform(4.5*pi, 9*pi)))
        
        til.append(Tiling(lis))

    for i in range(3):
        lis = []
        lis.append(None)
        lis.append((np.random.uniform(-4*pi, -2*pi) , np.random.uniform(2*pi, 4*pi)))
        lis.append((np.random.uniform(-pi, -0.5*pi) , np.random.uniform(0.5*pi, pi)))
        lis.append((np.random.uniform(-9*pi, -4.5*pi) , np.random.uniform(4.5*pi, 9*pi)))
        
        til.append(Tiling(lis))

    for i in range(3):
        lis = []
        lis.append((np.random.uniform(-pi, -0.5*pi) , np.random.uniform(0.5*pi, pi)))
        lis.append(None)
        lis.append((np.random.uniform(-pi, -0.5*pi) , np.random.uniform(0.5*pi, pi)))
        lis.append((np.random.uniform(-9*pi, -4.5*pi) , np.random.uniform(4.5*pi, 9*pi)))
        
        til.append(Tiling(lis))

    for i in range(3):
        lis = []
        lis.append((np.random.uniform(-pi, -0.5*pi) , np.random.uniform(0.5*pi, pi)))
        lis.append((np.random.uniform(-4*pi, -2*pi) , np.random.uniform(2*pi, 4*pi)))
        lis.append(None)
        lis.append((np.random.uniform(-9*pi, -4.5*pi) , np.random.uniform(4.5*pi, 9*pi)))
        
        til.append(Tiling(lis))

    for i in range(3):
        lis = []
        lis.append((np.random.uniform(-pi, -0.5*pi) , np.random.uniform(0.5*pi, pi)))
        lis.append((np.random.uniform(-4*pi, -2*pi) , np.random.uniform(2*pi, 4*pi)))
        lis.append((np.random.uniform(-pi, -0.5*pi) , np.random.uniform(0.5*pi, pi)))
        lis.append(None)
        
        til.append(Tiling(lis))

    #2
    for i in range(2):
        lis = []
        lis.append(None)
        lis.append(None)
        lis.append((np.random.uniform(-pi, -0.5*pi) , np.random.uniform(0.5*pi, pi)))
        lis.append((np.random.uniform(-9*pi, -4.5*pi) , np.random.uniform(4.5*pi, 9*pi)))
        
        til.append(Tiling(lis))

    for i in range(2):
        lis = []
        lis.append(None)
        lis.append((np.random.uniform(-4*pi, -2*pi) , np.random.uniform(2*pi, 4*pi)))
        lis.append(None)
        lis.append((np.random.uniform(-9*pi, -4.5*pi) , np.random.uniform(4.5*pi, 9*pi)))
        
        til.append(Tiling(lis))

    for i in range(2):
        lis = []
        lis.append(None)
        lis.append((np.random.uniform(-4*pi, -2*pi) , np.random.uniform(2*pi, 4*pi)))
        lis.append((np.random.uniform(-pi, -0.5*pi) , np.random.uniform(0.5*pi, pi)))
        lis.append(None)
        
        til.append(Tiling(lis))

    for i in range(2):
        lis = []
        lis.append((np.random.uniform(-pi, -0.5*pi) , np.random.uniform(0.5*pi, pi)))
        lis.append(None)
        lis.append(None)
        lis.append((np.random.uniform(-9*pi, -4.5*pi) , np.random.uniform(4.5*pi, 9*pi)))
        
        til.append(Tiling(lis))

    for i in range(2):
        lis = []
        lis.append((np.random.uniform(-pi, -0.5*pi) , np.random.uniform(0.5*pi, pi)))
        lis.append(None)
        lis.append((np.random.uniform(-pi, -0.5*pi) , np.random.uniform(0.5*pi, pi)))
        lis.append(None)
        
        til.append(Tiling(lis))

    for i in range(2):
        lis = []
        lis.append((np.random.uniform(-pi, -0.5*pi) , np.random.uniform(0.5*pi, pi)))
        lis.append((np.random.uniform(-4*pi, -2*pi) , np.random.uniform(2*pi, 4*pi)))
        lis.append(None)
        lis.append(None)
        
        til.append(Tiling(lis))

    for i in range(3):
        lis = []
        lis.append(None)
        lis.append(None)
        lis.append(None)
        lis.append((np.random.uniform(-9*pi, -4.5*pi) , np.random.uniform(4.5*pi, 9*pi)))
        
        til.append(Tiling(lis))

    for i in range(3):
        lis = []
        lis.append(None)
        lis.append(None)
        lis.append((np.random.uniform(-pi, -0.5*pi) , np.random.uniform(0.5*pi, pi)))
        lis.append(None)
        
        til.append(Tiling(lis))

    for i in range(3):
        lis = []
        lis.append(None)
        lis.append((np.random.uniform(-4*pi, -2*pi) , np.random.uniform(2*pi, 4*pi)))
        lis.append(None)
        lis.append(None)
        
        til.append(Tiling(lis))

    for i in range(3):
        lis = []
        lis.append((np.random.uniform(-pi, -0.5*pi) , np.random.uniform(0.5*pi, pi)))
        lis.append(None)
        lis.append(None)
        lis.append(None)
        
        til.append(Tiling(lis))

    return til


def features(tils,s):
    offset = 0
    indices = []

    for tiling in tils:
        indices.append(offset + tiling.get_index(s))
        offset += tiling.get_num_tiles()

    return indices



class Sarsa:

    def __init__(self, acrobat, alpha=0.2, lam=0.9, c=48, eps=0, tils=init_tilings(), max_steps=1000):
        self.acrobat = acrobat
        self.alpha = alpha
        self.lam = lam
        self.c = c
        self.eps = eps
        self.tils = tils
        self.max_steps = max_steps

        self.w_0 = np.zeros(18648)
        self.w_1 = np.zeros(18648)
        self.w_m1 = np.zeros(18648)
        self.w = [self.w_0, self.w_1,self.w_m1]

        self.e_0 = np.zeros(18648)
        self.e_1 = np.zeros(18648)
        self.e_m1 = np.zeros(18648)
        self.e = [self.e_0,self.e_1,self.e_m1]

        self.steps = []

    def greedy_policy(self, F):
        current_sum = -np.inf
        current_a = -1
        for a in range(-1, 2):
            sum_a = 0.
            for f in F:
                sum_a += self.w[a][f]
            
            if sum_a > current_sum:
                current_sum = sum_a
                current_a = a

            elif sum_a == current_sum:
                current_a = np.random.choice([current_a, a])

        return current_a, current_sum
            
    
    def run(self, num_trials=500):
        for _ in range(num_trials):
            print(f"Trial: {_}")
            self.acrobat.init()
            steps = 0
            s = self.acrobat.get_state()
            F = features(self.tils, s)
            a, Q = self.greedy_policy(F)
            
            while not self.acrobat.isterminal() and steps < 1000:
                for eb in self.e:
                    eb *= self.lam
                
                for b in range(-1, 2):
                    if b == a:
                        for f in F:
                            self.e[b][f] = 1

                    else:
                        for f in F:
                            self.e[b][f] = 0

                r = self.acrobat.do_action(a)
                new_s = self.acrobat.get_state()

                new_F = []
                new_a = 0
                new_Q = 0
                if not self.acrobat.isterminal():
                    new_F = features(self.tils,new_s)
                    new_a, new_Q = self.greedy_policy(new_F)

                for b in range(-1, 2):
                    self.w[b] += self.alpha/self.c * (r + new_Q - Q)*self.e[b]

                a = new_a
                s = new_s
                F = new_F

                steps += 1

            self.steps.append(steps)

                

