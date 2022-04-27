from sarsa import Sarsa
from environment import Acrobat
import numpy as np
from numpy import sin, cos, pi, floor
from tiling import Tiling, init_tilings
import matplotlib.pyplot as plt

number_of_tilings = 2
bins = 10

ac1 = Acrobat()

lis = [(-0.8*pi, 0.8*pi),
       (-3*pi, 3*pi),
       (-0.8*pi, 0.8*pi),
       (-7*pi, 7*pi)]

lis2 = [(-0.9*pi, 0.7*pi),
       (-3.5*pi, 2.5*pi),
       (-0.9*pi, 0.7*pi),
       (-6*pi, 8*pi)]

til = Tiling(lis, 10)
til2 = Tiling(lis2, 10)

til3 = init_tilings(number_of_tilings, bins, bins)

s1 = Sarsa(ac1, tils=[til, til2])

s1.run(500)
plt.plot(s1.steps)
plt.show()
print(s1.steps)

# ac1.animate()
