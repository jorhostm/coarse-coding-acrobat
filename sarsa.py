import numpy as np
from tiling import features


class Sarsa:
    def __init__(self, acrobat, alpha=0.001, lam=0.9, eps=0, tils=[], max_steps=1000):
        self.acrobat = acrobat
        self.alpha = alpha
        self.lam = lam
        self.c = len(tils)
        self.eps = eps
        self.tils = tils
        self.max_steps = max_steps
        self.len_e_w = 0

        for tile in self.tils:
            self.len_e_w += tile.get_num_tiles()

        self.w_0 = np.zeros(self.len_e_w)
        self.w_1 = np.zeros(self.len_e_w)
        self.w_m1 = np.zeros(self.len_e_w)
        self.w = [self.w_0, self.w_1, self.w_m1]

        self.e_0 = np.zeros(self.len_e_w)
        self.e_1 = np.zeros(self.len_e_w)
        self.e_m1 = np.zeros(self.len_e_w)
        self.e = [self.e_0, self.e_1, self.e_m1]

        self.steps = []

    def reset_eligibility(self):
        self.e_0 = np.zeros(self.len_e_w)
        self.e_1 = np.zeros(self.len_e_w)
        self.e_m1 = np.zeros(self.len_e_w)
        self.e = [self.e_0, self.e_1, self.e_m1]

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

    def run(self, num_trials=100):
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
                            self.e[b][f] += 1

                r = self.acrobat.do_action(a)
                new_s = self.acrobat.get_state()

                new_F = []
                new_a = 0
                new_Q = 0
                if not self.acrobat.isterminal():
                    new_F = features(self.tils, new_s)
                    new_a, new_Q = self.greedy_policy(new_F)

                for b in range(-1, 2):
                    cons = (self.alpha / self.c) * (r + new_Q - Q)
                    update = cons * self.e[b]
                    self.w[b] += update

                a = new_a
                F = new_F

                steps += 1

            self.steps.append(steps)
            self.reset_eligibility()


