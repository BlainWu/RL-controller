import numpy as np
import matplotlib.pyplot as plt


# generate multi-arm bandit
class BernoulliBandit:
    def __init__(self,K):
        self.K = K
        self.probs = np.random.uniform(size=K)
        self.best_id = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_id]

    def step(self,k):
        # chose k-th arm
        return 1 if np.random.rand() > self.probs[k] else 0

np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)
print("Generated a {}-arm bandit.".format(K))
print("The best choice is {}-th arm, and the probability is {}".format(bandit_10_arm.best_id,bandit_10_arm.best_prob))


class Solver:
    def __init__(self,bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0
        self.actions = [] # record actions
        self.regrets = [] # record regrets

    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def update_actions(self,k):
        self.actions.append(k)

    def run_one_step(self):
        # chose the arm to use
        raise NotImplementedError

    def run(self,num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.update_actions(k)
            self.update_regret(k)


class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy,self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.rand() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1./(self.counts[k] + 1)*(r-self.estimates[k])

        return k


def plot_results(solvers, solver_names):
    for idx,solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
epsilon_greedy_solver.run(5000)
print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])