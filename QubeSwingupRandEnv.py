from scipy.integrate import odeint
from numba import jit
import numpy as np
import gym

from gym_brt.envs.rendering import QubeRenderer
from gym_brt.control import flip_and_hold_policy


# @jit
def phys_params(
    var_fn=lambda mu, name: 0.1 * mu,
    Rm=8.4,
    kt=0.042,
    km=0.042,
    mr=0.095,
    Lr=0.085,
    Dr=0.00027,
    mp=0.024,
    Lp=0.129,
    Dp=0.00005,
    g=9.81,
):
    """Randomly sample values for the physical parameters

    Arguments:
        var_fn: Variance as a function of the mean and name for each parameter
        Rm: Motor - Resistance
        kt: Motor - Current-torque (N-m/A)
        km: Motor - Back-emf constant (V-s/rad)
        mr: Rotary Arm - Mass (kg)
        Lr: Rotary Arm - Total length (m)
        Dr: Rotary Arm - Equivalent viscous damping coefficient (N-m-s/rad)
        mp: Pendulum Link - Mass (kg)
        Lp: Pendulum Link - Total length (m)
        Dp: Pendulum Link - Equivalent viscous damping coefficient (N-m-s/rad)
        g : Gravity constant

    Return:
        params: Numpy array with all the parameters
            order: [Rm, kt, km, mr, Lr, Dr, mp, Lp, Dp, g, Jr, Jp]
    """
    # Motor
    Rm += np.random.rand() * var_fn(Rm, "Rm")
    kt += np.random.rand() * var_fn(kt, "kt")
    km += np.random.rand() * var_fn(km, "km")
    # Rotary Arm
    mr += np.random.rand() * var_fn(mr, "mr")
    Lr += np.random.rand() * var_fn(Lr, "Lr")
    Dr += np.random.rand() * var_fn(Dr, "Dr")
    # Pendulum Link
    mp += np.random.rand() * var_fn(mp, "mp")
    Lp += np.random.rand() * var_fn(Lp, "Lp")
    Dp += np.random.rand() * var_fn(Dp, "Dp")
    # Gravity
    g += np.random.rand() * var_fn(g, "g")

    Jr = mr * Lr ** 2 / 12  # Rotary arm: Moment of inertia about pivot (kg-m^2)
    Jp = mp * Lp ** 2 / 12  # Pendulum: Moment of inertia about pivot (kg-m^2)

    return np.array([Rm, kt, km, mr, Lr, Dr, mp, Lp, Dp, g, Jr, Jp])


# @jit
def diff_ns_ode(state, t, action, phys_params, dt):
    # def diff_ns_ode(state, t, action, dt):
    #     Rm = 8.4
    #     kt = 0.042
    #     km = 0.042
    #     mr = 0.095
    #     Lr = 0.085
    #     Dr = 0.00027
    #     mp = 0.024
    #     Lp = 0.129
    #     Dp = 0.00005
    #     g = 9.1
    #     Jr = mr * Lr ** 2 / 12  # Rotary arm: Moment of inertia about pivot (kg-m^2)
    #     Jp = mp * Lp ** 2 / 12  # Pendulum: Moment of inertia about pivot (kg-m^2)
    Rm, kt, km, mr, Lr, Dr, mp, Lp, Dp, g, Jr, Jp = phys_params

    Vm = action[0]  # Voltage applied to the motor
    theta, alpha, theta_dot, alpha_dot = state
    tau = -(km * (Vm - km * theta_dot)) / Rm  # torque

    # fmt: off
    # From Rotary Pendulum Workbook
    theta_dot_dot = float(-Lp*Lr*mp*(-8.0*Dp*alpha_dot + Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha))*np.cos(alpha) + (4.0*Jp + Lp**2*mp)*(4.0*Dr*theta_dot + Lp**2*alpha_dot*mp*theta_dot*np.sin(2.0*alpha) + 2.0*Lp*Lr*alpha_dot**2*mp*np.sin(alpha) - 4.0*tau))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp))
    alpha_dot_dot = float(2.0*Lp*Lr*mp*(4.0*Dr*theta_dot + Lp**2*alpha_dot*mp*theta_dot*np.sin(2.0*alpha) + 2.0*Lp*Lr*alpha_dot**2*mp*np.sin(alpha) - 4.0*tau)*np.cos(alpha) - 0.5*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp)*(-8.0*Dp*alpha_dot + Lp**2*mp*theta_dot**2*np.sin(2.0*alpha) + 4.0*Lp*g*mp*np.sin(alpha)))/(4.0*Lp**2*Lr**2*mp**2*np.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*np.sin(alpha)**2 + 4.0*Lr**2*mp))
    # fmt: on

    diff_state = np.array([theta_dot, alpha_dot, theta_dot_dot, alpha_dot_dot])
    return diff_state


# @jit
def next_state_ode(state, action, phys_params, dt=0.004):
    t = np.linspace(0.0, dt, 2)

    next_state = odeint(diff_ns_ode, state, t, args=(action, phys_params, dt))[1, :]
    # next_state = odeint(diff_ns_ode, state, t, args=(action, dt))[1, :]

    theta, alpha, theta_dot, alpha_dot = next_state
    theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
    alpha = ((alpha + np.pi) % (2 * np.pi)) - np.pi

    return np.array([theta, alpha, theta_dot, alpha_dot])


class QubeSwingupEnv(gym.Env):
    def __init__(self, use_simulator=None, frequency=250):
        self.state = np.array([0, 0, 0, 0])
        self._viewer = None
        self.pp = phys_params(lambda mu, name: 0.1 * mu)
        self.dt = 1 / frequency

        act_max = np.asarray([3], dtype=np.float32)
        obs_max = np.asarray([np.pi / 2, np.pi, np.inf, np.inf], dtype=np.float32)

        self.observation_space = gym.spaces.Box(-obs_max, obs_max)
        self.action_space = gym.spaces.Box(-act_max, act_max)

    def _reward(self):
        theta = self.state[0]
        alpha = self.state[1]
        reward = 1 - (0.8 * np.abs(alpha) + 0.2 * np.abs(theta)) / np.pi
        return reward

    def _done(self):
        theta = self.state[0]
        done = abs(theta) > (90 * np.pi / 180)
        return done

    def reset(self):
        # Start the pendulum stationary at the bottom (stable point)
        self.state = np.array([0, np.pi, 0, 0]) + np.random.randn(4) * 1.0
        # Generate new physical parameters
        self.pp = phys_params(lambda mu, name: 0.1 * mu)
        return self.state

    def step(self, action):
        self.state = next_state_ode(self.state, action, self.pp, self.dt)
        reward = self._reward()
        done = self._done()
        info = {}
        return self.state, reward, done, info

    def render(self, mode="human"):
        theta = self.state[0]
        alpha = self.state[1]
        if self._viewer is None:
            self._viewer = QubeRenderer(theta, alpha, 250)
        self._viewer.render(theta, alpha)


def main():
    num_episodes = 1200
    num_steps = 2500  # 10 seconds if frequency is 250Hz/period is 0.004s
    env = QubeSwingupEnv()

    for episode in range(num_episodes):
        state = env.reset()

        for step in range(num_steps):
            action = flip_and_hold_policy(state)
            state, reward, done, info = env.step(action)
            print(state)
            env.render()
            if done:
                break


if __name__ == "__main__":
    main()
