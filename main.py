import torch

class SVGD:
    def __init__(self, n_particles, n_dim):
        '''
        :param n_particles: int
        :param n_dim: int
        '''
        self.n_particles = n_particles
        self.particles = torch.randn(self.n_particles, n_dim) * 5

    def step(self, target_distribution_logprob, kernel, learning_rate=0.1):
        '''

        :param target_distribution_logprob: pytorch/pyro callable, return log_prob for each input x
        :param kernel: pytorch callable, need to be able to broadcast for one input
        :param learning_rate: float
        :return: None
        '''

        def dlog():
            particles = self.particles.clone().detach()
            particles.requires_grad_()
            torch.sum(target_distribution_logprob(particles)).backward()  # Because sum is linear
            return particles.grad

        def dkernel(x):
            particles = self.particles.clone().detach()
            particles.requires_grad_()
            torch.sum(kernel(particles, x)).backward()
            return particles.grad

        phi = lambda x: torch.mean(kernel(self.particles, x).reshape(-1, 1) * dlog() + dkernel(x), dim=0)
        phis = torch.stack([phi(self.particles[i, :]) for i in range(self.n_particles)], dim=0)
        self.particles = self.particles + learning_rate * phis

        return self.particles

