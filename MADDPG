class MADDPG:
   
        # TODO: Should I use DDPG and extend
        self.agents = [
            DDPG(state_dim, action_dim, critic_dim, hidden_dim, actor_lr, critic_lr, device)
            for state_dim, action_dim in zip(state_dims, action_dims)
        ]
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = nn.MSELoss()


return 0;
