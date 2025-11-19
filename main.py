import torch
from environment.cartpole_environment import CartPoleEnv
from agent.a2c_agent import A2CAgent
from agent.dqn_agent import DQNAgent
from agent.ppo_agent import PPOAgent
from core.model import NeuralNetwork
from core.interface import Interface
from core.runner import run_pipeline

if __name__ == "__main__":
    interface = Interface()

    env = CartPoleEnv()

    mode = interface.ask_mode()  # "dqn" or "a2c" or "ppo"

    if mode == "dqn":
        agent = DQNAgent(
            NeuralNetwork(lr=1e-4),
            buffer_size=10000,
            batch_size=64,
            epsilon=0.9
        )

        if interface.ask_load_dqn():
            try:
                agent.nn.load_state_dict(torch.load(interface.path))
                print("DQN Model not find, lauch...")
                interface.didtrainfct()
            except FileNotFoundError:
                print("DQN Model not find, lauch...")
                interface.didtrain = True
                interface.episodes = int(input("How many episodes would you like to train the model for? "))
        else:
            interface.didtrain = True
            interface.episodes = int(input("How many episodes would you like to train the model for? "))

    elif mode == "a2c":
        actor = NeuralNetwork(
            input_dim=4,
            hidden_dim=64,
            output_dim=2,
            mode="actor",
            lr=3e-4
        )
        critic = NeuralNetwork(
            input_dim=4,
            hidden_dim=64,
            output_dim=1,
            mode="critic",
            lr=1e-3
        )

        agent = A2CAgent(actor_nn=actor, critic_nn=critic)

        if interface.ask_load_a2c():
            try:
                agent.nna.load_state_dict(torch.load(interface.path.replace(".pth", "_actor.pth")))
                agent.nnc.load_state_dict(torch.load(interface.path.replace(".pth", "_critic.pth")))
                print("Model A2C find.")
                interface.didtrainfct()
            except FileNotFoundError:
                print("A2C Model not find, lauch...")
                interface.didtrain = True
                interface.episodes = int(input("How many episodes would you like to train the model for? "))
        else:
            interface.didtrain = True
            interface.episodes = int(input("How many episodes would you like to train the model for? "))

    elif mode == "ppo":
        actor = NeuralNetwork(
            input_dim=4,
            hidden_dim=64,
            output_dim=2,
            mode="actor",
            lr=3e-4
        )
        critic = NeuralNetwork(
            input_dim=4,
            hidden_dim=64,
            output_dim=1,
            mode="critic",
            lr=1e-3
        )

        agent = PPOAgent(
            actor_nn=actor,
            critic_nn=critic,
            buffer_size=1024,
            entropy_bonus=False  # Already False by default for CartPole
        )

        if interface.ask_load_ppo():
            try:
                agent.nna.load_state_dict(torch.load(interface.path.replace(".pth", "_actor.pth")))
                agent.nnc.load_state_dict(torch.load(interface.path.replace(".pth", "_critic.pth")))
                print("Model PPO find.")
                interface.didtrainfct()
            except FileNotFoundError:
                print("DQN Model not find, lauch...")
                interface.didtrain = True
                interface.episodes = int(input("How many episodes would you like to train the model for? "))
        else:
            interface.didtrain = True
            interface.episodes = int(input("How many episodes would you like to train the model for? "))

    run_pipeline(env=env, agent=agent, interface=interface, mode=mode)