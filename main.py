import numpy as np
from environment.cartpole_environment import CartPoleEnv
from agent.a2c_agent import A2CAgent
from agent.dqn_agent import DQNAgent
from agent.ppo_agent import PPOAgent
from core.common_methods_agent import NeuralNetwork
from core.interface import Interface
import torch
import os
import warnings
warnings.filterwarnings("ignore")

def main():
    # Entraînement
    if interface.didtrain:
        agent.train(env, interface.episodes)
        print(f"Entraînement terminé pour {interface.episodes} épisodes.")

        # Sauvegarde
        if mode == "dqn":
            if interface.ask_save_dqn():
                os.makedirs(os.path.dirname(interface.path), exist_ok=True)
                torch.save(agent.nn.state_dict(), interface.path)

        elif mode == "a2c":
            if interface.ask_save_a2c():
                os.makedirs(os.path.dirname(interface.path), exist_ok=True)
                torch.save(agent.nna.state_dict(), interface.path.replace(".pth", "_actor.pth"))
                torch.save(agent.nnc.state_dict(), interface.path.replace(".pth", "_critic.pth"))
        
        elif mode == "ppo":
            if interface.ask_save_ppo():
                os.makedirs(os.path.dirname(interface.path), exist_ok=True)
                torch.save(agent.nna.state_dict(), interface.path.replace(".pth", "_actor.pth"))
                torch.save(agent.nnc.state_dict(), interface.path.replace(".pth", "_critic.pth"))
        
    # Test
    if interface.didtestfct():
        agent.test_agent(env, testepisodes=100)
        
    if interface.didgraphicfct():
        interface.grahic_name = "assets/" # Dossier par défaut
        interface.grahic_name += input("Enter the filename to save the graphic (without extension, default is cartpole.gif): ")
        if interface.grahic_name:
            interface.grahic_name += ".gif"
        agent.graphic_agent(filename=interface.grahic_name if interface.grahic_name else "cartpole.gif")
        print(f"Graphic saved as {interface.grahic_name if interface.grahic_name else 'cartpole.gif'}")

    env.close()


if __name__ == "__main__":
    interface = Interface()
    env = CartPoleEnv()
    mode = interface.ask_mode()  # "dqn" or "a2c" or "ppo"

    if mode == "dqn":
        agent = DQNAgent(NeuralNetwork(lr=1e-4), buffer_size=10000, batch_size=64, epsilon=0.9)

        if interface.ask_load_dqn():
            try:
                agent.nn.load_state_dict(torch.load(interface.path))
                print("Modèle DQN chargé.")
                interface.didtrainfct()
            except FileNotFoundError:
                print("Modèle DQN non trouvé, démarrage du programme...")
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
            mode="actor"
            lr=3e-4
        )

        critic = NeuralNetwork(
            input_dim=4,
            hidden_dim=64,
            output_dim=1,
            mode="critic"
            lr=1e-3
        )
        
        agent = A2CAgent(actor_nn=actor, critic_nn=critic)

        if interface.ask_load_a2c():
            try:
                agent.nna.load_state_dict(torch.load(interface.path.replace(".pth", "_actor.pth")))
                agent.nnc.load_state_dict(torch.load(interface.path.replace(".pth", "_critic.pth")))
                print("Modèle A2C chargé.")
                interface.didtrainfct()
            except FileNotFoundError:
                print("Modèle A2C non trouvé, démarrage du programme...")
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
            mode="actor"
            lr=3e-4
        )

        critic = NeuralNetwork(
            input_dim=4,
            hidden_dim=64,
            output_dim=1,
            mode="critic"
            lr=1e-3
        )
        
        agent = PPOAgent(actor_nn=actor, critic_nn=critic, buffer_size=1024, entropy_bonus=False) # Already set to False by default (no need for cartpole)
        
        if interface.ask_load_ppo():
            try:
                agent.nna.load_state_dict(torch.load(interface.path.replace(".pth", "_actor.pth")))
                agent.nnc.load_state_dict(torch.load(interface.path.replace(".pth", "_critic.pth")))
                print("Modèle PPO chargé.")
                interface.didtrainfct()
            except FileNotFoundError:
                print("Modèle PPO non trouvé, démarrage du programme...")
                interface.didtrain = True
                interface.episodes = int(input("How many episodes would you like to train the model for? "))
        else:
            interface.didtrain = True
            interface.episodes = int(input("How many episodes would you like to train the model for? "))
    
    main()
