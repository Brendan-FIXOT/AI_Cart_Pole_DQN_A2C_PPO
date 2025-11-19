import os

class Interface:
    def __init__(self):
        self.didtrain = False
        self.episodes = None
        self.path_dqn = "model_saved/dqn/"
        self.path_a2c = "model_saved/a2c/"
        self.path_ppo = "model_saved/ppo/"
        self.grahic_name = None
        self.filename = None
        self.path = None

        # Création des dossiers si besoin
        os.makedirs(self.path_dqn, exist_ok=True)
        os.makedirs(self.path_a2c, exist_ok=True)

    def ask_mode(self):
        user_input = input("Choose the agent mode - DQN or A2C or PPO (d/a/p): ").lower()
        if user_input == 'd':
            return "dqn"
        elif user_input == 'a':
            return "a2c"
        elif user_input == 'p':
            return "ppo"
        else:
            print("Invalid input, defaulting to DQN.")
            return "dqn"

    def didtrainfct(self):
        user_input = input("Do you want to train the model? (y/n): ").lower()
        if user_input == 'y':
            self.didtrain = True
            self.episodes = int(input("How many episodes would you like to train the model for? "))

    def didtestfct(self):
        return input("Do you want to test the model? (y/n): ").lower() == 'y'

    def didgraphicfct(self):
        return input("Do you want to create a graphic of the agent's performance? (y/n): ").lower() == 'y'
    
    def ask_env(self):
        user_input = input("Choose the environment - CartPole or Breakout (c/b): ").lower()
        if user_input == 'b':
            self.grahic_name = "Breakout"
            return "Breakout"
        else:
            self.grahic_name = "CartPole"
            return "CartPole"

    # ---------------------------
    # DQN
    # ---------------------------
    def ask_save_dqn(self):
        user_input = input("Do you want to save the DQN model? (y/n): ").lower()
        if user_input == 'y':
            self.filename = input("Enter the filename to save the DQN model (without extension): ") + ".pth"
            self.path = os.path.join(self.path_dqn, self.filename)
            print(f"The DQN model will be saved at: {self.path}")
            return True
        return False

    def ask_load_dqn(self):
        user_input = input("Do you want to load an existing DQN model? (y/n): ").lower()
        if user_input == 'y':
            self.filename = input("Enter the filename of the DQN model to load (without extension): ") + ".pth"
            self.path = os.path.join(self.path_dqn, self.filename)
            print(f"The DQN model will be loaded from: {self.path}")
            return True
        return False

    # ---------------------------
    # A2C
    # ---------------------------
    def ask_save_a2c(self):
        user_input = input("Do you want to save the A2C model? (y/n): ").lower()
        if user_input == 'y':
            self.filename = input("Enter the filename to save the A2C model (without extension): ") + ".pth"
            self.path = os.path.join(self.path_a2c, self.filename)
            print(f"The A2C model will be saved at: {self.path}")
            return True
        return False

    def ask_load_a2c(self):
        user_input = input("Do you want to load an existing A2C model? (y/n): ").lower()
        if user_input == 'y':
            self.filename = input("Enter the filename of the A2C model to load (without extension): ") + ".pth"
            self.path = os.path.join(self.path_a2c, self.filename)
            print(f"The A2C model will be loaded from: {self.path}")
            return True
        return False

    # ---------------------------
    # PPO
    # ---------------------------
    def ask_save_ppo(self):
        user_input = input("Do you want to save the PPO model? (y/n): ").lower()
        if user_input == 'y':
            self.filename = input("Enter the filename to save the PPO model (without extension): ") + ".pth"
            self.path = os.path.join(self.path_ppo, self.filename)
            print(f"The PPO model will be saved at: {self.path}")
            return True
        return False
    
    def ask_load_ppo(self):
        user_input = input("Do you want to load an existing PPO model? (y/n): ").lower()
        if user_input == 'y':
            self.filename = input("Enter the filename of the PPO model to load (without extension): ") + ".pth"
            self.path = os.path.join(self.path_ppo, self.filename)
            print(f"The PPO model will be loaded from: {self.path}")
            return True
        return False