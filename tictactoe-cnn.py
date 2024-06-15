import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

players = {1: "x", -1: "o", 0: "."}

params = {
    "exploratin_factor": 2,
    "num_iters": 1000,
    "kernel_size": 3,
    "padding": 1,
    "embd_size": 3,
    "hidden_size": 64,
    "num_blocks": 4,
    "num_epochs": 1000,
    "batch_size": 64,
    "num_self_plays": 500,
    "load_data": True,
    "load_model": True,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")    
}

def print_state(state):
    print("\n\t", end="")
    print("\n\t".join("  ".join([players[col] for col in row]) for row in state))
    print()

def encode_state(state):
    # m, n = state.shape
    # the output shape is 3 x m x n
    return np.stack(
        [state == 1, state == 0, state == -1]
    ).astype(np.float32)

class ResBlock(nn.Module):
    def __init__(self, hidden_size, params):
        super().__init__()
        kernel_size = params["kernel_size"]
        padding = params["padding"]

        self. res_block = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_size)
        )
        
    def forward(self, x):
        out = x + self.res_block(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, board_size, embd_size, hidden_size, num_blocks, params):
        super().__init__()
        kernel_size = params["kernel_size"]
        padding = params["padding"]

        self.start_block = nn.Sequential(
            nn.Conv2d(embd_size, hidden_size, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU()
        )
        
        self.res_blocks = nn.ModuleList(
            [ResBlock(hidden_size, params) for _ in range(num_blocks)]
        )
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_size, embd_size, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(embd_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(embd_size * board_size**2, board_size**2)
        )
        
        self.reward_head = nn.Sequential(
            nn.Conv2d(hidden_size, embd_size, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(embd_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(embd_size * board_size**2, 1),
            nn.Tanh()            
        )
        
    def forward(self, x):
        '''
        Input x is a 4D tensor of size (batch_size, embd_size, height, width).
        '''
        out = self.start_block(x)
        
        for res_block in self.res_blocks:
            out = res_block(out)
            
        policy_logit = self.policy_head(out)
        reward = self.reward_head(out)
        
        return policy_logit, reward


class TicTacToe(object):
    def __init__(self):
        self.board_size = 3
        self.first_player = 1

    def __repr__(self):
        return self.__class__.__name__

    def init_state(self):
        return np.zeros((self.board_size, self.board_size), dtype=np.int8)

    def next_state(self, state, action, player):
        row, col = self.coord(action)
        state[row, col] = player
        return state

    def coord(self, action):
        return action//self.board_size, action%self.board_size
    
    def won(self, state, action):
        if action is None:
            return False
        row, col = self.coord(action)
        player = state[row, col]

        win_sum = player * self.board_size
        row_win = np.sum(state[row, :]) == win_sum
        col_win = np.sum(state[:, col]) == win_sum
        diag_win = np.sum(np.diag(state)) == win_sum
        offdiag_win = np.sum(np.diag(np.fliplr(state))) == win_sum
    
        return row_win or col_win or diag_win or offdiag_win
    
    def opponent(self, player):
        return -player

    def opponent_reward(self, reward):
        return -reward
    
    def neutral_perspective(self, state, player):
        '''
        Perspective of the first (default) player.
        For player 1, the state is unchaged while player -1's perspective
        of the state is reversed so as if player 1 sees the state.
        '''
        return player * state 
    
    def available_actions(self, state):
        return np.where(state.reshape(-1) == 0)[0]

    def is_over(self, state, action):
        '''
        The game is over when the player wins or there is no mover left.
        '''
        return self.won(state, action) or (np.sum(state == 0) == 0)

    def reward(self, won):
        return 1 if won else 0

    
if __name__ == "__main__":
    t3 = TicTacToe()
    state = t3.init_state()
    player = t3.first_player
    reward = 0

    model = ResNet(
        t3.board_size, 
        embd_size=params["embd_size"], 
        hidden_size=params["hidden_size"], 
        num_blocks=params["num_blocks"],
        params=params).to(device=params["device"])

    check_point = torch.load(f"./models/{t3}_{params["num_epochs"]}.pt")
    model.load_state_dict(check_point["model"])

    model.eval()

    while True:
        print(f"player '{players[player]}' to play...")
        print_state(state)

        available_actions = t3.available_actions(state)
        print(f"available actions: {available_actions}")

        if len(available_actions) == 0:
            print("draw!")
            break        

        if player == t3.first_player:
            action = int(input("enter your action: "))
        else:                        
            neutral_state = t3.neutral_perspective(state, player)
            enc_state = encode_state(neutral_state)
            enc_state = torch.tensor(enc_state, dtype=torch.float32).unsqueeze(0)
            logit, reward = model(enc_state)
            
            policy = F.softmax(logit, dim=1).squeeze(0).detach().cpu().numpy()
            action = np.argmax(policy)

        state = t3.next_state(state, action, player)

        if t3.won(state, action):
            print_state(state)
            print(f"player {players[player]} won!")
            break
        
        player = t3.opponent(player)
