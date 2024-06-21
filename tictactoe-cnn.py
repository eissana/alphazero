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
    
    def no_move_left(self, state):
        return (np.sum(state == 0) == 0)

    def is_over(self, state, action):
        '''
        The game is over when the player wins or there is no mover left.
        '''
        return self.won(state, action) or self.no_move_left(state)

    def reward(self, state):
        """
        Call this when game is over. The given state must be a final state:
        The current player wins or no move to make.
        Returns 1 when the current player wins and 0 when it's draw
        (no moves available).
        """
        return 0 if self.no_move_left(state) else 1


class Node(object):
    def __init__(self, game, state, params, parent=None, parent_action=None):
        """
        `parent` took `parent_action` which resulted in the current `state`.
        """
        self.game = game
        self.state = state
        self.params = params
        self.parent = parent
        self.parent_action = parent_action

        self.children = []
        self.available_actions = (state.reshape(-1) == 0).astype(np.int8)

        # reward reward for the current state which can be positive or negative.
        self.reward_sum = 0
        # number of visits at this node.
        self.visit_count = 0

    def is_fully_expanded(self):
        """
        Used in the node selection. Starting from the root, we select a child node
        which is fully expanded. We continue this process until we reach a node that
        is not fully expanded. Then, we expand the node.
        """
        return (np.sum(self.available_actions) == 0) and len(self.children) > 0

    def expected_reward(self, child):
        return -child.reward_sum / child.visit_count

    def ucb(self, child):
        """
        Computes the Upper Confidence Bound (UCB).
        """
        exploratin_factor = self.params["exploratin_factor"]
        exploration = (np.log(self.visit_count) / child.visit_count) ** 0.5
        return self.expected_reward(child) + exploratin_factor * exploration

    def select(self):
        return max(self.children, key=self.ucb)

    def expand(self):
        action = np.random.choice(np.where(self.available_actions == 1)[0])
        # mark as explored so it won't be available anymore.
        self.available_actions[action] = 0

        player = self.game.first_player
        other_player = self.game.opponent(player)
        # we assume this player is always the player 1, without loss of generality.
        child_state = self.game.next_state(self.state.copy(), action, player=player)
        child_state = self.game.neutral_perspective(child_state, other_player)
        child_node = Node(self.game, child_state, self.params, self, action)

        self.children.append(child_node)
        return child_node

    def simulate(self):
        state = self.state.copy()
        parent_action = self.parent_action
        # We can assume the player at this node is the player 1, without loss
        # of generality. If this player wins we return 1, if opponent wins we return
        # -1, and if draw, we return 0.
        player = self.game.first_player

        while not self.game.is_over(state, parent_action):
            parent_action = np.random.choice(np.where(state.reshape(-1) == 0)[0])
            state = self.game.next_state(state, parent_action, player)
            player = self.game.opponent(player)

        reward = self.game.reward(state)
        # `player` took `parent_action` which resulted in winning `state` so the winner is the
        # other player and we need to rever reward.
        reward = self.game.opponent_reward(reward)
        return (
            reward
            if player == self.game.first_player
            else self.game.opponent_reward(reward)
        )

    def backward(self, reward):
        """
        Backpropagate reward and visit counts from the node to the root.
        """
        node = self
        while node is not None:
            node.reward_sum += reward
            node.visit_count += 1
            # parent node is the opponent of the child node.
            reward = self.game.opponent_reward(reward)
            node = node.parent


class MCTS(object):
    """
    Monte Carlo Tree Search implementation. It has no notion of players. It sees the state
    from the perspective of alternating players in each iteration. This is useful when two
    machines play as we can just change the perspective.
    """

    def __init__(self, game, params):
        self.game = game
        self.params = params

    def best_policy(self, state, parent_action):
        num_iters = self.params["num_iters"]
        root = Node(self.game, state, self.params, parent_action=parent_action)

        for _ in range(num_iters):
            # starting from the root, select a child with the highest UCB if
            # it's fully expanded, or expand it.
            node = self.find_node(root)
            reward = node.simulate()
            node.backward(reward)

        return self.compute_policy(root)

    def find_node(self, root):
        """
        Starting from the root node, selects a node that is fully expanded and has highest UCB
        until:
            * The game is over. It return the last node.
            * It finds a node that is not fully expanded. It expands the node and returns it.
        """
        node = root
        while not self.game.is_over(node.state, node.parent_action):
            if not node.is_fully_expanded():
                return node.expand()
            node = node.select()
        return node

    def compute_policy(self, root):
        """
        Returns the probability distribution of the visit counts.
        """
        out = np.zeros(self.game.board_size**2)
        for child in root.children:
            out[child.parent_action] = child.visit_count
        out /= np.sum(out)
        return out


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

        if player == t3.first_player:
            action = int(input("enter your action: "))
        else:
            # For complicated game we use CNN with MCTS but here only CNN works.                        
            neutral_state = t3.neutral_perspective(state, player)
            enc_state = encode_state(neutral_state)
            enc_state = torch.tensor(enc_state, dtype=torch.float32).unsqueeze(0)
            logit, reward = model(enc_state)
            
            policy = F.softmax(logit, dim=1).squeeze(0).detach().cpu().numpy()
            action = np.argmax(policy)

        state = t3.next_state(state, action, player)

        if t3.is_over(state, action):
            print_state(state)
            if t3.no_move_left(state):
                print("draw!")
            else:
                print(f"player {players[player]} won!")
            break
        
        player = t3.opponent(player)
