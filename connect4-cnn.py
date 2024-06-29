import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

players = {1: "B", -1: "R", 0: "."}


def print_state(state):
    print("\n\t", end="")
    print("\n\t".join("  ".join([players[col] for col in row]) for row in state))
    print()


def encode_state(state):
    """
    m, n = state.shape
    output size: 3 x m x n
    """
    return np.stack([state == 1, state == 0, state == -1]).astype(np.float32)


def decode_state(enc_state):
    return 1 * enc_state[0] + 0 * enc_state[1] + (-1) * enc_state[2]


class Connect4(object):
    def __init__(self):
        # Grid shape.
        self.nrows = 6
        self.ncols = 7
        # number of connections to be made for winning.
        self.k = 4
        self.first_player = 1

    def __repr__(self):
        return self.__class__.__name__

    def init_state(self):
        return np.zeros((self.nrows, self.ncols), dtype=np.int8)

    def next_state(self, state, action, player):
        """
        Returns the new state resulted by applying `action` taken by `player`
        on the current `state`.
        """
        col = action
        row = np.argwhere(state[:, col] == 0)[-1][0]
        state[row, col] = player
        return state

    def available_actions(self, state):
        return np.where(state[0, :] == 0)[0]

    def opponent(self, player):
        return -player

    def opponent_reward(self, reward):
        return -reward

    def neutral_perspective(self, state, player):
        return player * state

    def won(self, state, action):
        """
        Returns true if `action` taken resulted in a winnng `state`.
        """
        if action is None:
            return False

        col = action
        # Find the first nonzero element in this column
        row = np.argwhere(state[:, col] != 0)[0][0]
        player = state[row, col]

        for i in range(self.nrows - self.k + 1):
            total = np.sum(state[i : i + self.k, col])
            if total == self.k * player:
                return True

        for j in range(self.ncols - self.k + 1):
            total = np.sum(state[row, j : j + self.k])
            if total == self.k * player:
                return True

        diag = np.diag(state, col - row)
        for j in range(len(diag) - self.k + 1):
            total = np.sum(diag[j : j + self.k])
            if total == self.k * player:
                return True

        flipped_col = self.ncols - col - 1
        diag = np.diag(np.fliplr(state), flipped_col - row)
        for j in range(len(diag) - self.k + 1):
            total = np.sum(diag[j : j + self.k])
            if total == self.k * player:
                return True

        return False

    def is_over(self, state, action):
        if self.won(state, action):
            return 1
        if np.sum(state[0, :] == 0) == 0:
            return 0
        return -1


class ResBlock(nn.Module):
    def __init__(self, hidden_size, params):
        super().__init__()

        kernel_size = params["kernel_size"]
        padding = params["padding"]
        dropout = params["dropout"]

        self.res_block = nn.Sequential(
            nn.Conv2d(
                hidden_size, hidden_size, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Conv2d(
                hidden_size, hidden_size, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + self.res_block(x)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class ResNet(nn.Module):
    def __init__(self, game, embd_size, hidden_size, num_blocks, params):
        super().__init__()

        kernel_size = params["kernel_size"]
        padding = params["padding"]
        dropout = params["dropout"]

        self.start_block = nn.Sequential(
            nn.Conv2d(embd_size, hidden_size, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.res_blocks = nn.ModuleList(
            [ResBlock(hidden_size, params) for _ in range(num_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_size, embd_size, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(embd_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(embd_size * game.nrows * game.ncols, game.ncols),
            nn.Dropout(dropout),
        )

        self.reward_head = nn.Sequential(
            nn.Conv2d(hidden_size, embd_size, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(embd_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(embd_size * game.nrows * game.ncols, 1),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.start_block(x)

        for res_block in self.res_blocks:
            out = res_block(out)

        logit = self.policy_head(out)
        reward = self.reward_head(out)

        return logit, reward


class Node(object):
    def __init__(
        self,
        game,
        state,
        params,
        parent=None,
        parent_action=None,
        prior_prob=0,
        visit_count=0,
    ):
        self.game = game
        self.state = state
        self.params = params
        self.parent = parent
        self.parent_action = parent_action

        self.children = []

        self.reward_sum = 0
        self.visit_count = visit_count
        self.prior_prob = prior_prob

    def expected_reward(self, child):
        if child.visit_count == 0:
            return 0
        return -child.reward_sum / child.visit_count

    def ucb(self, child):
        """
        Computes the Upper Confidence Bound (UCB).
        """
        exploration_factor = self.params["exploration_factor"]
        exploration = (self.visit_count) ** 0.5 / (1 + child.visit_count)
        return (
            self.expected_reward(child)
            + exploration_factor * child.prior_prob * exploration
        )

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        return max(self.children, key=self.ucb)

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob == 0:
                continue

            player = self.game.first_player
            other_player = self.game.opponent(player)

            child_state = self.game.next_state(self.state.copy(), action, player)
            child_state = self.game.neutral_perspective(child_state, other_player)

            child = Node(self.game, child_state, self.params, self, action, prob)
            self.children.append(child)

    def backward(self, reward):
        """
        Backpropagate reward and visit counts from the node to the root.
        """
        self.reward_sum += reward
        self.visit_count += 1
        # parent node is the opponent of the child node.
        reward = self.game.opponent_reward(reward)
        if self.parent is not None:
            self.parent.backward(reward)


class MCTS(object):
    def __init__(self, params):
        self.game = Connect4()

        embd_size = params["embd_size"]
        hidden_size = params["hidden_size"]
        num_blocks = params["num_blocks"]
        device = params["device"]
        self.model = ResNet(self.game, embd_size, hidden_size, num_blocks, params).to(
            device
        )

        self.params = params

    def policy_reward(self, state):
        enc_state = encode_state(state)
        enc_state = torch.tensor(enc_state, device=self.params["device"]).unsqueeze(0)

        logit, reward = self.model(enc_state)
        reward = reward.item()
        # mask out illegal moves
        logit[:, state[0, :] != 0] = -np.inf
        policy = F.softmax(logit, dim=1).squeeze(0).detach().cpu().numpy()

        return policy, reward

    def dirichlet_policy(self, policy, state):
        eps, alpha = self.params["dirichlet_epsilon"], self.params["dirichlet_alpha"]
        dirichlet_noise = np.random.dirichlet([alpha] * len(policy))
        policy = (1 - eps) * policy + eps * dirichlet_noise
        # mask out illegal moves
        policy[state[0, :] != 0] = 0
        policy /= np.sum(policy)
        return policy

    @torch.no_grad()
    def best_policy(self, state):
        policy, _ = self.policy_reward(state)
        policy = self.dirichlet_policy(policy, state)

        root = Node(self.game, state, self.params, visit_count=1)
        root.expand(policy)

        num_simulations = self.params["num_simulations"]

        for _ in tqdm(range(num_simulations), desc="simulation"):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            is_over = self.game.is_over(node.state, node.parent_action)
            # check if it's not over.
            if is_over < 0:
                policy, reward = self.policy_reward(node.state)
                node.expand(policy)
            else:
                # 0 for draw and 1 for win
                reward = is_over
                reward = self.game.opponent_reward(reward)

            node.backward(reward)

        return self.compute_policy(root)

    def compute_policy(self, root):
        out = np.zeros(self.game.ncols)

        for child in root.children:
            out[child.parent_action] = child.visit_count

        out /= np.sum(out)
        return out


class Loss(object):
    def __init__(self):
        self.train = []
        self.eval = []


class Losses(object):
    def __init__(self):
        self.policy = Loss()
        self.reward = Loss()


class AlphaZero(object):
    def __init__(self, params):
        self.mcts = MCTS(params)

        lr = params["lr"]
        weight_decay = params["weight_decay"]
        self.optimizer = torch.optim.Adam(
            self.mcts.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.params = params

    def generate_dataset(self, num_self_plays, data=None):
        if data is None:
            states, policies, rewards = [], [], []
        else:
            states, policies, rewards = data

        for _ in tqdm(range(num_self_plays), desc="self plays"):
            s, p, r = self.generate_data()
            states.extend(s)
            policies.extend(p)
            rewards.extend(r)

        return states, policies, rewards

    def generate_data(self):
        game = self.mcts.game
        player = game.first_player
        state = game.init_state()

        reward = 0
        states, policies, players = [], [], []

        while True:
            players.append(player)

            neutral_state = game.neutral_perspective(state, player)
            states.append(encode_state(neutral_state))

            policy = self.mcts.best_policy(neutral_state)
            policies.append(policy)

            action = np.random.choice(len(policy), p=policy)
            state = game.next_state(state, action, player)

            is_over = game.is_over(state, action)
            if is_over >= 0:
                reward = is_over
                break

            player = game.opponent(player)

        rewards = [
            reward if p == player else game.opponent_reward(reward) for p in players
        ]

        return states, policies, rewards

    def train_model(self, train_data, eval_data, losses=None):
        if losses is None:
            losses = Losses()

        states, policies, rewards = train_data
        num_epochs = self.params["num_epochs"]
        batch_size = self.params["batch_size"]
        device = self.params["device"]

        for _ in tqdm(range(num_epochs), desc="epochs"):
            self.mcts.model.train()
            batch_indices = np.random.choice(len(states), batch_size, replace=False)

            enc_states = torch.tensor(
                states[batch_indices], dtype=torch.float32, device=device
            )
            target_policies = torch.tensor(
                policies[batch_indices], dtype=torch.float32, device=device
            )
            target_rewards = torch.tensor(
                rewards[batch_indices], dtype=torch.float32, device=device
            ).unsqueeze(1)

            pred_logits, pred_rewards = self.mcts.model(enc_states)

            pred_policies = F.softmax(pred_logits, dim=1)
            policy_loss = F.cross_entropy(pred_policies, target_policies)
            reward_loss = F.mse_loss(pred_rewards, target_rewards)

            losses.policy.train.append(policy_loss.item())
            losses.reward.train.append(reward_loss.item())

            self.optimizer.zero_grad(set_to_none=True)
            policy_loss.backward(retain_graph=True)
            reward_loss.backward()
            self.optimizer.step()

            self.mcts.model.eval()
            policy_loss, reward_loss = self.eval_model(eval_data)
            losses.policy.eval.append(policy_loss.item())
            losses.reward.eval.append(reward_loss.item())

        return losses

    @torch.no_grad()
    def eval_model(self, eval_data):
        device = self.params["device"]

        states, policies, rewards = eval_data
        states, policies, rewards = (
            np.array(states),
            np.array(policies),
            np.array(rewards),
        )

        enc_states = torch.tensor(states, dtype=torch.float32, device=device)
        target_policies = torch.tensor(policies, dtype=torch.float32, device=device)
        target_rewards = torch.tensor(
            rewards, dtype=torch.float32, device=device
        ).unsqueeze(1)

        pred_logits, pred_rewards = self.mcts.model(enc_states)

        pred_policies = F.softmax(pred_logits, dim=1)
        policy_loss = F.cross_entropy(pred_policies, target_policies)
        reward_loss = F.mse_loss(pred_rewards, target_rewards)

        return policy_loss, reward_loss

    def split(self, data):
        states, policies, rewards = data
        states, policies, rewards = (
            np.array(states),
            np.array(policies),
            np.array(rewards),
        )
        num_records = len(states)
        train_ratio = self.params["train_ratio"]
        train_size = int(train_ratio * num_records)

        idx = np.random.choice(num_records, num_records, replace=False)
        train_idx, eval_idx = idx[:train_size], idx[train_size:]

        train_data = states[train_idx], policies[train_idx], rewards[train_idx]
        eval_data = states[eval_idx], policies[eval_idx], rewards[eval_idx]

        return train_data, eval_data

    def learn(self, losses=None):
        num_training_cycles = self.params["num_training_cycles"]
        model_num_offset = self.params["model_num_offset"]
        num_self_plays = self.params["num_self_plays"]
        root_dir = self.params["root_dir"]

        for training_cycle in tqdm(range(num_training_cycles), desc="training cycles"):
            data = self.generate_dataset(num_self_plays)
            train_data, eval_data = self.split(data)

            losses = self.train_model(train_data, eval_data, losses)

            model_checkpoint = {
                "model": self.mcts.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            model_num = training_cycle + model_num_offset
            model_location = f"{root_dir}/models/model_{self.mcts.game}_{model_num}.pt"

            torch.save(model_checkpoint, model_location)

        torch.save(
            model_checkpoint, f"{root_dir}/models/model_{self.mcts.game}_latest.pt"
        )

        losses_location = f"{root_dir}/data/losses_{self.mcts.game}.pt"
        torch.save(losses, losses_location)

        return losses


if __name__ == "__main__":
    params = {
        # CNN Model parameters
        "kernel_size": 3,
        "padding": 1,
        "embd_size": 3,
        "hidden_size": 128,
        "num_blocks": 8,
        "dropout": 0.1,
        # MCTS parameters
        "num_simulations": 800,
        "exploration_factor": 2,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3,
        # AlphaZero parameters
        "num_training_cycles": 1,
        "num_self_plays": 20,
        "num_epochs": 100,
        "batch_size": 32,
        "train_ratio": 0.9,
        "model_num_offset": 2,
        # Training settings
        "load_model": True,
        "save_model": True,
        # Plotting paramaters
        "loss_eval_size": 10,
        # Optimizer parameters
        "lr": 0.001,
        "weight_decay": 0.0001,
        # General parameters
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "root_dir": ".",
    }

    alpha0 = AlphaZero(params)

    game = alpha0.mcts.game
    player = game.first_player
    state = game.init_state()

    root_dir = params["root_dir"]

    if params["load_model"]:
        model_checkpoint = torch.load(f"{root_dir}/models/model_{game}_latest.pt")
        alpha0.mcts.model.load_state_dict(model_checkpoint["model"])
        alpha0.optimizer.load_state_dict(model_checkpoint["optimizer"])
    #     losses = torch.load(f"{root_dir}/data/losses_{game}.pt")
    # else:
    losses = None

    if params["save_model"]:
        losses = alpha0.learn(losses)

        fig, (ax1, ax2) = plt.subplots(2)
        loss_eval_size = params["loss_eval_size"]

        ax1.plot(
            torch.tensor(losses.policy.train).view(-1, loss_eval_size).mean(axis=1)
        )
        ax1.plot(torch.tensor(losses.policy.eval).view(-1, loss_eval_size).mean(axis=1))
        ax1.legend(["training policy loss", "evaluation policy loss"])
        ax1.grid()

        ax2.plot(
            torch.tensor(losses.reward.train).view(-1, loss_eval_size).mean(axis=1)
        )
        ax2.plot(torch.tensor(losses.reward.eval).view(-1, loss_eval_size).mean(axis=1))
        ax2.legend(["training reward loss", "evaluation reward loss"])
        ax2.grid()

        plt.savefig(f"{root_dir}/images/loss.png")
        plt.show()

    untrained_mcts = MCTS(params)

    while True:
        print_state(state)

        available_actions = game.available_actions(state)
        print(f"available actions: {game.available_actions(state)}")

        neutral_state = game.neutral_perspective(state, player)

        if player == game.first_player:
            policy = untrained_mcts.best_policy(neutral_state)
        else:
            policy = alpha0.mcts.best_policy(neutral_state)

        action = np.argmax(policy)
        print(f"player {players[player]} move: {action}")

        state = game.next_state(state, action, player)

        is_over = game.is_over(state, action)
        if is_over >= 0:
            print_state(state)
            if is_over == 0:
                print("draw!")
            else:
                print(f"player {players[player]} won!")
            break

        player = game.opponent(player)
