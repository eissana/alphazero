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


def encode_state_parallel(states):
    """
    b, m, n = states.shape
    output size: b x 3 x m x n
    """
    out = np.stack([states == 1, states == 0, states == -1]).astype(np.float32)
    out = np.swapaxes(out, 0, 1)
    return out


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
            nn.Conv2d(hidden_size, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.nrows * game.ncols, game.ncols),
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
        return 1 - 0.5 * (child.reward_sum / child.visit_count + 1)

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
        node = self
        while node is not None:
            node.reward_sum += reward
            node.visit_count += 1
            reward = self.game.opponent_reward(reward)
            node = node.parent


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

    def select(self, node):
        while node.is_fully_expanded():
            node = node.select()
        return node

    def policies_rewards(self, states):
        device = self.params["device"]

        enc_states = encode_state_parallel(states)
        enc_states = torch.tensor(enc_states, dtype=torch.float32, device=device)

        logits, rewards = self.model(enc_states)
        # mask out illegal moves
        logits[states[:, 0, :] != 0] = -np.inf
        policies = F.softmax(logits, dim=1).detach().cpu().numpy()

        return policies, rewards

    def dirichlet_policies(self, policies, states):
        batch_size, action_size = policies.shape
        eps, alpha = self.params["dirichlet_epsilon"], self.params["dirichlet_alpha"]
        dirichlet_noise = np.random.dirichlet([alpha] * action_size, size=batch_size)
        policies = (1 - eps) * policies + eps * dirichlet_noise
        # mask out illegal moves
        policies[states[:, 0, :] != 0] = 0
        policies /= np.sum(policies, axis=1, keepdims=True)
        return policies

    @torch.no_grad()
    def best_policies(self, states):
        states = np.array(states)
        policies, _ = self.policies_rewards(states)
        policies = self.dirichlet_policies(policies, states)

        roots = [Node(self.game, state, self.params, visit_count=1) for state in states]
        for root, policy in zip(roots, policies):
            root.expand(policy)

        num_simulations = self.params["num_simulations"]

        for _ in tqdm(range(num_simulations), desc="simulation"):
            batch_nodes = []
            for root in roots:
                node = self.select(root)

                is_over = self.game.is_over(node.state, node.parent_action)
                # check if the game is over.
                if is_over >= 0:
                    # 0 is for draw and 1 is for win
                    reward = is_over
                    reward = self.game.opponent_reward(reward)
                    node.backward(reward)
                else:
                    batch_nodes.append(node)

            if len(batch_nodes) == 0:
                continue

            batch_states = np.array([node.state for node in batch_nodes])
            policies, rewards = self.policies_rewards(batch_states)

            for node, policy, reward in zip(batch_nodes, policies, rewards):
                node.expand(policy)
                node.backward(reward.item())

        return self.compute_policies(roots)

    def compute_policies(self, roots):
        out = np.zeros((len(roots), self.game.ncols))

        for i, root in enumerate(roots):
            for child in root.children:
                out[i, child.parent_action] = child.visit_count

        out /= np.sum(out, axis=1, keepdims=True)
        return out


class ParallelPlay(object):
    def __init__(self, game):
        self.game = game
        self.current_state = game.init_state()
        self.states = []
        self.policies = []
        self.rewards = []
        self.players = []


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

    def generate_dataset(self, data=None):
        if data is None:
            states, policies, rewards = [], [], []
        else:
            states, policies, rewards = data

        num_self_plays = self.params["num_self_plays"]
        num_threads = self.params["num_threads"]

        for _ in tqdm(range(num_self_plays // num_threads), desc="self plays"):
            s, p, r = self.generate_data()
            states.extend(s)
            policies.extend(p)
            rewards.extend(r)

        return states, policies, rewards

    def generate_data(self):
        """
        Runs `num_threads` games in parallel and generates data from all the states
        in all the games.
        outout: (states, policies, rewards)
        """
        game = self.mcts.game
        player = game.first_player

        num_threads = self.params["num_threads"]
        par_plays = [ParallelPlay(game) for _ in range(num_threads)]

        # storing all states, policies, and rewards from all the threads.
        out_states, out_policies, out_rewards = [], [], []

        with tqdm(total=len(par_plays), desc="parallel plays") as pbar:
            while len(par_plays) > 0:
                states = np.array([p.current_state for p in par_plays])
                neutral_states = game.neutral_perspective(states, player)
                policies = self.mcts.best_policies(neutral_states)

                # loop in reverse order so that deletion of an element does not
                # mess up the indexes.
                for i in range(len(par_plays) - 1, -1, -1):
                    par_plays[i].states.append(neutral_states[i])
                    par_plays[i].policies.append(policies[i])
                    par_plays[i].players.append(player)

                    action = np.random.choice(len(policies[i]), p=policies[i])
                    par_plays[i].current_state = game.next_state(
                        par_plays[i].current_state, action, player
                    )

                    is_over = game.is_over(par_plays[i].current_state, action)
                    # check if the game is over.
                    if is_over >= 0:
                        out_states.extend(par_plays[i].states)
                        out_policies.extend(par_plays[i].policies)

                        # 0 is for draw and 1 is for win
                        reward = is_over
                        out_rewards.extend(
                            [
                                reward if p == player else game.opponent_reward(reward)
                                for p in par_plays[i].players
                            ]
                        )

                        del par_plays[i]
                        pbar.update(1)

                player = game.opponent(player)

        return out_states, out_policies, out_rewards

    def train_model(self, train_data, eval_data, losses=None):
        if losses is None:
            losses = Losses()

        states, policies, rewards = train_data
        states = encode_state_parallel(states)
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
        states = encode_state_parallel(states)

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

    def learn(self, losses):
        num_training_cycles = self.params["num_training_cycles"]
        model_num_offset = self.params["model_num_offset"]
        root_dir = self.params["root_dir"]

        for training_cycle in tqdm(range(num_training_cycles), desc="training cycles"):
            data = self.generate_dataset()
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


def tournment(num_plays, params):
    game = Connect4()

    untrained_mcts = MCTS(params)
    trained_mcts = MCTS(params)

    root_dir = params["root_dir"]
    model_checkpoint = torch.load(f"{root_dir}/models/model_{game}_latest.pt")
    trained_mcts.model.load_state_dict(model_checkpoint["model"])

    lr = params["lr"]
    weight_decay = params["weight_decay"]
    optimizer = torch.optim.Adam(
        trained_mcts.model.parameters(), lr=lr, weight_decay=weight_decay
    )
    optimizer.load_state_dict(model_checkpoint["optimizer"])

    result = {
        game.first_player: 0,
        game.opponent(game.first_player): 0,
        0: 0,
    }
    for play in tqdm(range(num_plays), desc="plays"):
        player = (
            game.first_player if play % 2 == 0 else game.opponent(game.first_player)
        )
        state = game.init_state()

        while True:
            neutral_state = game.neutral_perspective(state, player)

            if player == game.first_player:
                policy = untrained_mcts.best_policies([neutral_state])
            else:
                policy = trained_mcts.best_policies([neutral_state])

            action = np.argmax(policy[0])
            state = game.next_state(state, action, player)

            is_over = game.is_over(state, action)
            if is_over >= 0:
                if is_over > 0:
                    result[player] += 1
                else:
                    result[0] += 1
                break
            # switch player after each move.
            player = game.opponent(player)

    return result


def train_model(params):
    alpha0 = AlphaZero(params)

    game = alpha0.mcts.game
    player = game.first_player
    state = game.init_state()

    root_dir = params["root_dir"]

    if params["load_model"]:
        model_checkpoint = torch.load(f"{root_dir}/models/model_{game}_latest.pt")
        alpha0.mcts.model.load_state_dict(model_checkpoint["model"])
        alpha0.optimizer.load_state_dict(model_checkpoint["optimizer"])
        # losses = torch.load(f"{root_dir}/data/losses_{game}.pt")

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
            policy = untrained_mcts.best_policies([neutral_state])
        else:
            policy = alpha0.mcts.best_policies([neutral_state])

        policy = policy[0]

        plt.bar(range(len(policy)), policy)
        plt.show()

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


if __name__ == "__main__":
    params = {
        # CNN Model parameters
        "kernel_size": 3,
        "padding": 1,
        "embd_size": 3,
        "hidden_size": 128,
        "num_blocks": 9,
        "dropout": 0.0,
        # MCTS parameters
        "num_simulations": 100,
        "exploration_factor": 2,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3,
        # AlphaZero parameters
        "num_training_cycles": 1,
        "num_self_plays": 10,
        "num_threads": 10,
        "num_epochs": 100,
        "batch_size": 16,
        "train_ratio": 0.9,
        "model_num_offset": 0,
        # Training settings
        "load_model": True,
        "save_model": False,
        # Plotting paramaters
        "loss_eval_size": 10,
        # Optimizer parameters
        "lr": 0.2,
        "weight_decay": 0.1,
        # General parameters
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "root_dir": ".",
    }
    # train_model(params)

    result = tournment(20, params)
    print(result)
