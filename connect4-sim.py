import numpy as np

players = {1: "B", -1: "R", 0: "."}

params = {
    "exploration_factor": 2**0.5,
    "num_iters": 1000,
}


def print_state(state):
    print("\n\t", end="")
    print("\n\t".join("  ".join([players[col] for col in row]) for row in state))
    print()


class Connect4(object):
    def __init__(self):
        # Grid shape.
        self.nrows = 6
        self.ncols = 7
        # number of connections to be made for winning.
        self.k = 4
        self.first_player = 1

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

    def no_move_left(self, state):
        return np.sum(state[0, :] == 0) == 0

    def is_over(self, state, action):
        return self.won(state, action) or self.no_move_left(state)

    def reward(self, final_state):
        """
        Call this when game is over. The given state must be a final state:
        The current player wins or no move to make.
        Returns 1 when the current player wins and 0 when it's draw
        (no moves available).
        """
        return 0 if self.no_move_left(final_state) else 1


class Node(object):
    def __init__(self, game, state, params, parent=None, parent_action=None):
        self.game = game
        self.state = state
        self.params = params
        self.parent = parent
        self.parent_action = parent_action

        self.children = []
        self.available_actions = (state[0, :] == 0).astype(np.int8)

        self.reward_sum = 0
        self.visit_count = 0

    def expected_reward(self, child):
        return -child.reward_sum / child.visit_count

    def ucb(self, child):
        """
        Computes the Upper Confidence Bound (UCB).
        """
        exploration_factor = self.params["exploration_factor"]
        exploration = (np.log(self.visit_count) / child.visit_count) ** 0.5
        return self.expected_reward(child) + exploration_factor * exploration

    def is_fully_expanded(self):
        return len(self.children) > 0 and (np.sum(self.available_actions) == 0)

    def select(self):
        return max(self.children, key=self.ucb)

    def expand(self):
        action = np.random.choice(np.where(self.available_actions == 1)[0])
        self.available_actions[action] = 0

        player = self.game.first_player
        other_player = self.game.opponent(player)

        child_state = self.game.next_state(self.state.copy(), action, player)
        child_state = self.game.neutral_perspective(child_state, other_player)

        child = Node(self.game, child_state, self.params, self, action)
        self.children.append(child)

        return child

    def simulate(self):
        state = self.state.copy()
        parent_action = self.parent_action
        player = self.game.first_player

        while not self.game.is_over(state, parent_action):
            parent_action = np.random.choice(np.where(state[0, :] == 0)[0])
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
        node = self
        while node is not None:
            node.reward_sum += reward
            node.visit_count += 1
            node = node.parent
            reward = self.game.opponent_reward(reward)


class MCTS(object):
    def __init__(self, game, params):
        self.game = game
        self.params = params

    def best_policy(self, state):
        root = Node(self.game, state, self.params)
        num_iters = self.params["num_iters"]

        for _ in range(num_iters):
            node = self.find_node(root)
            reward = node.simulate()
            node.backward(reward)

        return self.compute_policy(root)

    def find_node(self, root):
        node = root

        while not self.game.is_over(node.state, node.parent_action):
            if not node.is_fully_expanded():
                return node.expand()
            node = node.select()

        return node

    def compute_policy(self, node):
        out = np.zeros(self.game.ncols)
        for child in node.children:
            out[child.parent_action] = child.visit_count
        out /= np.sum(out)
        return out


if __name__ == "__main__":
    c4 = Connect4()
    player = c4.first_player
    state = c4.init_state()
    mcts = MCTS(c4, params)

    while True:
        print_state(state)
        print(f"player {players[player]} is to play...")

        available_actions = c4.available_actions(state)
        print(f"available actions: {c4.available_actions(state)}")

        if player == c4.first_player:
            action = int(input("provide a valid move: "))
        else:
            neutral_state = c4.neutral_perspective(state, player)
            policy = mcts.best_policy(neutral_state)
            action = np.argmax(policy)

        state = c4.next_state(state, action, player)

        if c4.is_over(state, action):
            print_state(state)
            if c4.no_move_left(state):
                print("draw!")
            else:
                print(f"player {players[player]} won!")
            break

        player = c4.opponent(player)
