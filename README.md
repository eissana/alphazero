# AlphaZero

In this repo we implement algorithms for learning how play certain games like TicTacToe, Connect4, Chess, Go, etc. We use reinforcement learning (RL), Monte Carlo tree search (MCTS), simulation, and neural networks.

1. Create an environment: `python -m venv venv`
2. Activate the environment: `source venv/bin/activate` (to deactivate just run `deactivate`)
3. Install requirements: `pip install -r requirements.txt`
4. Run simulation-based TicTacToe model: `python -m tictactoe-sim`
5. Run CNN-based TicTacToe model: `python -m tictactoe-cnn`
6. Run simulation-based Connect4 model: `python -m connect4-sim`
7. Run CNN-based Connect4 model: `python -m connect4-cnn`

For the CNN-based models:

* Set `load_data=False` and `save_data=True` to generate new data from scratch and save them.
* Set `load_data=True` and `save_data=True` to append more records to the saved data.
* Set `load_data=True` and `save_data=False` to load and use the saved data for training.
* Set `load_model=False` and `save_model=True` to train a model from scratch and save it.
* Set `load_model=True` and `save_model=True` to load and retrain a model and save it.
* Set `load_model=True` and `save_model=False` to load and use the saved model for inference (play a game).
