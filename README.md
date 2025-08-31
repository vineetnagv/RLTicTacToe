# Tic-Tac-Toe RL Agent

This project implements a Reinforcement Learning (RL) agent that learns to play and win at Tic-Tac-Toe. The agent is trained by playing against an opponent that makes random moves, and its learned policy is saved at various stages of the training process. The project also includes a graphical user interface (GUI) built with Tkinter, allowing a human player to play against the trained RL agent.

## How It Works

The project is centered around a Reinforcement Learning agent that learns to play Tic-Tac-Toe through self-play against a random opponent. The agent uses a Temporal Difference (TD) learning approach to update its strategy after each game.

### Game Logic and Representation

-   **Board Representation**: The Tic-Tac-Toe board is represented as a 3x3 list of lists (a matrix). The values in the matrix are:
    -   `0`: Empty cell
    -   `1`: Player 'X' (the RL agent)
    -   `-1`: Player 'O' (the opponent)
-   **Game Functions**:
    -   `initialize_board()`: Creates an empty 3x3 board.
    -   `make_move()`: Places a player's mark on the board.
    -   `check_win()`: Checks if a player has won the game.
    -   `is_board_full()`: Checks if the board is full (a draw).
    -   `is_match_over()`: Determines if the game has ended.
    -   `get_available_moves()`: Returns a list of empty cells where a move can be made.

### The Reinforcement Learning Agent

The `RLAgent` class implements the learning logic for the Tic-Tac-Toe agent.

-   **State Representation**: The agent perceives the game state as a tuple of tuples, which is an immutable version of the board. This allows the agent to use board states as keys in its `states_value` dictionary.

-   **Action Selection (Epsilon-Greedy Strategy)**:
    -   The agent uses an **epsilon-greedy** policy to balance **exploration** and **exploitation**.
    -   With a probability of **epsilon (ε)**, the agent chooses a random move from the available moves (exploration). This allows the agent to discover new strategies and avoid getting stuck in a suboptimal strategy.
    -   With a probability of **1-epsilon**, the agent chooses the move that leads to the state with the highest known value (exploitation). This leverages the agent's current knowledge to make the best move.
    -   The value of epsilon **decays** over time, so the agent explores more at the beginning of training and exploits more as it becomes more experienced. The decay is calculated to reach a minimum epsilon value by the end of the training episodes.

-   **Value Function**:
    -   The agent's knowledge is stored in the `states_value` dictionary. This dictionary maps each game state (represented as a tuple) to a numerical value.
    -   This value represents the agent's estimate of the **expected future reward** from that state. A higher value means the agent is more confident about winning from that state.

-   **Learning Algorithm (Temporal Difference Learning)**:
    -   After each game, the agent updates the values of the states it visited during the game.
    -   The `update_state_values` function implements the TD learning update rule. It iterates through the states visited in the game in reverse order and updates their values based on the final reward (1 for a win, -1 for a loss, 0 for a draw).
    -   The update rule is: `value += self.alpha * (self.gamma * reward - value)`
        -   `alpha`: The learning rate (0.3 in this implementation), which determines how much the agent updates its knowledge based on new information.
        -   `gamma`: The discount factor (0.9 in this implementation), which determines the importance of future rewards.
        -   `reward`: The reward received at the end of the game.

### Training Process

-   The `train` function orchestrates the training of the RL agent.
-   The agent plays a very large number of episodes (games) against an opponent that makes random moves.
-   After each episode, the agent's `states_value` dictionary is updated using the TD learning rule.
-   The agent's learned policy (the `states_value` dictionary) is saved at different percentages of the total training episodes (1%, 5%, 10%, 20%, 50%, 75%, and 100%). This allows for the analysis of the agent's performance at different stages of learning.

### Graphical User Interface (GUI)

-   The `TicTacToeGUI` class, built with `tkinter`, provides a graphical interface for a human to play against the trained RL agent.
-   The GUI displays the Tic-Tac-Toe board, and the human player can make moves by clicking on the buttons.
-   The agent's moves are determined by its learned policy (it will always choose the move that leads to the state with the highest value, as exploration is turned off during gameplay).

## How to Run

To run the project, you need to have Python installed. The only external library required is `tkinter`, which is usually included with Python.

1.  **Training the Agent**:
    To train the agent, simply run the `RLtictactoe.py` file from your terminal:
    ```sh
    python RLtictactoe.py
    ```
    This will start the training process. The agent will play a large number of games, and the learned policies will be saved as `.pkl` files in the same directory.

2.  **Playing Against the Agent**:
    After the training process is complete (or interrupted), the script will automatically load the latest policy and open the Tic-Tac-Toe GUI. You can then play against the trained agent. The agent plays as 'X', and the human player plays as 'O'.

## Requirements

This project uses only standard Python libraries. No external libraries are required to be installed. A `requirements.txt` file is included for completeness.

-   **Python 3.x**
-   **tkinter**: For the GUI. This is usually included with Python.

## File Structure

The project consists of the following files:

-   `RLtictactoe.py`: The main Python script containing the game logic, RL agent, and GUI.
-   `rl_agent_policy_at_*.pkl`: Saved policy files at different stages of training (e.g., `rl_agent_policy_at_1%.pkl`, `rl_agent_policy_at_5%.pkl`, etc.).
-   `README.md`: This file.

## Libraries Used

-   **`random`**: For making random moves for the opponent and for the agent's exploration.
-   **`pickle`**: For saving and loading the agent's learned policy.
-   **`time`**: For tracking the time elapsed during the game.
-   **`tkinter`**: For creating the graphical user interface.
