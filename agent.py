# agent.py
import random


class SimpleReflexAgent:
    """A stateless condition-action agent for partial percepts."""

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here'):
            return 'Suck'
        if percept.get('wall_ahead'):
            return 'Left'
        return 'Up'


class ModelBasedAgent:
    """A condition-action agent that uses memory to avoid repeating loops."""

    def __init__(self):
        self.percept_history = []
        self.last_action = None
        self.wall_escape_index = 0
        self.relative_pos = (0, 0)
        self.visited_cells = {self.relative_pos}

    def sense_and_act(self, percept: dict) -> str:
        self._update_state(percept)

        if percept.get('food_here'):
            action = 'Suck'
        elif percept.get('wall_ahead'):
            escape_actions = ['Left', 'Right', 'Down', 'Up']
            action = escape_actions[self.wall_escape_index % len(escape_actions)]
            self.wall_escape_index += 1
        else:
            self.wall_escape_index = 0
            action = self._least_visited_move()

        self.last_action = action
        return action

    def _update_state(self, percept: dict):
        self.percept_history.append(dict(percept))

        if self.last_action in ('Up', 'Down', 'Left', 'Right') and not percept.get('wall_ahead'):
            x, y = self.relative_pos
            if self.last_action == 'Up':
                y += 1
            elif self.last_action == 'Down':
                y -= 1
            elif self.last_action == 'Left':
                x -= 1
            elif self.last_action == 'Right':
                x += 1

            self.relative_pos = (x, y)
            self.visited_cells.add(self.relative_pos)

    def _least_visited_move(self) -> str:
        x, y = self.relative_pos
        candidates = {
            'Up': (x, y + 1),
            'Right': (x + 1, y),
            'Down': (x, y - 1),
            'Left': (x - 1, y)
        }

        for action, position in candidates.items():
            if position not in self.visited_cells:
                return action
        return 'Up'


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)
