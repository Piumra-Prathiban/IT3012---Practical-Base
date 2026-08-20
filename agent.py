# agent.py
from collections import deque
import heapq
import math
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


class SearchAgent:
    """An offline planning agent with uninformed graph search strategies."""

    ACTIONS = (
        ('Up', (0, 1)),
        ('Right', (1, 0)),
        ('Down', (0, -1)),
        ('Left', (-1, 0))
    )

    def __init__(self):
        self.plan = []
        self.active_algo = 'AStar' #choose from BFS, DFS, UCS, AStar

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here'):
            self.plan = []
            return 'Suck'

        if not self.plan:
            start_pos = tuple(percept['agent_pos'])
            food_positions = [tuple(food) for food in percept.get('all_food', [])]
            walls = percept.get('walls', [])
            grid_size = percept['grid_size']
            self.plan = self._plan_to_closest_food(start_pos, food_positions, walls, grid_size)

        if self.plan:
            return self.plan.pop(0)
        return 'Up'

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Return the shortest unweighted path from start to goal using BFS."""
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        wall_set = set(walls)
        frontier = deque([(start, [])])
        reached = {start}

        while frontier:
            state, path = frontier.popleft()
            if state == goal:
                return path

            for action, next_state in self._successors(state, wall_set, grid_size):
                if next_state not in reached:
                    reached.add(next_state)
                    frontier.append((next_state, path + [action]))

        return None

    def dfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Return a path from start to goal using DFS."""
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        wall_set = set(walls)
        frontier = [(start, [])]
        reached = {start}

        while frontier:
            state, path = frontier.pop()
            if state == goal:
                return path

            for action, next_state in self._successors(state, wall_set, grid_size):
                if next_state not in reached:
                    reached.add(next_state)
                    frontier.append((next_state, path + [action]))

        return None

    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        """Return the lowest-cost path from start to goal using UCS."""
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        wall_set = set(walls)
        frontier = [(0, start, [])]
        reached = {start: 0}

        while frontier:
            cost, state, path = heapq.heappop(frontier)
            if state == goal:
                return path

            if cost > reached[state]:
                continue

            for action, next_state in self._successors(state, wall_set, grid_size):
                new_cost = cost + 1
                if next_state not in reached or new_cost < reached[next_state]:
                    reached[next_state] = new_cost
                    heapq.heappush(frontier, (new_cost, next_state, path + [action]))

        return None

    def manhattan_distance(self, pos, goal):
        x1, y1 = pos
        x2, y2 = goal
        return abs(x1 - x2) + abs(y1 - y2)

    def euclidean_distance(self, pos, goal):
        x1, y1 = pos
        x2, y2 = goal
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        """Return the lowest estimated-cost path from start to goal using A*."""
        start = tuple(start_pos)
        goal = tuple(goal_pos)
        wall_set = set(walls)
        reached_states = set()

        h_cost = self._heuristic(start, goal, heuristic_type)
        frontier = [(h_cost, 0, start, [])]

        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(frontier)

            if current_pos == goal:
                return path_taken

            if current_pos in reached_states:
                continue

            reached_states.add(current_pos)

            for action, next_pos in self._successors(current_pos, wall_set, grid_size):
                if next_pos not in reached_states:
                    new_g_cost = g_cost + 1
                    new_h_cost = self._heuristic(next_pos, goal, heuristic_type)
                    new_f_cost = new_g_cost + new_h_cost
                    heapq.heappush(frontier, (new_f_cost, new_g_cost, next_pos, path_taken + [action]))

        return None

    def _heuristic(self, pos, goal, heuristic_type):
        if heuristic_type == 'euclidean':
            return self.euclidean_distance(pos, goal)
        return self.manhattan_distance(pos, goal)

    def _successors(self, state, walls, grid_size):
        width, height = grid_size
        x, y = state

        for action, (dx, dy) in self.ACTIONS:
            next_state = (x + dx, y + dy)
            if self._is_valid_position(next_state, walls, width, height):
                yield action, next_state

    def _is_valid_position(self, position, walls, width, height):
        x, y = position
        return 0 <= x < width and 0 <= y < height and position not in walls

    def _plan_to_closest_food(self, start_pos, food_positions, walls, grid_size):
        best_path = None

        for food_pos in food_positions:
            path = self._run_active_search(start_pos, food_pos, walls, grid_size)
            if path is not None and (best_path is None or len(path) < len(best_path)):
                best_path = path

        return best_path or []

    def _run_active_search(self, start_pos, goal_pos, walls, grid_size):
        active_algo = self.active_algo.upper()

        if active_algo == 'DFS':
            return self.dfs_search(start_pos, goal_pos, walls, grid_size)
        if active_algo == 'UCS':
            return self.ucs_search(start_pos, goal_pos, walls, grid_size)
        if active_algo == 'ASTAR':
            return self.astar_search(start_pos, goal_pos, walls, grid_size)
        return self.bfs_search(start_pos, goal_pos, walls, grid_size)


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


if __name__ == "__main__":
    agent = SearchAgent()
    print("Manhattan:", agent.manhattan_distance((0, 0), (3, 4)))
    print("Euclidean:", agent.euclidean_distance((0, 0), (3, 4)))
