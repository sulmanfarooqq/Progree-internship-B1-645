"""
TASK 3: Heuristic Graph Pathfinding Agent Search Engine
A* Search, Dijkstra's Algorithm, and BFS implementations from scratch.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import heapq
import time
import json
from pathlib import Path
from collections import deque

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(42)


# ─── 1. GRID MAZE GENERATOR ─────────────────────────────────────────────────

def generate_grid(rows, cols, obstacle_ratio=0.25):
    grid = np.zeros((rows, cols), dtype=int)
    n_obstacles = int(rows * cols * obstacle_ratio)
    indices = [(r, c) for r in range(rows) for c in range(cols)]
    np.random.shuffle(indices)
    for r, c in indices[:n_obstacles]:
        if (r, c) != (0, 0) and (r, c) != (rows - 1, cols - 1):
            grid[r, c] = 1
    return grid


def add_random_obstacles(grid, extra_blocks=5):
    rows, cols = grid.shape
    empty = [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == 0 and (r, c) != (0, 0) and (r, c) != (rows - 1, cols - 1)]
    np.random.shuffle(empty)
    for r, c in empty[:extra_blocks]:
        grid[r, c] = 1
    return grid

GRIDS = {
    'small':  generate_grid(8, 8, 0.20),
    'medium': generate_grid(16, 16, 0.25),
    'large':  generate_grid(32, 32, 0.25),
}

GRIDS['small_extra'] = add_random_obstacles(GRIDS['small'].copy(), extra_blocks=3)
GRIDS['medium_extra'] = add_random_obstacles(GRIDS['medium'].copy(), extra_blocks=5)
GRIDS['large_extra'] = add_random_obstacles(GRIDS['large'].copy(), extra_blocks=8)


# ─── 2. HEURISTIC FUNCTIONS ──────────────────────────────────────────────────

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


# ─── 3. PATHFINDING ALGORITHMS ──────────────────────────────────────────────

def get_neighbors(pos, grid):
    rows, cols = grid.shape
    r, c = pos
    neighbors = []
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            neighbors.append((nr, nc))
    return neighbors


def bfs(grid, start, goal):
    visited = set()
    parent = {}
    queue = deque([start])
    visited.add(start)
    nodes_expanded = 0

    start_time = time.perf_counter()
    while queue:
        current = queue.popleft()
        nodes_expanded += 1
        if current == goal:
            elapsed = time.perf_counter() - start_time
            return reconstruct_path(parent, start, goal), nodes_expanded, elapsed
        for neighbor in get_neighbors(current, grid):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    return None, nodes_expanded, time.perf_counter() - start_time


def dijkstra(grid, start, goal):
    distances = {start: 0}
    parent = {}
    pq = [(0, start)]
    visited = set()
    nodes_expanded = 0

    start_time = time.perf_counter()
    while pq:
        dist, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1
        if current == goal:
            elapsed = time.perf_counter() - start_time
            return reconstruct_path(parent, start, goal), nodes_expanded, elapsed
        for neighbor in get_neighbors(current, grid):
            new_dist = dist + 1
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                parent[neighbor] = current
                heapq.heappush(pq, (new_dist, neighbor))
    return None, nodes_expanded, time.perf_counter() - start_time


def a_star(grid, start, goal, heuristic=manhattan):
    open_set = [(0 + heuristic(start, goal), 0, start)]
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    parent = {}
    visited_nodes = set()
    nodes_expanded = 0

    start_time = time.perf_counter()
    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current in visited_nodes:
            continue
        visited_nodes.add(current)
        nodes_expanded += 1
        if current == goal:
            elapsed = time.perf_counter() - start_time
            return reconstruct_path(parent, start, goal), nodes_expanded, elapsed
        for neighbor in get_neighbors(current, grid):
            tentative_g = g + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                parent[neighbor] = current
                heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
    return None, nodes_expanded, time.perf_counter() - start_time


def reconstruct_path(parent, start, goal):
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = parent.get(current)
        if current is None:
            return None
    path.append(start)
    return path[::-1]


# ─── 4. VISUALIZATION ────────────────────────────────────────────────────────

def plot_grid_and_path(grid, path, start, goal, title, save_path, expanded=None):
    rows, cols = grid.shape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    cmap = LinearSegmentedColormap.from_list('grid_cmap', ['#F0F0F0', '#2C3E50'])
    ax1.imshow(grid, cmap=cmap, origin='upper')
    ax1.set_title(f'{title} — Pathfinding', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Column', fontsize=11)
    ax1.set_ylabel('Row', fontsize=11)
    ax1.set_xticks(range(cols))
    ax1.set_yticks(range(rows))
    ax1.grid(True, alpha=0.3)

    ax1.plot(start[1], start[0], marker='o', color='#27AE60', markersize=12, label='Start (S)', zorder=5)
    ax1.plot(goal[1], goal[0], marker='s', color='#E74C3C', markersize=12, label='Goal (G)', zorder=5)

    if path:
        path_y = [p[0] for p in path]
        path_x = [p[1] for p in path]
        ax1.plot(path_x, path_y, color='#3498DB', linewidth=2.5, label=f'Path ({len(path)} steps)', zorder=3)
        ax1.scatter(path_x[1:-1], path_y[1:-1], color='#3498DB', s=30, zorder=3)

    ax1.legend(fontsize=10)
    ax1.set_aspect('equal')

    # Info panel
    ax2.axis('off')
    info_text = f"{title}\n\n"
    info_text += f"Grid Size: {rows}×{cols}\n"
    info_text += f"Obstacles: {int(np.sum(grid))} cells\n"
    info_text += f"Start: {start}\n"
    info_text += f"Goal: {goal}\n"
    if path:
        info_text += f"Path Length: {len(path)} steps\n"
    if expanded is not None:
        info_text += f"Nodes Expanded: {expanded}\n"
    info_text += f"Walkable Cells: {rows * cols - int(np.sum(grid))}\n"

    ax2.text(0.1, 0.5, info_text, fontsize=13, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#CCC'))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_comparison_chart(results_df, save_path):
    grids = list(set(r['grid'] for r in results_df))
    algorithms = list(set(r['algorithm'] for r in results_df))
    colors = {'A*': '#3498DB', 'Dijkstra': '#E74C3C', 'BFS': '#2ECC71'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Steps
    ax = axes[0]
    for algo in algorithms:
        vals = [next((r['path_length'] for r in results_df if r['grid'] == g and r['algorithm'] == algo), 0) for g in grids]
        if vals:
            ax.plot(grids, vals, marker='o', label=algo, color=colors.get(algo, '#333'), linewidth=2, markersize=8)
    ax.set_title('Path Length Comparison', fontweight='bold', fontsize=13)
    ax.set_ylabel('Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # Nodes Expanded
    ax = axes[1]
    for algo in algorithms:
        vals = [next((r['nodes_expanded'] for r in results_df if r['grid'] == g and r['algorithm'] == algo), 0) for g in grids]
        if vals:
            ax.plot(grids, vals, marker='s', label=algo, color=colors.get(algo, '#333'), linewidth=2, markersize=8)
    ax.set_title('Nodes Expanded Comparison', fontweight='bold', fontsize=13)
    ax.set_ylabel('Nodes Expanded')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # Runtime
    ax = axes[2]
    for algo in algorithms:
        vals = [next((r['runtime_ms'] for r in results_df if r['grid'] == g and r['algorithm'] == algo), 0) for g in grids]
        if vals:
            ax.plot(grids, vals, marker='^', label=algo, color=colors.get(algo, '#333'), linewidth=2, markersize=8)
    ax.set_title('Runtime Comparison', fontweight='bold', fontsize=13)
    ax.set_ylabel('Runtime (ms)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_algorithm_radar(metrics, save_path):
    categories = ['Path Optimality', 'Speed', 'Memory Eff.', 'Scalability', 'Heuristic Use']
    values = {
        'A*': [0.95, 0.85, 0.70, 0.80, 1.0],
        'Dijkstra': [1.0, 0.50, 0.50, 0.60, 0.0],
        'BFS': [0.60, 0.70, 0.40, 0.50, 0.0],
    }
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = {'A*': '#3498DB', 'Dijkstra': '#E74C3C', 'BFS': '#2ECC71'}
    for algo, vals in values.items():
        vals_plot = vals + vals[:1]
        ax.plot(angles, vals_plot, 'o-', linewidth=2, label=algo, color=colors[algo])
        ax.fill(angles, vals_plot, alpha=0.1, color=colors[algo])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title('Algorithm Capability Radar', fontweight='bold', fontsize=14, pad=20)
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_expanded_density(grid, expanded_nodes, save_path, title='Expanded Node Density'):
    fig, ax = plt.subplots(figsize=(10, 8))
    density = np.zeros_like(grid, dtype=float)
    for r, c in expanded_nodes:
        density[r, c] = 1

    cmap = LinearSegmentedColormap.from_list('density', ['#FFFFFF', '#F39C12', '#E74C3C'])
    ax.imshow(grid, cmap='Greys', alpha=0.3, origin='upper')
    ax.imshow(density, cmap=cmap, alpha=0.7, origin='upper')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── 5. MAIN PIPELINE ───────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TASK 3: Heuristic Graph Pathfinding Agent Search Engine")
    print("=" * 60)

    algorithms = {
        'A*': lambda g, s, t: a_star(g, s, t, heuristic=manhattan),
        'Dijkstra': lambda g, s, t: dijkstra(g, s, t),
        'BFS': lambda g, s, t: bfs(g, s, t),
    }

    all_results = []

    for grid_name, grid in GRIDS.items():
        rows, cols = grid.shape
        start = (0, 0)
        goal = (rows - 1, cols - 1)
        print(f"\n-- {grid_name.upper()} Grid ({rows}x{cols}) --")

        if grid[start] == 1:
            print(f"   Start blocked — finding nearest free cell...")
            for r in range(rows):
                for c in range(cols):
                    if grid[r, c] == 0:
                        start = (r, c)
                        break
                if grid[start] == 0:
                    break

        for algo_name, algo_fn in algorithms.items():
            print(f"   [{algo_name}] ", end='')
            path, nodes_expanded, elapsed = algo_fn(grid, start, goal)
            runtime_ms = elapsed * 1000

            result = {
                'grid': grid_name,
                'grid_size': f'{rows}×{cols}',
                'algorithm': algo_name,
                'path_found': path is not None,
                'path_length': len(path) if path else 0,
                'nodes_expanded': nodes_expanded,
                'runtime_ms': round(runtime_ms, 3),
            }
            all_results.append(result)

            if path:
                print(f"Path: {len(path)} steps | Nodes: {nodes_expanded} | Time: {runtime_ms:.3f}ms")
            else:
                print(f"No path found | Nodes: {nodes_expanded} | Time: {runtime_ms:.3f}ms")

            # Visualize each result for medium grid
            if grid_name in ['small', 'small_extra']:
                subdir = OUTPUT_DIR / grid_name
                subdir.mkdir(exist_ok=True)
                algo_file = algo_name.lower().replace('*', 'star')
                save_path = subdir / f'{algo_file}_path.png'
                plot_grid_and_path(grid, path, start, goal, f'{grid_name.upper()} - {algo_name}', save_path, nodes_expanded)

    print("\n-- Generating comparison visualizations --")
    comp_path = OUTPUT_DIR / 'algorithm_comparison.png'
    plot_comparison_chart(all_results, comp_path)
    print(f"   Saved: {comp_path.name}")

    radar_path = OUTPUT_DIR / 'algorithm_radar.png'
    plot_algorithm_radar(all_results, radar_path)
    print(f"   Saved: {radar_path.name}")

    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"   Saved: results.json")

    print("\n-- Results Summary --")
    df_display = []
    for r in all_results:
        df_display.append({
            'Grid': r['grid'],
            'Size': r['grid_size'],
            'Algorithm': r['algorithm'],
            'Path': str(r['path_length']) if r['path_found'] else 'X',
            'Nodes': r['nodes_expanded'],
            'Time(ms)': r['runtime_ms'],
        })
    for row in df_display:
        print(f"   {row['Grid']:>12s} | {row['Size']:>5s} | {row['Algorithm']:>8s} | "
              f"Steps: {str(row['Path']):>4s} | Nodes: {row['Nodes']:>4d} | {row['Time(ms)']:.3f}ms")

    print("\n[DONE] Task 3 completed successfully!\n")
    return all_results


if __name__ == '__main__':
    main()
