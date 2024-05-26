\Завдання 1 

### 1. Функція, яка реалізує розвертання односпрямованого зв'язаного списку
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
def reverse_linked_list(head):
    """
    Розгортає односпрямований зв'язаний список.
    Args:
        head (Node): Голова списку.
    Returns:
        Node: Нова голова розгорнутого списку.
    """
    prev_node = None
    current_node = head
    while current_node is not None:
        next_node = current_node.next
        current_node.next = prev_node
        prev_node = current_node
        current_node = next_node
    return prev_node 
### 2. Алгоритм сортування вставками для односпрямованого зв'язаного списку 
def insert_sort(head):
    """
    Сортує односпрямований зв'язаний список за допомогою алгоритму сортування вставками.
    Args:
        head (Node): Голова списку.
    Returns:
        Node: Голова відсортованого списку.
    """
    if head is None or head.next is None:
        return head
    sorted_head = head
    current_node = head.next
    while current_node is not None:
        # Зберігаємо посилання на наступний вузол
        next_node = current_node.next
        # Вставляємо поточний вузол у відсортований список
        sorted_head = insert_node(sorted_head, current_node)
        # Переходимо до наступного вузла
        current_node = next_node
    return sorted_head
def insert_node(head, node):
    """
    Вставляє вузол у відсортований односпрямований зв'язаний список.
    Args:
        head (Node): Голова списку.
        node (Node): Вузол, який потрібно вставити.
    Returns:
        Node: Нова голова списку після вставки.
    """
    # Якщо голова списку пуста або значення вузла менше за головного
    if head is None or node.data < head.data:
        node.next = head
        return node
    # Шукаємо позицію для вставки
    current = head
    while current.next is not None and node.data > current.next.data:
        current = current.next
    # Вставляємо вузол
    node.next = current.next
    current.next = node
    return head 
### 3. Функція, яка об'єднує два відсортованих односпрямованих зв'язаних списки в один відсортований список
def merge_sorted_lists(head1, head2):
    """
    Об'єднує два відсортованих односпрямованих зв'язаних списки в один відсортований список.
    Args:
        head1 (Node): Голова першого списку.
        head2 (Node): Голова другого списку.
    Returns:
        Node: Голова об'єднаного відсортованого списку.
    """
    if head1 is None:
        return head2
    if head2 is None:
        return head1
    if head1.data <= head2.data:
        head = head1
        head1 = head1.next
    else:
        head = head2
        head2 = head2.next
    current = head
    while head1 is not None and head2 is not None:
        if head1.data <= head2.data:
            current.next = head1
            head1 = head1.next
        else:
            current.next = head2
            head2 = head2.next
        current = current.next
    if head1 is not None:
        current.next = head1
    elif head2 is not None:
        current.next = head2
    return head 





\Завдання 2 

import turtle
def pythagorean_tree(length, angle, depth):
    """
    Рекурсивно створює фрактал Пифагорового дерева.
    Аргументи:
        length (float): початкова довжина гілки
        angle (float): кут розгалуження гілки
        depth (int): поточна глибина рекурсії
    """
    if depth == 0:
        return
    turtle.forward(length)
    turtle.left(angle)
    pythagorean_tree(length * 0.7, angle, depth - 1)
    turtle.right(2 * angle)
    pythagorean_tree(length * 0.7, angle, depth - 1)
    turtle.left(angle)
    turtle.backward(length)
def main():
    """
    Основна функція, яка запитує у користувача глибину рекурсії
    та створює фрактал Пифагорового дерева.
    """
    turtle.speed(0)
    turtle.penup()
    turtle.goto(-300, -300)
    turtle.pendown()
    depth = int(input("Введіть глибину рекурсії: "))
    pythagorean_tree(100, 45, depth)
    turtle.done()
if __name__ == "__main__":
    main() 



\Завдання 3 

import heapq
def dijkstra(graph, start):
    """
    Реалізує алгоритм Дейкстри для пошуку найкоротших шляхів у зваженому графі.
    Args:
        graph (dict): Словник, що представляє зважений граф, де ключі -- вершини,
                     а значення -- списки кортежів (вершина, вага).
        start (str): Початкова вершина.
    Returns:
        dict: Словник, де ключі -- вершини, а значення -- найкоротші відстані
              від початкової вершини.
    """
    dist = {v: float('inf') for v in graph}
    dist[start] = 0
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
    return dist
# Приклад використання
graph = {
    'A': [('B', 5), ('C', 1)],
    'B': [('A', 5), ('C', 2), ('D', 1)],
    'C': [('A', 1), ('B', 2), ('D', 4), ('E', 8)],
    'D': [('B', 1), ('C', 4), ('E', 3), ('T', 5)],
    'E': [('C', 8), ('D', 3), ('T', 5)]
}
start = 'A'
distances = dijkstra(graph, start)
print(f"Найкоротші відстані від вершини {start}:")
for vertex, distance in distances.items():
    print(f"{vertex}: {distance}") 
Найкоротші відстані від вершини A:
A: 0
B: 5
C: 1
D: 6
E: 9 



\Завдання 4 

import networkx as nx
import matplotlib.pyplot as plt
class Node:
    """
    Клас, що представляє вузол двійкового дерева.
    """
    def __init__(self, key, color="skyblue"):
        self.left = None
        self.right = None
        self.val = key
        self.color = color
def add_heap_edges(graph, node, pos, x=0, y=0, layer=1):
    """
    Функція, що додає ребра до графу та обчислює позиції вузлів для візуалізації двійкової купи.
    """
    if node is not None:
        graph.add_node(node.val, color=node.color)
        if node.left:
            graph.add_edge(node.val, node.left.val)
            l = x - 1 / 2 ** layer
            pos[node.left.val] = (l, y - 1)
            l = add_heap_edges(graph, node.left, pos, x=l, y=y - 1, layer=layer + 1)
        if node.right:
            graph.add_edge(node.val, node.right.val)
            r = x + 1 / 2 ** layer
            pos[node.right.val] = (r, y - 1)
            r = add_heap_edges(graph, node.right, pos, x=r, y=y - 1, layer=layer + 1)
    return graph
def draw_heap(heap_root):
    """
    Функція, що візуалізує двійкову купу.
    """
    heap = nx.DiGraph()
    pos = {(heap_root.val): (0, 0)}
    heap = add_heap_edges(heap, heap_root, pos)
    colors = [node[1]['color'] for node in heap.nodes(data=True)]
    plt.figure(figsize=(8, 5))
    nx.draw(heap, pos=pos, with_labels=True, arrows=False, node_size=2500, node_color=colors)
    plt.show()
# Приклад використання
root = Node(0)
root.left = Node(4)
root.left.left = Node(5)
root.left.right = Node(10)
root.right = Node(1)
root.right.left = Node(3)
draw_heap(root) 





\Завдання 5 

import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import colorsys
class Node:
    """Клас вузла бінарного дерева"""
    def __init__(self, key, color="skyblue"):
        self.left = None
        self.right = None
        self.val = key
        self.color = color
def add_edges(graph, node, pos, x=0, y=0, layer=1):
    """Рекурсивна функція для додавання ребер та вузлів до графа"""
    if node is not None:
        graph.add_node(node.val, color=node.color)
        if node.left:
            graph.add_edge(node.val, node.left.val)
            l = x - 1 / 2 ** layer
            pos[node.left.val] = (l, y - 1)
            l = add_edges(graph, node.left, pos, x=l, y=y - 1, layer=layer + 1)
        if node.right:
            graph.add_edge(node.val, node.right.val)
            r = x + 1 / 2 ** layer
            pos[node.right.val] = (r, y - 1)
            r = add_edges(graph, node.right, pos, x=r, y=y - 1, layer=layer + 1)
    return graph
def dfs_traversal(root, colors):
    """Функція обходу дерева в глибину"""
    if root is None:
        return
    colors.append(root.color)
    dfs_traversal(root.left, colors)
    dfs_traversal(root.right, colors)
def bfs_traversal(root, colors):
    """Функція обходу дерева в ширину"""
    if root is None:
        return
    queue = deque([root])
    while queue:
        node = queue.popleft()
        colors.append(node.color)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
def draw_tree(tree_root, traversal_type):
    """Функція для візуалізації бінарного дерева"""
    tree = nx.DiGraph()
    pos = {(tree_root.val): (0, 0)}
    tree = add_edges(tree, tree_root, pos)
    colors = []
    if traversal_type == 'dfs':
        dfs_traversal(tree_root, colors)
    elif traversal_type == 'bfs':
        bfs_traversal(tree_root, colors)
    # Змінюємо кольори вузлів від темного до світлого відтінку
    hsv_colors = [(col_idx / len(colors), 1, 1) for col_idx in range(len(colors))]
    rgb_colors = ['#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)) for r, g, b in [colorsys.hsv_to_rgb(*hsv) for hsv in hsv_colors]]
    plt.figure(figsize=(8, 5))
    nx.draw(tree, pos=pos, with_labels=True, arrows=False, node_size=2500, node_color=rgb_colors)
    plt.title(f"{traversal_type.upper()} Traversal")
    plt.show()
# Створюємо дерево
root = Node(0)
root.left = Node(4)
root.left.left = Node(5)
root.left.right = Node(10)
root.right = Node(1)
root.right.left = Node(3)
# Відображаємо дерево з обходом в глибину та в ширину
draw_tree(root, 'dfs')
draw_tree(root, 'bfs') 





\Завдання 6 

def greedy_algorithm(items, budget):
    """
    Реалізація жадібного алгоритму для вирішення задачі вибору їжі.
    Функція повертає список страв, які максимізують співвідношення
    калорій до вартості, без перевищення бюджету.
    Args:
        items (dict): Словник страв із інформацією про вартість і калорійність.
        budget (int): Обмежений бюджет.
    Returns:
        list: Список обраних страв.
    """
    sorted_items = sorted(items.items(), key=lambda x: x[1]["calories"] / x[1]["cost"], reverse=True)
    selected_items = []
    total_cost = 0
    for item, info in sorted_items:
        if total_cost + info["cost"] <= budget:
            selected_items.append(item)
            total_cost += info["cost"]
    return selected_items
def dynamic_programming(items, budget):
    """
    Реалізація алгоритму динамічного програмування для вирішення задачі вибору їжі.
    Функція повертає оптимальний набір страв, який максимізує вміст калорій
    у межах заданого бюджету.
    Args:
        items (dict): Словник страв із інформацією про вартість і калорійність.
        budget (int): Обмежений бюджет.
    Returns:
        list: Список обраних страв.
    """
    n = len(items)
    dp = [[0] * (budget + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        item_name, item_info = list(items.items())[i - 1]
        item_cost, item_calories = item_info["cost"], item_info["calories"]
        for j in range(1, budget + 1):
            if j >= item_cost:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - item_cost] + item_calories)
            else:
                dp[i][j] = dp[i - 1][j]
    selected_items = []
    j = budget
    for i in range(n, 0, -1):
        if dp[i][j] != dp[i - 1][j]:
            selected_items.append(list(items.keys())[i - 1])
            j -= list(items.values())[i - 1]["cost"]
    return selected_items
# Використання функцій
items = {
    "pizza": {"cost": 50, "calories": 300},
    "hamburger": {"cost": 40, "calories": 250},
    "hot-dog": {"cost": 30, "calories": 200},
    "pepsi": {"cost": 10, "calories": 100},
    "cola": {"cost": 15, "calories": 220},
    "potato": {"cost": 25, "calories": 350}
}
budget = 100
print("Жадібний алгоритм:")
print(greedy_algorithm(items, budget))
print("\nАлгоритм динамічного програмування:")
print(dynamic_programming(items, budget)) 






\Завдання 7 

import random
import matplotlib.pyplot as plt
def simulate_dice_throws(num_throws):
    """Симуляція кидків кубиків та обчислення ймовірностей сум чисел на них."""
    sums_count = {i: 0 for i in range(2, 13)}
    
    for _ in range(num_throws):
        dice1 = random.randint(1, 6)
        dice2 = random.randint(1, 6)
        dice_sum = dice1 + dice2
        sums_count[dice_sum] += 1
    
    probabilities = {
        sum_val: count / num_throws 
        for sum_val, count in sums_count.items()
    }
    
    return probabilities
def plot_probabilities(probabilities):
    """Побудова графіка ймовірностей сум чисел на кубиках."""
    sums = list(probabilities.keys())
    probs = list(probabilities.values())
    
    plt.bar(sums, probs, color="blue", alpha=0.7)
    plt.xlabel("Сума чисел на кубиках")
    plt.ylabel("Ймовірність")
    plt.title("Ймовірності сум чисел на кубиках (Метод Монте-Карло)")
    plt.xticks(sums)
    plt.grid(True)
    plt.show()
def main():
    """Основна функція для виконання симуляції та виводу результатів."""
    num_throws = 1_000_000  # Велика кількість кидків для точної симуляції
    probabilities = simulate_dice_throws(num_throws)
    
    print("| Сума | Ймовірність (%) |")
    print("| ---- | ---------------- |")
    for sum_val in range(2, 13):
        print(f"| {sum_val}    | {probabilities[sum_val]*100:.2f}% |")
      
    
    plot_probabilities(probabilities)
if __name__ == "__main__":
    main()
