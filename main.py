import json
import heapq


def load_json(filename: str) -> dict:
    with open(filename, 'r') as file:
        data = json.load(file)
    return data
# Helper function for reading a json file


class Node:
    def __init__(self, num: str, pred: object, dist: float, cost: float):
        self.num = num
        self.pred = pred  # Predecessor of the current node, needed to get the path
        self.dist = dist
        self.cost = cost
        # dist and cost are the cummulative dsitances/costs required to reach the node from start

    def __lt__(self, next):
        return self.dist < next.dist
    # Overriding the comparison operator to order nodes in priority queue by their distance


class Graph:
    def __init__(self):
        self.adj_list = load_json('G.json')
        # The graph is given as an adjacency list where the neighbor list of node ‘v’ can be accessed with G[‘v’]

        self.coord = load_json('Coord.json')
        # The coordination of a node ‘v’ is a pair (X, Y) which can be accessed with Coord[‘v’]

        self.dist = load_json('Dist.json')
        # The distance between a pair of node (v, w) can be accessed with Dist[‘v,w’]

        self.cost = load_json('Cost.json')
        # The energy cost between a pair of node (v, w) can be accessed with Cost[‘v,w’]

        print(
            f"Initialised graph from the JSON files. {len(self.coord)} nodes and {len(self.dist)} edges\n")

    def getAdj(self, node: str) -> 'list[str]':
        return self.adj_list[node]
    # Helper function that returns the neighbours of a node

    def getCoord(self, node: str) -> 'list[float, float]':
        return self.coord[node]
    # Helper function that returns the coordinates of a node

    def getDist(self, node1: str, node2: str) -> float:
        return self.dist[f"{node1},{node2}"]
    # Helper function that returns the distance between a pair of nodes

    def getCost(self, node1: str, node2: str) -> float:
        return self.cost[f"{node1},{node2}"]
    # Helper function that returns the cost between a pair of nodes

    def getPath(self, start: str, end: Node) -> 'list[str]':
        curr = end
        path = [end.num]
        while path[-1] != start:
            curr = curr.pred
            path.append(curr.num)
        return path[::-1]
    # Helper function that returns the path, leveraging the predecessor attribute of the Node object

    def getEuclidean(self, node1: str, node2: str) -> float:
        x1, y1 = self.getCoord(node1)
        x2, y2 = self.getCoord(node2)
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5
    # Helper function that returns the euclidean distance between the two nodes

    def astar(self, start: str, end: str, budget: int = 0, ucs: bool = False):
        distances = {start: 0.}
        # Dictionary that tracks the path's length from start node to current node {g(n)}

        costs = {start: 0.}
        # Dictionary that tracks the path's energy cost from start node to current node {Used for task 2 and 3}

        startNode = Node(start, None, 0., 0.)
        pq = [(0., startNode)]
        # Priority Queue (minimizing heap) that prioritizes nodes based on their heuristic values
        # Note how we are enqueuing the node objects

        while pq:
            _, curr = heapq.heappop(pq)
            # Deque node object from heap with lowest priority value {heuristic in this case}

            if curr.num == end:
                endNode = curr
                path = self.getPath(start, endNode)
                return (path, list(distances.keys()), curr.dist, curr.cost)
            # If the target is found. Note how we terminate once the target is popped from the priority queue

            for adj in self.getAdj(curr.num):
                new_distance = curr.dist + self.getDist(curr.num, adj)
                new_cost = curr.cost + self.getCost(curr.num, adj)
                # The helper funtions take string arguments
                # Note how we are using curr.dist instead of distances[curr]
                # This was causing the bug :(

                if budget > 0 and new_cost > budget:
                    continue
                # If exploring this node exceeds our budget, we skip it {it won't get enqueued}

                heuristic_dist = 0 if ucs else self.getEuclidean(adj, end)
                # The ucs flag allows us to switch between UCS and A* algorithms easily

                heuristic = new_distance + heuristic_dist
                # f(n) = g(n) + h(n) {standard unweighted A*}
                # Setting h(n) = 0 makes it UCS/Djikstra

                if new_distance < distances.get(adj, float('inf')) \
                        or new_cost < costs.get(adj, float('inf')):
                    distances[adj] = new_distance
                    costs[adj] = new_cost
                    # Updating the state dicts

                    adjNode = Node(adj, curr, new_distance, new_cost)
                    heapq.heappush(pq, (heuristic, adjNode))
                    # Updating the priority queue
                    # We use the Lazy Deletion method to perform edge relaxation vs an Indexed Priority Queue

                # Check if relaxed distance is shorter {unvisited nodes have infinite distance}

        return (self.getPath(start, curr), list(distances.keys()), distances[curr.num], costs[curr.num])
        # If target is not found, we return the best path we have so far


def printPath(path):
    print("Shortest Path: S->", end="")
    for node in path[1:-1]:
        print(f"{node}->", end="")
    print("T\n")
    # Helper function that will print the path


# def checkPath(path):
#     G = Graph()
#     distance, cost = 0, 0
#     for i in range(len(path)-1):
#         x, y = path[i], path[i+1]
#         distance += G.getDist(x, y)
#         cost += G.getCost(x, y)

#     print(f"Verified Distance: {distance}")
#     print(f"Verified Cost: {cost}")
#     return
#     # Helper function to verify our output


def printKeyStats(total_cost, total_distance):
    print(f"Shortest Distance: {total_distance}")
    print(f"Total Energy Cost: {total_cost}")
    # Helper function that will print the key statistics


def printExtraStats(path, explored):
    print(f"Path Length: {len(path)}")
    print(f"Nodes Explored: {len(explored)}")
    # Helper function that prints extra statistics used for debugging


def printFinalResults(start, end, budget, ucs, graph, printextra):
    path, explored, total_distance, total_cost = graph.astar(
        start, end, budget, ucs)

    printPath(path)
    # checkPath(path)
    printKeyStats(total_cost, total_distance)
    if printextra:
        printExtraStats(path, explored)
    # Function that will print the final results neatly


def main():
    graph = Graph()
    start = '1'
    end = '50'
    printExtra = True
    # Flag for printing extra stats (Path Length and Number of Nodes Explored)

    # For Task 1: budget = 0; ucs = True
    print("\n\n---Task 1---\n")
    printFinalResults(start, end, 0, True, graph, printExtra)

    # For Task 2: budget = 287932; ucs = True
    print("\n\n---Task 2---\n")
    printFinalResults(start, end, 287932, True, graph, printExtra)

    # For Task 3: budget = 287932; ucs = False (will run A*)
    print("\n\n---Task 3---\n")
    printFinalResults(start, end, 287932, False, graph, printExtra)


if __name__ == "__main__":
    main()
