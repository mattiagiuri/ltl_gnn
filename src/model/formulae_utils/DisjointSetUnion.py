class DisjointSetUnion:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        """Find the root of the set containing x."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """Union the sets containing x and y."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            # Union by rank
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

    def add(self, x):
        """Add a new element to the DSU."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def get_partition(self):
        """Get the partition as a dictionary mapping a representative to its set."""
        sets = {}
        for element in self.parent:
            root = self.find(element)

            if root not in sets:
                sets[root] = set()
            sets[root].add(element)

        return sets


def partition_colors(strings):
    dsu = DisjointSetUnion()

    # Process each string
    for s in strings:
        colors = s.split("&")
        for color in colors:
            dsu.add(color)

        # Union all colors in the same string
        for i in range(len(colors) - 1):
            dsu.union(colors[i], colors[i + 1])

    # Get the final partition
    partition = dsu.get_partition()

    # Map each color to its corresponding set
    color_to_set = {color: partition[dsu.find(color)] for color in dsu.parent}
    return color_to_set, dsu


if __name__ == "__main__":
    # Example usage
    sample_vocab = {0: "PAD", 1: "EPSILON", 2: "NULL", 3: "green", 4: "blue", 5: "aqua", 6: "blue&green", 7:"blue&aqua", 8: "blue&aqua&green", 9: "red&magenta",
                    10: "yellow", 11: "red", 12: "orange", 13: "magenta", 14: "green&aqua", 15: ""}

    # strings = ["red&blue&green", "blue&yellow", "purple&orange", "green&yellow"]
    strings = list(sample_vocab.values())
    result, _ = partition_colors(strings)

    # Output each color and its set
    for color, color_set in result.items():
        print(f"{color}: {color_set}")
