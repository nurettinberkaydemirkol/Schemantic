from schemantic import VectorCube 

data = [
    (0, [0.1] * 32, "first"),
    (1, [0.9] * 32, "second"),
    (2, [0.2, 0.1] * 16, "third"),
]

cube = VectorCube(data, cluster_type="knn")  
query = [0.11] * 32
results = cube.query(query)

# first
print(results)