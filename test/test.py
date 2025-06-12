from schemantic import VectorCube 

data = [
    (0, [0.1] * 32, "ilk belge"),
    (1, [0.9] * 32, "ikinci belge"),
    (2, [0.2, 0.1] * 16, "üçüncü belge"),
]

cube = VectorCube(data)

query = [0.87] * 32

results = cube.query(query)

print(results)