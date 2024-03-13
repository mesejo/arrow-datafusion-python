import re
import time
from collections import deque
from operator import itemgetter

import numpy as np
import pandas as pd
import xgboost as xgb
import json
from numba import njit


@njit
def inference(conditions, indices, children, weights, depth, table):
    table_length = len(table)
    cursor = np.zeros(table_length, dtype=np.int32)
    for i in range(depth + 1):
        thresholds = conditions[cursor]
        features = indices[cursor]

        values = np.empty(table_length, dtype=np.float32)
        for j, (r, f) in enumerate(zip(table, features)):
            values[j] = r[f]

        branch = (values > thresholds).astype(np.int32)

        for j, (c, b) in enumerate(zip(cursor, branch)):
            cursor[j] = children[c, b]

    return cursor


model = xgb.XGBClassifier(objective="binary:logistic")
model.load_model("frauds-model-single-tree.json")

data = json.loads(model.get_booster().get_dump(dump_format="json")[0])

fields = itemgetter("missing", "no", "yes", "split", "split_condition")
children = itemgetter("children")
stack = deque([data])

left = []
right = []
indices = []
conditions = []
weights = []
depth = 0

branches = deque([[0]])
leafs = {}
while stack:
    node = stack.popleft()
    branch = branches.popleft()
    try:
        missing, no, yes, split, split_condition = fields(node)
    except KeyError:
        if "leaf" in node:
            left.append(node["nodeid"])
            right.append(node["nodeid"])
            indices.append(0)
            conditions.append(0)
            weights.append(node["leaf"])
            depth = max(node.get("depth", 0), depth)
            leafs[node["nodeid"]] = branch[:] + [node["nodeid"]]
            continue

    left.append(yes)
    right.append(no)
    indices.append(int(re.sub(r"\D+", "", split)))
    conditions.append(split_condition)
    weights.append(0)
    depth = max(node.get("depth", 0), depth)

    for child in children(node):
        stack.append(child)
        branches.append(branch[:] + [child["nodeid"]])

children = np.stack([np.array(left), np.array(right)], axis=-1)
indices = np.array(indices)
conditions = np.array(conditions)
weights = np.array(weights)


csv_table = pd.read_csv("frauds.csv", dtype=np.float32)
print(csv_table["class"].value_counts())

csv_table = csv_table.drop("class", axis=1)
table = csv_table.to_numpy()

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.perf_counter()
inference(conditions, indices, children, weights, depth, table)
end = time.perf_counter()
print("Elapsed (with compilation) = {}s".format((end - start)))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.perf_counter()
predictions = inference(conditions, indices, children, weights, depth, table)
end = time.perf_counter()
print("Elapsed (after compilation) = {}s".format((end - start)))

start = time.perf_counter()
pred_leaf_index = model.get_booster().predict(xgb.DMatrix(table), pred_leaf=True)
end = time.perf_counter()
print("Elapsed (xgboost) = {}s".format((end - start)))

for i, (p, p1) in enumerate(zip(pred_leaf_index.astype(np.int64), predictions)):
    if p != p1:
        print("diff", i, p, p1)
        print(leafs[p])
        print(leafs[p1])

print(conditions[41])
print(conditions[11])
print(conditions[87])
# diff 167995 114 113
# diff 176635 96 123
# diff 207523 114 77

mask = pred_leaf_index.astype(np.int64) != predictions
csv_table = csv_table[mask]
print(csv_table.iloc[:, [indices[11], indices[87], indices[41]]])
