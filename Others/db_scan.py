# import numpy as np
# from sklearn.cluster import DBSCAN
# import matplotlib.pyplot as plt

# # Data points and labels
# points = np.array([
#     (2, 10),  # A1
#     (2,  5),  # A2
#     (8,  4),  # A3
#     (5,  8),  # A4
#     (7,  5),  # A5
#     (6,  4),  # A6
#     (1,  2),  # A7
#     (4,  9),  # A8
# ])
# names = ['A1','A2','A3','A4','A5','A6','A7','A8']

# # Run DBSCAN
# db = DBSCAN(eps=2, min_samples=2, metric='euclidean')
# db.fit(points)
# labels = db.labels_

# # Plot
# plt.figure(figsize=(6, 6))
# for lbl in np.unique(labels):
#     mask = labels == lbl
#     plt.scatter(points[mask, 0], points[mask, 1], label=f'Cluster {lbl}' if lbl != -1 else 'Noise')

# # Annotate points
# for (x, y), name in zip(points, names):
#     plt.text(x + 0.1, y + 0.1, name)

# plt.title('DBSCAN Clustering (eps=2, MinPts=2)')
# plt.xlabel('X coordinate')
# plt.ylabel('Y coordinate')
# plt.legend()
# plt.tight_layout()
# plt.show()



# import numpy as np
# from scipy.cluster.hierarchy import linkage, dendrogram
# from sklearn.cluster import AgglomerativeClustering
# import matplotlib.pyplot as plt

# # Dataset điểm
# points = np.array([
#     (2, 10),  # A1
#     (2, 5),   # A2
#     (8, 4),   # A3
#     (5, 8),   # A4
#     (7, 5),   # A5
#     (6, 4),   # A6
#     (1, 2),   # A7
#     (4, 9),   # A8
# ])

# # 1. Tính ma trận liên kết bằng Single linkage
# Z = linkage(points, method='single', metric='euclidean')

# # Vẽ dendrogram
# plt.figure(figsize=(8, 4))
# dendrogram(
#     Z,
#     labels=[f"A{i+1}" for i in range(points.shape[0])],
#     leaf_rotation=0,
#     leaf_font_size=10
# )
# plt.title("Dendrogram AGNES (Single Linkage)")
# plt.xlabel("Điểm dữ liệu")
# plt.ylabel("Khoảng cách")
# plt.tight_layout()
# plt.show()

# # 2. Ví dụ gom cụm với số cụm = 3
# clustering = AgglomerativeClustering(
#     n_clusters=3,
#     affinity='euclidean',
#     linkage='single'
# )
# labels = clustering.fit_predict(points)

# # In nhãn cụm
# import pandas as pd
# df = pd.DataFrame(points, columns=['x', 'y'])
# df['Cluster'] = labels + 1  # đánh số cụm từ 1

# import ace_tools as tools; tools.display_dataframe_to_user(name="Kết quả gom cụm AGNES", dataframe=df)
# # Vẽ scatter plot với nhãn cụm


# import numpy as np
# import matplotlib.pyplot as plt

# # Dữ liệu và nhãn cụm từ ví dụ trước
# points = np.array([
#     (2, 10),  # A1
#     (2, 5),   # A2
#     (8, 4),   # A3
#     (5, 8),   # A4
#     (7, 5),   # A5
#     (6, 4),   # A6
#     (1, 2),   # A7
#     (4, 9),   # A8
# ])
# labels = np.array([3, 1, 2, 3, 2, 2, 1, 3])  # Nhãn cụm

# # Vẽ scatter plot
# plt.figure(figsize=(6, 6))
# for cluster in np.unique(labels):
#     mask = labels == cluster
#     plt.scatter(points[mask, 0], points[mask, 1], label=f"Cụm {cluster}")

# # Gắn nhãn điểm
# for i, (x, y) in enumerate(points):
#     plt.text(x + 0.1, y + 0.1, f"A{i+1}")

# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Minh hoạ gom cụm AGNES trên trục x-y")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd

# Tập điểm dữ liệu
points = np.array([
    (2, 10),  # A1
    (2, 5),   # A2
    (8, 4),   # A3
    (5, 8),   # A4
    (7, 5),   # A5
    (6, 4),   # A6
    (1, 2),   # A7
    (4, 9),   # A8
])
labels = DBSCAN(eps=2, min_samples=2).fit_predict(points)

# Hiển thị kết quả trong DataFrame
df = pd.DataFrame(points, columns=['x', 'y'])
df['Cluster'] = labels
# import ace_tools as tools; tools.display_dataframe_to_user(name="Kết quả DBSCAN", dataframe=df)

# Vẽ scatter plot
plt.figure(figsize=(6,6))
unique_labels = set(labels)
for lbl in unique_labels:
    mask = labels == lbl
    if lbl == -1:
        label_name = 'Noise'
        marker = 'x'
    else:
        label_name = f'Cluster {lbl+1}'
        marker = 'o'
    plt.scatter(points[mask, 0], points[mask, 1],
                label=label_name, marker=marker)
# Gắn nhãn từng điểm
for i, (x, y) in enumerate(points):
    plt.text(x+0.1, y+0.1, f"A{i+1}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Kết quả phân cụm DBSCAN (eps=2, min_samples=2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
