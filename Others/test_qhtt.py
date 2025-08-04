import numpy as np

B = np.array([
    [-1, 1, 12, 4],
    [ 2, 0,  8, 2],
    [ 1, 0,  0, -3],
    [ 0, 1,  2, -1]
])

B_inv = np.linalg.inv(B)
yT = np.array([-1, 0, 1, 0]) @ B_inv

print("yT:", yT)
print("B_inv:", B_inv)




# import numpy as np

# # Ma trận cơ sở B (các cột a1, a2, a3, a5)
# B = np.array([
#     [-1, 1, 12,  4],
#     [ 2, 0,  8,  2],
#     [ 1, 0,  0, -3],
#     [ 0, 1,  2, -1]
# ], dtype=float)

# # Véc-tơ b
# b = np.array([36, 48, 16, 40], dtype=float)

# # Giải hệ B * x_B = b
# x_B = np.linalg.solve(B, b)

# # Gán x4 = 0 và ghép thành vector đầy đủ
# x = np.array([x_B[0], x_B[1], x_B[2], 0.0, x_B[3]])

# print("Solution with x4 = 0:")
# for i, xi in enumerate(x, start=1):
#     print(f"x{i} = {xi}")
