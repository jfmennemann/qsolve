"""
import torch

A = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]], dtype=torch.complex128)

# print(A.round())

Q, R = torch.linalg.qr(A)

# print(Q.round())
# print(R.round())

Q_times_R = Q @ R

# print(Q_times_R.round())

c_00 = torch.sum(torch.conj(Q[:, 0]) * Q[:, 0])
c_01 = torch.sum(torch.conj(Q[:, 0]) * Q[:, 1])

print(c_00.cpu().numpy())
print(c_01.cpu().numpy())

tmp = torch.adjoint(Q) @ Q

tmp = tmp.cpu().numpy()

print(tmp.round())

input()
"""