import cvxpy as cp
import numpy as np

'''Demo of the library'''
# Problem data.
m = 30
n = 20
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Construct the problem.
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A @ x - b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
print(x.value)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print(constraints[0].dual_value)

'''Least square example'''
# Import packages.
import cvxpy as cp
import numpy as np

# Generate data.
m = 20
n = 15
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Define and solve the CVXPY problem.
x = cp.Variable(n)
cost = cp.sum_squares(A @ x - b)
prob = cp.Problem(cp.Minimize(cost))
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("The optimal x is")
print(x.value)
print("The norm of the residual is ", cp.norm(A @ x - b, p=2).value)

''' Example adapted from https://www.cvxpy.org/examples/basic/least_squares.html'''
'Not possible to give the code any hint in the beginning, but the warm start is using the previous result, ' \
    'it might be already what is mentioned in paper 39 '

# Problem data.
m = 30
n = 20
# Y will be the spherical basis
Y = np.random.randn(m, n)
# h will be the hrtf set
h = np.random.randn(m)
# frequency dependent factor
Lambda = 0

# Define and solve the CVXPY problem.
w = cp.Variable(n)
cost = Lambda * cp.sum_squares(Y @ w - h) + (1-Lambda) * cp.sum_squares(cp.abs(Y) @ w - cp.abs(h)) # not taking the absolute value of x here but should take it
# constraints = [0 <= x, x <= 1]
prob = cp.Problem(cp.Minimize(cost))

# The optimal objective value is returned by `prob.solve()`.
prob.solve()
# Print result
print("\nThe optimal value is", prob.value)
print("The optimal x is")
print(x.value)
# print("The norm of the residual is ", cp.norm(A @ x - b, p=2).value)
