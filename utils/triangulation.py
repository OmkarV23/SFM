import cupy as cp
import numpy as np
from utils import projection_mat
import scipy.optimize as optimize

class Triangulation():

    def __init__(self, K, C1, R1, inliers):
        self.K = K
        # self.C1 = C1, self.R1 = R1
        # self.C2 = C2, self.R2 = R2
        self.u1 = inliers['u1']
        self.u2 = inliers['u2']
        self.proj_mat1 = projection_mat(K,R1,C1[:,None])

    def LinearTriangulation(self, C2, R2):

        def skew(pts):
            return cp.array([[0, -pts[2], pts[1]], [pts[2], 0, -pts[0]], [-pts[1], pts[0], 0]])

        world_pts = []

        proj_mat2 = projection_mat(self.K,R2,C2[:,None])
        for i,j in zip(self.u1,self.u2):
            mat = cp.vstack((skew(i) @ self.proj_mat1, skew(j) @ proj_mat2))
            u,s,vt = cp.linalg.svd(mat)
            X = vt[-1] / vt[-1,-1]
            X = cp.reshape(X, (len(X), -1)).transpose()
            world_pts.append(X)
        world_pts = cp.array(world_pts)
        return world_pts

    def NonLinearTriangulation(self, C2, R2, X3d):
        proj_mat2 = projection_mat(self.K,R2,C2[:,None]).get()
        
        def projectionloss(X, pts1, pts2, P1, P2):
            p1_1T, p1_2T, p1_3T = P1 # rows of P1
            p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,-1), p1_2T.reshape(1,-1),p1_3T.reshape(1,-1)
            
            p2_1T, p2_2T, p2_3T = P2 # rows of P2
            p2_1T, p2_2T, p2_3T = p2_1T.reshape(1,-1), p2_2T.reshape(1,-1), p2_3T.reshape(1,-1)

            ## reprojection error for reference camera points - j = 1
            u1,v1 = pts1[0], pts1[1]
            u1_proj = np.divide(p1_1T.dot(X) , p1_3T.dot(X))
            v1_proj = np.divide(p1_2T.dot(X) , p1_3T.dot(X))
            E1= np.square(v1 - v1_proj) + np.square(u1 - u1_proj)

            
            ## reprojection error for second camera points - j = 2    
            u2,v2 = pts2[0], pts2[1]
            u2_proj = np.divide(p2_1T.dot(X) , p2_3T.dot(X))
            v2_proj = np.divide(p2_2T.dot(X) , p2_3T.dot(X))    
            E2= np.square(v2 - v2_proj) + np.square(u2 - u2_proj)
            
            error = E1 + E2
            return error.squeeze()

        x3D_ = []
        for i in range(len(X3d)):
            optimized_params = optimize.least_squares(fun=projectionloss, x0=X3d[i].get(), method="trf", args=[self.u1[i], self.u2[i], self.proj_mat1.get(), proj_mat2])
            X1 = optimized_params.x
            x3D_.append(X1)
        return cp.array(x3D_)

