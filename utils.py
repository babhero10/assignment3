import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_image(img_path):
    return cv2.imread(img_path)


def show_images(imgs, titles):
    assert len(imgs) == len(titles)

    plt.figure(figsize=(15, 5))

    sub_plots_num = len(imgs)

    for i in range(sub_plots_num):
        plt.subplot(100 + 10 * sub_plots_num + i + 1)
        plt.imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def normalize(pts):
    centroid = np.mean(pts, axis=0)
    d = np.mean(np.linalg.norm(pts - centroid, axis=1))
    s = np.sqrt(2) / d
    T = np.array([[s, 0, -s*centroid[0]],
                  [0, s, -s*centroid[1]],
                  [0, 0, 1]])
    pts_homo = np.column_stack((pts, np.ones(len(pts))))
    pts_norm = (T @ pts_homo.T).T
    return pts_norm, T

def compute_F_8points(pts1, pts2):
    p1n, T1 = normalize(pts1)
    p2n, T2 = normalize(pts2)
    
    A = np.zeros((8, 9))
    for i in range(8):
        u, v = p1n[i, 0], p1n[i, 1]
        u_p, v_p = p2n[i, 0], p2n[i, 1]
        A[i] = [u_p*u, u_p*v, u_p, v_p*u, v_p*v, v_p, u, v, 1]
    
    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)
    
    U, S, Vt = np.linalg.svd(F_norm)
    S[2] = 0
    F_norm = U @ np.diag(S) @ Vt
    
    F = T2.T @ F_norm @ T1
    
    return F

def compute_F_7points(pts1, pts2):
    def get_coeffs(M, N):
        coeffs = np.zeros(4)
        
        def poly_val(l):
            return np.linalg.det(l * M + N)
        
        x = np.array([0, 1, -1, 2])
        y = np.array([poly_val(v) for v in x])
        p = np.polyfit(x, y, 3)
        return p
        
    p1n, T1 = normalize(pts1)
    p2n, T2 = normalize(pts2)
    
    A = np.zeros((7, 9))
    for i in range(7):
        u, v = p1n[i, 0], p1n[i, 1]
        u_p, v_p = p2n[i, 0], p2n[i, 1]
        A[i] = [u_p*u, u_p*v, u_p, v_p*u, v_p*v, v_p, u, v, 1]
    
    _, _, Vt = np.linalg.svd(A)
    f1 = Vt[-1].reshape(3, 3)
    f2 = Vt[-2].reshape(3, 3)
    
    M = f1 - f2
    N = f2
    
    coeffs = get_coeffs(M, N)
    
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    
    final_Fs = []
    for r in real_roots:
        F_norm = r * M + N
        F = T2.T @ F_norm @ T1
        if abs(F[2, 2]) > 1e-9:
            F /= F[2, 2]
        final_Fs.append(F)

    return final_Fs
        
def to_homo(p):
    return np.array([p[0], p[1], 1])
