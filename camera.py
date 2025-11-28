import cv2
import numpy as np
from scipy.linalg import eig
from sklearn.utils.extmath import randomized_svd

def dmd_window(X, r=None):
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    if r is not None:
        U, s, Vh = randomized_svd(X1, n_components=r)
    else:
        U, s, Vh = np.linalg.svd(X1, full_matrices=False)
    S_inv = np.diag(1.0 / s)
    Atilde = U.T @ X2 @ Vh.T @ S_inv
    lam, W = eig(Atilde)
    Phi = X2 @ Vh.T @ S_inv @ W
    fps = 30
    win = X.shape[1]
    dt = 1.0 / fps
    t = np.arange(win) * dt
    x0 = X[:, 0]
    b = np.linalg.lstsq(Phi, x0, rcond=None)[0]
    omega = np.log(lam) / dt
    dynamics = b[:, None] * np.exp(omega[:, None] * t[None, :])  
    X_dmd = Phi @ dynamics
    return omega, Phi, dynamics, X_dmd

# Parameters
win = 30
svd_rank = 5
eps = 0.1
H, Wd = 240, 320  
cap = cv2.VideoCapture(0)
buffer = []

# Window arrangement parameters
SCREEN_W, SCREEN_H = 1920, 1080     
WIN_W, WIN_H = Wd, H                
GAP = 50                            

total_width = 3 * WIN_W + 2 * GAP    
start_x = (SCREEN_W - total_width) // 2
pos_y = (SCREEN_H - WIN_H) // 2      

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Background", cv2.WINDOW_NORMAL)
cv2.namedWindow("Foreground", cv2.WINDOW_NORMAL)

cv2.moveWindow("Original",  start_x, pos_y)
cv2.moveWindow("Background", start_x + WIN_W + GAP, pos_y)
cv2.moveWindow("Foreground", start_x + 2*(WIN_W + GAP), pos_y)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (Wd, H))
    f = gray.astype(np.float32) / 255.0
    buffer.append(f)
    if len(buffer) > win:
        buffer.pop(0)
    if len(buffer) < win:
        cv2.imshow("Original", cv2.resize(frame, (Wd, H)))
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    F = np.stack(buffer, axis=0)
    X = F.reshape(win, -1).T

    omega, Phi, dynamics, X_dmd = dmd_window(X, r=svd_rank)
    idx_bg = np.where(np.abs(omega) < eps)[0]
    if idx_bg.size > 0:
        Phi_bg = Phi[:, idx_bg]
        dyn_bg = dynamics[idx_bg, :]
        X_bg = np.abs(Phi_bg @ dyn_bg)
    else:
        X_bg = np.abs(X_dmd)

    X_fg = X - X_bg
    R = np.minimum(X_fg, 0)
    X_fg = X_fg - R
    X_bg = X_bg + R
    t_last = -1

    bg_last = X_bg[:, t_last].reshape(H, Wd)
    orig_last = X[:, t_last].reshape(H, Wd)
    fg_last = X_fg[:, t_last].reshape(H, Wd)

    fg_last_boost = np.clip(fg_last * 4, 0, 1)
    orig_bgr = cv2.cvtColor((orig_last * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    bg_bgr = cv2.cvtColor(np.clip(bg_last * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    fg_bgr = cv2.cvtColor((fg_last_boost * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    cv2.imshow("Original", orig_bgr)
    cv2.imshow("Background", bg_bgr)
    cv2.imshow("Foreground", fg_bgr)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
