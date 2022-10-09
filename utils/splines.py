import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, splev, splprep
import json, cv2

file = '/Users/aungriah/Documents/MT/MPC/SpeedMPC/Params/track.json'
with open(file,'r') as f:
    track = json.load(f)
    # X_opt = track['X_i']
    # Y_opt = track['Y_i']
    X_m = track['X']
    Y_m = track['Y']
    # X_o = track['X_o']
    # Y_o = track['Y_o']

plt.figure(1)
# plt.plot(X_opt,Y_opt, 'k')
# plt.plot(X_m,Y_m, 'r--')
plt.plot(X_m,Y_m, 'k')
plt.xlabel('X [m]',fontsize=20, fontname='Times')
plt.ylabel('Y [m]',fontsize=20, fontname='Times')
plt.title('Example track without measurement errors',fontsize=20, fontname='Times')
plt.show()


# X_opt = X_opt[500:1000]
# X_opt[200] += 5
# X_opt[220] -= 5
# Y_opt = Y_opt[500:1000]
# X_opt[280] -= 2
# Y_opt[300] += 2

# X_opt.append(X_opt[0])
# Y_opt.append(Y_opt[0])
# cubic spline fitting

# d = np.diff(np.array([X_opt,Y_opt]))
# s_opt = np.append([0],np.cumsum(LA.norm(d, axis=0)))
# cs_x = CubicSpline(s_opt, X_opt)
# cs_y = CubicSpline(s_opt, Y_opt)
#
#
# grad_x = cs_x(s_opt,1)
# gradgrad_x = cs_x(s_opt,2)
# grad_y = cs_y(s_opt,1)
# gradgrad_y = cs_y(s_opt,2)
#
# h_cs = np.arctan2(grad_y,grad_x)
# kappa_cs = (grad_x*gradgrad_y - grad_y*gradgrad_x)/np.power((grad_x**2 + grad_y**2),1.5)
# bsline fitting

# tck, u = splprep([X_opt, Y_opt], s=100,per=0)
# u2 = np.linspace(0,1,s_opt.size)
# new_points = splev(u2, tck,der=0)
# new_grad = splev(u2, tck,der=1)
# new_gradgrad = splev(u2, tck,der=2)
#
# h = np.arctan2(new_grad[1],new_grad[0])
# kappa = (new_grad[0]*new_gradgrad[1] - new_grad[1]*new_gradgrad[0])/np.power((new_grad[0]**2 + new_grad[1]**2),1.5)
# plt.rcParams['font.family'] = 'Times'
# plt.figure(1)
# plt.plot(kappa_cs, 'r')
# plt.plot(kappa, 'g')
# plt.xlabel('Arc length ' +  r'$s$', fontsize=20, fontname='Times')
# plt.ylabel('Curvature ' + r'$\kappa(s)$',fontsize=20, fontname='Times')
# plt.title('Curvature of Cubic spline vs. B-spline',fontsize=20, fontname='Times')
# plt.legend(('Cubic spline', 'B-spline'),fontsize=20)
# plt.figure(1)
# plt.plot(cs_x(s_opt), cs_y(s_opt), 'r')
# plt.show()
# plt.plot(new_points[0], new_points[1],'g--')
# plt.xlabel('X[m]', fontsize=20, fontname='Times')
# plt.ylabel('Y[m]',fontsize=20, fontname='Times')
# plt.title('Curve fitting with Cubic spline vs. B-spline',fontsize=20, fontname='Times')
# plt.legend(('Cubic spline', 'B-spline'),fontsize=20)
# print(len(new_points))
# plt.figure(2)
# plt.plot(X_opt, Y_opt, 'k')
# plt.plot(new_points[0], new_points[1],'g')
# plt.xlabel('X [m]',fontsize=20, fontname='Times')
# plt.ylabel('Y [m]',fontsize=20, fontname='Times')
# plt.title('Example track with measurement errors',fontsize=20, fontname='Times')
# plt.show()
# mean_rgb = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
# std_rgb = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
# image = 'sample.jpg'
# img = cv2.imread('frame0127.jpg')
# img = (img.astype(np.float32) / 255. - mean_rgb) / std_rgb
# cv2.imwrite('preprocessed2.jpg', img*255.)





