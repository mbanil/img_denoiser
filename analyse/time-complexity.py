
import matplotlib.pyplot as plt
import numpy as np

imgSize = [128,256,512,1024]
t1 = np.array([0.75,1.895,22.13,283.056])
t2 = np.array([0.79,1.994,25.386,398.365])
t3 = np.array([0.75,2.156,18.336,2.156])

t = np.add(t1,t2)
t = np.add(t,t3)
t = t/3
  
# plotting the points 
plt.plot(imgSize, t)
plt.xlabel('imageSize(n*n)')
plt.ylabel('Time in secs')
plt.title('Time-Complexity')

plt.savefig('C:/My Documents/TUD-MCL/Semester 4/Thesis/repo/img-denoiser/analyse/plot.png')    
