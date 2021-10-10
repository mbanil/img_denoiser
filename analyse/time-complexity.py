
import matplotlib.pyplot as plt
import numpy as np

imgSize = [128,256,512,1024]
pixels = [number ** 2 for number in imgSize]
# t1 = np.array([0.75,1.895,22.13,283.056])
# t2 = np.array([0.79,1.994,25.386,398.365])
# t3 = np.array([0.75,2.156,18.336,2.156])

# t = np.add(t1,t2)
# t = np.add(t,t3)
# t = t/3

pixels = np.array([128*128, 256*256,500*600,500*800,625*800,600*1024,800*1024,1024*1024])

t = np.array([0.79,1.99, 25.91,43.12,65.33,103.76,198.67,313.77])
  
# plotting the points 
plt.plot(pixels, t,'-bo')
plt.xlabel('No. of pixels')
plt.ylabel('Time in secs')
plt.title('Time-Complexity')

plt.savefig('C:/My Documents/TUD-MCL/Semester 4/Thesis/repo/img-denoiser/analyse/plot.png')    
