**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1**

* Ziad Ben Hadj-Alouane
  * [LinkedIn](https://www.linkedin.com/in/ziadbha/), [personal website](https://www.seas.upenn.edu/~ziadb/)
* Tested on: Windows 10, i7-8750H @ 2.20GHz, 16GB, GTX 1060

---
## Output
<p align="center">
  <img width="400" height="400" src="https://github.com/ziedbha/Project1-CUDA-Flocking/blob/master/images/flocking.gif"/>
  <img width="400" height="400" src="https://github.com/ziedbha/Project1-CUDA-Flocking/blob/master/images/flocking-screenshot.png"/>
</p>

---

## Performance Analysis
* For each implementation, how does changing the number of boids affect performance? Why do you think this is?
	* The more boids we have, the more FPS we lose. Each boid will need its own thread, and at some point we reach the max amount of threads of our GPU, meaning other threads have to wait to be executed, which means we lose performance and therefore FPS.
<p align="center">
  <img src="https://github.com/ziedbha/Project1-CUDA-Flocking/blob/master/images/Performance%20Analysis/boids_vs_fps.png"/>
</p>

* For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
	* There isn't much improvement since my GPU's warp size is 32. However, there are some cases where I chose a warp size less than 32, which meant that the GPU wasn't loading the max amount of data per instruction. There are also other cases where the block size was not a multiple of 32 which also had the same effect. In general, it is best to stick to multiples of 32 above 32 (i.e 128 is a good number!)
	<p align="center">
  <img src="https://github.com/ziedbha/Project1-CUDA-Flocking/blob/master/images/Performance%20Analysis/block_vs_fps.png"/>
</p>

* For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
	* Yes, coherent grid was more performant in all of my tests. Memory stored is contiguous in the coherent case, making cache misses less likely since we check memory in a contiguous fashion.
 
* Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? 
	* On paper, the trade-off looks minimal since the cell width is half as big, so the cell itself is 1/4 the volume of the original cell, which makes checking 27 cells not as big of a trade-off. However, now we have to allocate more memory (start & end indices) since we have more cells in our grid. This made the pre-processing step a bit slower compared to the 8-cell case. In addition, the parallel computations were affected from this more expensive memory allocation since we risk cache missing more frequently. This hurt the performance a bit for the grid implementations, but not as much on my machine.
