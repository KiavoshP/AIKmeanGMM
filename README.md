# Assignment 5 - Expectation Maximization

Expectation Maximization - Assignment 5 - CS3600

<p align="center">
<img src="images/k6_Starry.png" width="400"/> <br />

<img src="images/pcd_clustered.gif" width="400"/> 
</p>

#### Jupyter Notebook:
You will be using **jupyter notebook** to complete this assignment. 

To open the jupyter notebook, navigate to your assignment folder, activate the conda environment `conda activate ai_env`, and run `jupyter notebook`. 

Project description and all of the functions that you will implement are in `solution.ipynb` file.

**ATTENTION:** You are free to add additional cells for debugging your implementation, however, please don't write any inline code in the cells with function declarations, only edit the section *inside* the function, which has comments like: `# TODO: finish this function`.

## Grading

The grade you receive for the assignment will be distributed as follows:

1. k-Means Clustering (39 points)
2. Gaussian Mixture Model (60 points)
3. Bayesian Information Criterion (12 points EC)
4. Return your name (1 point)

**Note: For this assignment, we do not have any bonuses.**

## Submission
The tests for the assignment are provided in `mixture_tests.py`. All the tests are already embedded into the respective ipython notebook cells, so they will run automatically whenever you run the cells with your code. Local tests are sufficient for verifying the correctness of your implementation. The tests on Gradescope will be similar to the ones provided here. You'll need to ensure that your submissions are sufficiently vectorized so that algorithms won't time out.

To get the submission file, make sure to save your notebook and run:

`python notebook2script.py submit`

Once the execution is complete, open autogenerated `submit/submission.py` and verify that it contains all of the imports, functions, and classes you are required to implement. Only then proceed to the [Gradescope](https://www.gradescope.com/) for submission.

In your Gradescope submission history, you can mark certain submissions as **Active**. Please ensure this is your best submission.

#### Do NOT erase the #export at the top of any cells as it is used by notebook2script.py to extract cells for submission.

#### You will be allowed 3 submissions every 3 hours on gradescope. Make sure you test everything before submitting it. The code will be allowed to run for not more than 40 minutes per submission. In order for the code to run quickly, make sure to VECTORIZE the code (more on this in the notebook itself).


## Resources

1. Canvas lectures on Unsupervised Learning (Lesson 7)
2. The `gaussians.pdf`  in the `read/` folder will introduce you to multivariate normal distributions.
3. A youtube video by Alexander Ihler, on multivariate EM algorithm details:
[https://www.youtube.com/watch?v=qMTuMa86NzU](https://www.youtube.com/watch?v=qMTuMa86NzU)
4. The `em.pdf` chapter in the `read/` folder. This will be especially useful for Part 2 of the assignment.
5. Numpy and vectorization related
    * [Stackexchange discussion](https://softwareengineering.stackexchange.com/questions/254475/how-do-i-move-away-from-the-for-loop-school-of-thought)
    * [Hackernoon article](https://hackernoon.com/speeding-up-your-code-2-vectorizing-the-loops-with-numpy-e380e939bed3)
    * [Numpy einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) (highly recommended)
    * [Slicing and indexing](http://scipy-lectures.org/intro/numpy/array_object.html#indexing-and-slicing)
    * [Copies and views](http://scipy-lectures.org/intro/numpy/array_object.html#copies-and-views)
    * [Fancy indexing](http://scipy-lectures.org/intro/numpy/array_object.html#fancy-indexing)

