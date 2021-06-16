# Topic 3 -- Optimizing Your Learning Algorithm

## Overview of this Repository
This repository contains all the teaching material related to **Optimizing Your Learning Algorithm**. The master `branch` contains the sample code for the instructor to **reference**, and the `workshop` branch contains the **empty notebooks** for the instructor and students to program in.

## For the Instructors:

This section details instructions to guide the instructor in delivering the course. The instructor should fill out the blank notebooks in the `workshop` branch according to the reference in the `master` branch (if it is possible, use dual monitors so you can have the reference code opened side by side.) **There will be function calls already written in the blank notebook, please run those calls without modifications.**

Below is the curriculum of this repository, as well as the order of content to be delivered. **Be sure to familiarize yourself with the code before teaching! Feel free to explore the notebooks for course material as well as the programming exercise `Cat-astrophe.ipynb`.**

### Getting set up (Jupyter Notebok)
1. Clone this repo into a working directory
2. Switch to the `workshop` branch:
   ```bash
    $ git checkout workshop
   ```
3. Create and activate a virtual environment with the command below:
    ```bash
    # MacOS/Linux
    $ python3 -m venv env 
    $ source env/bin/activate
    ```
    ```bat
    :: Windows
    \> python -m venv env 
    \> .\env\Scripts\activate
    ```
4. If you are in the virtual environment, you should see the `(env)` marker. Now, install all the dependencies:
    ```bash
    # MacOS/Linux
    $ pip install -r requirements.txt
    ```
    ```bat
    :: Windows
    \> python -m pip install -r requirements.txt
    ```
5. You are ready to go!

### Getting Started (Google Colab)
1. Clone this repo into your working directory
2. Switch to the `workshop` branch
    ```bash
    $ git checkout workshop
    ```
3. Upload whichever notebook you need to work on into Colab.
4. Drag the `colab.zip` file into Colab.
5. Unzip the file and install the dependencies using `pip` within Colab.
6. You're ready to go!


## Topic 3 -- Optimizing Your Learning Algorithm Notebook

### Installing Dependencies
- Talk a bit about the dependencies you're using

### Feature Scaling
- Must cover the cost function of non-scaled and scaled inputs
- Try demonstrating gradient descent in onenote
    - Show why its easier to perform gradient descent on the scaled input cost function

### Bias and Variance 
- General Gist:
    - Look for clues in the text for what to talk about.
    - code is commented to make it easier to explain to students what the code does.

- Start off by laying out an experiment 
    -  fit `cars.csv` dataset with polynomials with degree = 1,3 and 10, see which one does better
    - Polynomial degree = 1 is going to **underfit**, degree = 3 is going to **fit very well**, degree = 10 is going to **overfit**

- Training Models Different Degrees
    - Paraphrase the text to explain what is going on to students
    - Commentate on your code when you program
    - Comment on the $R^2$ scores

- Overfitting and Underfitting
    - Paraphrase the text when teaching 
    - **Mention Underfitting = High bias and Overfitting = High Variance**
    - Be sure to explain the two concepts before explaining overfitting and underfitting
    - Explain overfitting, underfitting by showing the graphs and commentating on what's shown. 
    - Be sure to comment on how the function is fitting the training set, as well as the testing set

- Observations
    - Ask students to fill out the observations table
  

### Cross Validation
- Be sure to talk about how cross validation is very similar to the test set, and how we use it during training to evaluate how our model is performing on unseen data.
- Stress that it is not used to train, ie. it is not used to update the weights and bias
- Cost plots
    - Cost over a number of iterations (cost over time)
    - MUST mention the differences between the training cost and cross validation costs.

- Secondary Metric
    - Mention how looking at just training and validation loss is not enough
    - Recall the GPU performance predictor we made, where the loss was huge, but we were not underfitting
    - display $R^2$ / Accuracy and F1 alongside loss

### How to Deal with High Bias and High Variance
- Paraphrase from text

### Regularization
- Paraphrase from the text
- Must mention that regularization reduces the complexity of the learning algorithm, preventing it from overfitting
    - Maybe also say that it "trades" some variance for bias.

- L2 Regularization
    - Use L2 Regularization to prevent the degree 10 polynomial from overfitting
- Dont need to get into too much detail about L1
- L2 Regularization in Action

    - Use L2 Regualrization to fix the overfitting problem we had before
    - Challenge the students to search for the ideal `alpha` (Regularization Parameter) value

- Observations'
    - Ask students to fill out the observations table

### The ML Process
- Paraphrase from the text
- be sure to make it clear that ML is an **iterative** process
- Explain what hyperparameters are
    - The parameters of a model are all the weights $w$ and biases $b$. These are **learnable parameters** that you cannot directly tune
    - Hyperparameters are parameters that you can tune. For example, learning rate, regularization parameter, etc.
- Iterate, Iterate, Iterate!
    - 5% of the time is spent on coding, the other 95% is spent on training, finding the correct hyperparameters, tuning, testing.
    - Talk about the "Idea, Code, Experiment" picture by Andrew Ng.

- Suggestions on training your ML Model
    - **These suggestions are important!** Go over these suggestions with students at a reasonable pace so that they fully understand.


## Cat-astrophe
- This notebook revolves around a whacky story about the city of Dogtopia. Feel free to narrate the text.
- There are only two cells the students need to edit: The cell containing the L2 Hyperparameter, and the cell under **Testing Your Model**.
- **Recommended to be run solely in Colab**
- Briefly commentate on each cell, but don't go too in-depth

### Their Model
- Do not explain anything about the model itself; Convolutional Neural Networks are outside the scope of this course
- Train their model, and commentate on the loss and accuracy metrics
- Their model should overfit like crazy.

### Your Take
##### THE GAME OF L2 EVOLUTION
**=== RULES ===**
- Do NOT change learning rate
- Do NOT change number of EPOCHS
- have each student choose a random L2 regularization value they think is gonna work the best
- Have them train their models
- Whoever has the highest cross validation accuracy gets to "pass down" their L2 value
- Repeat! Have students choose L2 values close to the previous winning entry
- Try to converge on the optimal L2 value

### Testing Your Model
- Write code for displaying testing inputs and predictions