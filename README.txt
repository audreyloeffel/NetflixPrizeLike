# Netflix-Prize like Recommender System
===========================================================================

This is our implementation of the Recommender Systems Project.
This repository contains all the code needed to reproduce our best score,
but also to conduct much more experiments, like running any other
machine learning method contained in one of the frameworks we implemented.

A computer having 4GB of RAM should suffice to run this code without any
trouble.

You will find your submission csv file in the data folder when the
algorithm finishes.

In order to run this project, you need to:

  1. Download the right requirements for our best solution:
     - Numpy
     - Scipy
     - Sklearn (Careful with this one on ubuntu, sometimes you will need
       to use ap-get to install it.)

     [Optional] If you want to run the other librairies that we used:
      - Install the Spark framework with pyspark
        https://spark.apache.org/mllib/
      - Install the surprise library
        pip3.5 install surprise

  2. Put in the data directory the files needed for training and submission
     (data_train.csv and sampleSubmission.csv)

  3. Run the following command: "python python/run.py" at the root of the
     project. (Assuming you are running python3.5 as default, otherwise,
     run "python3.5 python/run.py" instead)

  4. The output predictions will be in the "data/submission.csv" file.

===========================================================================
