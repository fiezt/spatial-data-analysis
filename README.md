# Publications
This work is associated with the following publications:


#### Tanner Fiez, Lillian Ratliff. "Data-Driven Spatio-Temporal Analysis of Curbside Parking Demand," submitted to IEEE Transactions on Intelligent Transportation Systems (ITS) 2018.

#### Chase Dowling, Tanner Fiez, Lillian Ratliff, Baosen Zhang. "Fusing data streams to model urban congestion caused by drivers cruising for curbside parking," submitted to Nature Communications 2018.

#### Tanner Fiez, Lillian Ratliff, Chase Dowling, Baosen Zhang. "Data-Driven Spatio-Temporal Modeling of Parking Demand," submitted to American Control Conference (ACC) 2018.

# Instructions
Code is contained in the code folder of the repository. The file structure is
set up such that if you want to run the analysis you should be able to run one
file and produce the analysis used for each of the papers.

For the ITS paper:

     python its_belltown.py
     python its_belltown_denny.py
     python its_belltown_commcore.py

For the Nature paper:

     python nature.py

For the ACC paper:

     python acc.py

Note that the results depend on initializations, the number of restarts, etc,
due to the nonconvex objective of GMM so figures and results may not be
exactly the same as in the papers when running the analysis, but regardless
the they will be close and the conclusions the same.

If there are any problems, questions, or bugs, please email me at fiezt@uw.edu.

# Dependencies
    pandas==0.20.3
    seaborn==0.8.1
    scipy==0.19.1
    gmplot==1.1.1
    matplotlib==2.0.2
    numpy==1.13.1
    scikit_learn==0.19.0
