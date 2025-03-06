DATA CHALLENGE
-

The goal for us is to get an understanding of how you approach and think about
problems, and how you work with data. While the deliverable includes a machine learning model, the 
evaluation is much deeper than that -- we care about how you're getting to that final state, your logic, 
and your code.  

This repository has 2 years worth of Lending Club loan files stored in the data/ directory (gzipped csvs). 
These files are quarterly, and have data on loans that Lending Club has 
issued (date, amount, term, interest rate), metadata about the customer who took them 
out (such as employment, annual income, FICO), and the loan status. There is a data dictionary stored
in the docs/ directory.

Model Usage: Your goal is to inform investors on the best loans to invest in. This means: I am going to Lending Club and 
ready to invest $100. There is a list of loans on their site (which have not
yet been funded) that I get to choose from, and I want to know which ones are the best to invest in.
Keep that goal in mind as you build your feature set and final solution.

Have fun!
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*



So I have succefully created a based on given information.
Future scope: Docker implementations so you don't have to follow below steps

but for this time, you have to follow below steps to run above project in your local machine

First create conda environment 

conda create - p venv python==3.12 -y

Once its created, you have to activate that environment

conda activate ./venv

________________________________________________________________________________________________________

I have already added artifacts folders to this git repository (not recomendade).
so we have 2 routes to run this project.

1. Create training pipelines, and prediction pipelines

please follow run below line in your cmd 

>>python main.py

once its done, you can also find some interesting information about training and predictions pipelines into logs folder

then run below script

>>fastapi run app.py

NOTE: At the moment, I am just showing those predictions but in future with the help of javascript and HTML, 
we can update investments column based on investment.

***Feature Selections***
As we all know, This data set has so many feature.
- I have uploaded pandas profiling report to artifacts which gives more details about correlation of the features.
- I have also added one more sheet to excel fiel in docs folders which has more details about which features I have used and why I have used.

- for the Model evaluation, I have used F1 score. We can discuss more about this metric.


**While running ipynb files, we have to make explicitly, "extracted files" folder inside reserach folder and "artifacts" folder under main branch.
