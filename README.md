# Sentiment Analysis
Sentiment Analysis for tweets

## Steps to run the pipeline

### Create virtual conda environment

<p> After creating virtual environment, install python version>=3.8</p>

'''
conda create -n sentiment

'''

### Install requirements

<p> Install all the required dependencies from requirements.txt file</p>

'''
pip install -r requirements.txt
'''

### Train model 
<ul>
<li> Create folder named data</li>
<li>Add training data in csv format to the folder</li>
<li> run training.py script</li>
</ul>

'''
python training.py
'''

### Start FastAPI server

'''
uvicorn app:app --reload
'''

<p> Hit the url with the text whose sentiment needs to be identified, the api returns a dictonary with the text and its predicted sentiment</p>

