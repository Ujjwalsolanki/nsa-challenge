#nsa
from pathlib import Path
from fastapi import FastAPI, Form, UploadFile, File, Request

import uvicorn

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os

import pandas as pd

app = FastAPI()
# for static files
# app.mount("/uploads",StaticFiles(directory="../uploads"),name="static")

templates = Jinja2Templates(directory='templates')

def get_data():

    artifacts_folder_path:Path = Path('artifacts/')

    use_columns = ['id', 'predictions', 'funded_amnt', 'term', 'int_rate', 'investments']
    df = pd.read_csv(os.path.join(artifacts_folder_path,'sample_loans_with_predictions.csv'), usecols=use_columns)
    df_sorted = df.sort_values(by='predictions',ascending=False)


    ##implement a logic for investments done by 10%, 20% then upto 80% or 100%
    df_sorted = df_sorted[df_sorted['investments']<df_sorted['funded_amnt']]
    print(df_sorted.head())
    return df_sorted


@app.get('/', response_class=HTMLResponse)
def main(request: Request):

    data = get_data()

    return templates.TemplateResponse('index.html', {'request': request, 'data': data.to_html()})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
