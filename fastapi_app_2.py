from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import json
import os
from typing import List, Optional
import shutil
from Backend_LLM_2 import DataProcessor
import pandas as pd
import uvicorn


app = FastAPI()

PORT = 8000

UPLOAD_FOLDER = 'Actual_JSONs_Holder'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_json(mode, json_str, alias_name=None):
    if json_str is None:
        return None
    json_str = json_str.replace("'", '"').replace('False', 'false').replace('True', 'true')
    if mode == 'CSV_Tool_Mode' and alias_name:
        json_str = json_str[:-1] + f', "query_alias": "{alias_name}"' + json_str[-1]
    return json_str

def preprocess_query_string(query_string):
    return query_string.replace("'", '"')

@app.post("/EAO")
async def process_data(
    question: str = Form(...), 
    json_string: Optional[str] = Form(None), 
    query_text: Optional[str] = Form(None),
    mode: str = Form('Single_Json_Mode'),  # Default mode set here
    files: List[UploadFile] = File(default=None)
):
    try:
        query_str = preprocess_query_string(query_text)
        json_string = preprocess_json(mode, json_string)
        try:
                json_data = json.loads(json_string)
                queryID = json_data.get('queryId')
                json_filename = f"{json_data.get('queryId')}.json"
                with open(os.path.join(UPLOAD_FOLDER, json_filename), 'w') as json_file:
                    json.dump(json_data, json_file)
        except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON string")

        if mode == 'CSV_Tool_Mode' and files:
            for file in files:
                if file and file.filename.endswith('.csv'):
                    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                    query_aliases = process_csv(mode, file.filename, UPLOAD_FOLDER)
                    processor = DataProcessor(mode, query_aliases)
                    response = processor.get_response(question)
                    os.remove(file_path)  # Clean up after processing
                    return JSONResponse(content={"response": response})
                else:
                    raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are allowed.")
        else:
            processor = DataProcessor(queryID, mode, query_str)
            response = processor.get_response(question)
            return JSONResponse(content={"response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_csv(mode, name, input_folder):
    csv_path = os.path.join(input_folder, name)
    df = pd.read_csv(csv_path)
    if 'json_stats' not in df.columns or 'query_alias' not in df.columns:
        raise ValueError("CSV file must contain 'json_stats' and 'query_alias' columns.")
    df['json_stats'] = df.apply(lambda row: preprocess_json(mode, row['json_stats'], row['query_alias']), axis=1)
    for _, row in df.iterrows():
        json_str = row['json_stats']
        alias_value = row['query_alias']
        output_file_path = os.path.join(input_folder, f"{alias_value}.json")
        with open(output_file_path, 'w') as outfile:
            outfile.write(json_str)
    query_aliases = df['query_alias'].tolist()
    os.remove(csv_path)  # Clean up CSV file after processing
    return query_aliases

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_app_2:app",
        host="127.0.0.1",
        port=PORT,
        # log_level=LOG_LEVEL,
        proxy_headers=True,
        workers=1,
    )