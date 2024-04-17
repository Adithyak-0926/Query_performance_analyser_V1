from flask import Flask, request, jsonify
import os
import json
from Backend_LLM import DataProcessor
import pandas as pd

app = Flask(__name__)

UPLOAD_FOLDER = 'Actual_JSONs_Holder'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/EAO_chat', methods=['POST'])
def process_data():
    try:
        question = request.form.get('question')
        mode = request.form.get('mode', 'Single_Json_Mode')  
        json_string = request.form.get('json_string')  
        query_String = request.form.get('query_text')
        # Handling file uploads
        uploaded_files = request.files.getlist('file')  # Supports multiple file uploads

        query_str = preprocess_query_string(query_String)
        for file in uploaded_files:
            if file:
                # filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

        # need to develop this method yet.
        if json_string:
            try:
                # json_string = preprocess_queryLogstr(json_string)
                json_string = preprocess_json(mode,json_string)
                json_data = json.loads(json_string)
                queryId = json_data.get("queryId")
                json_filename = f"{queryId}.json"  
                with open(os.path.join(app.config['UPLOAD_FOLDER'], json_filename), 'w') as json_file:
                    json.dump(json_data, json_file)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON string"}), 400
        
        if mode == 'CSV_Tool_Mode' :
            csv_file = request.files.get('file')

            if csv_file and csv_file.filename.endswith('.csv'):
                query_aliases = process_csv(mode,csv_file.filename,app.config['UPLOAD_FOLDER'])
                processor = DataProcessor(mode,query_aliases)
            else:
                return jsonify({"error": "Invalid file type. Only CSV files are allowed."}), 400
            
        else: 
            processor = DataProcessor(mode,query_str)

        response = processor.get_response(question)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def preprocess_json( mode,json_str, alias_name=None):
        if json_str is None:
            return None
        json_str = json_str.replace("'", '"')
        json_str = json_str.replace('False', '"false"')
        json_str = json_str.replace('True','"true"')
        if mode == 'CSV_Tool_Mode':
            json_str = json_str[:-1] + f', "query_alias": "{alias_name}"' + json_str[-1]
        return json_str 
def preprocess_query_string(query_string):
    query_string = query_string.replace("'", '"')
    return query_string
def preprocess_queryLogstr(json_str):
    return json_str

def process_csv(mode,name,input_folder):
         csv_path = os.path.join(input_folder,name)
         df = pd.read_csv(csv_path)    
         json_stats_column = 'json_stats'  
         query_alias_column = 'query_alias'

         # Preprocess JSON strings
         df[json_stats_column] = df.apply(lambda row: preprocess_json(mode,row[json_stats_column], row[query_alias_column]), axis=1)
         for _, row in df.iterrows():
             alias_value = row['query_alias']
             json_str = row[json_stats_column]
             output_file_path = os.path.join(input_folder, f"{alias_value}.json")
             with open(output_file_path, 'w') as outfile:
                 outfile.write(json_str)

         query_aliases = df[query_alias_column].to_list()
         if 'json_stats' not in df.columns or 'query_alias' not in df.columns:
             raise ValueError("CSV file must contain 'json_stats' and 'query_alias' columns.")
         os.remove(csv_path)

         return query_aliases

if __name__ == '__main__':
    app.run(debug=True)