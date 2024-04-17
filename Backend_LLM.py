import json
import os
import pandas as pd
import ast
import time
from dotenv import load_dotenv

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import JSONLoader, DirectoryLoader

load_dotenv()
openai_access_key = os.getenv('openai_access_key')
claude_api_access_key = os.getenv('claude_api_access_key')

class DataProcessor:
    def __init__(self, mode, query, query_aliases = None):
        self.mode = mode
        self.query = query
        self.input_folder = 'Actual_JSONs_Holder'
        self.output_folder = 'Temp_Json_Folder'
        self.query_aliases = query_aliases if query_aliases is not None else []
        self.document = self.load_document('DocumentationV2.md')
        self.GPT4 = ChatOpenAI(api_key=openai_access_key, model="gpt-4-0125-preview", temperature=0, timeout=600)    
        self.Claude_haiku = ChatAnthropic(anthropic_api_key=claude_api_access_key,model_name="claude-3-haiku-20240307",max_tokens_to_sample=250)
        self.Claude_sonet = ChatAnthropic(anthropic_api_key=claude_api_access_key,model_name="claude-3-sonnet-20240229")
        self.Claude_opus = ChatAnthropic(anthropic_api_key=claude_api_access_key,model_name="claude-3-opus-20240229")


    def required_file_extractor(self, question_q,query_aliases):
     few_shot_examples = [
     {"question": "give me top three queries with most readIOTime?", "query_aliases_list" : ['bootstrap-1', 'bootstrap-2', 'bootstrap-3', 'TPCDS-1', 'TPCDS-2', 'TPCDS-3'], "response": "bootstrap-1, bootstrap-2, bootstrap-3, TPCDS-1, TPCDS-2, TPCDS-3"},
     {"question": "For TPCDS-2, how many files are pruned?", "query_aliases_list" : ['bootstrap-1', 'bootstrap-2', 'bootstrap-3', 'TPCDS-1', 'TPCDS-2', 'TPCDS-3'], "response": "TPCDS-2"},
     {"question": "For TPCDS-5, which are the most expensive operators", "query_aliases_list" : ['bootstrap-1', 'bootstrap-2', 'bootstrap-3', 'TPCDS-1', 'TPCDS-2', 'TPCDS-3'], "response": ""},
     {"question": "What are the most expensive operations for TPCDS-1 and BOOTSTRAP-1?", "query_aliases_list" : ['bootstrap-1', 'bootstrap-2', 'bootstrap-3', 'TPCDS-1', 'TPCDS-2', 'TPCDS-3'], "response": "TPCDS-1,bootstrap-1"},
     {"question": "In comparision to TPCDS-2, which queries are more performant?", "query_aliases_list" : ['bootstrap-1', 'bootstrap-2', 'bootstrap-3', 'TPCDS-1', 'TPCDS-2', 'TPCDS-3'], "response": "bootstrap-1, bootstrap-2, bootstrap-3, TPCDS-1, TPCDS-2, TPCDS-3"}
     ]

     file_extractor_template = PromptTemplate(
          input_variables=['question', 'query_aliases_list'],
          template="You are a file name extractor. Looking at the Question : {question} and the full names list : {query_aliases_list},  give the list of aliases which would be required to answer the question  i.e; the response: {response}"
     )

     few_shot_prompt_template = FewShotPromptTemplate(
      example_prompt=file_extractor_template,
      examples=few_shot_examples,
      prefix="Instructions:\n1) return full query alias list if no specific query or alias is mentioned in question\n2) If aliases is mentioned in question, and question is particularly asked about them, return only those aliases\n3) If alias mentioned in question is not present in our full aliases list, return empty response\nNote: only return list of wanted names and do not return a single token extra, like if file names you want to respnd are X-1, Y1 then return ['X-1','Y1'].",
      suffix="Question: {question}\nfull aliases list: {query_aliases_list}\nResponse:",
      input_variables=file_extractor_template.input_variables,
      example_separator="\n\n"
     )
     file_name_extractor = LLMChain(llm=self.GPT4, prompt=few_shot_prompt_template,verbose=False)
     with get_openai_callback() as cb:
        file_name_list = file_name_extractor({'question' : question_q, 'query_aliases_list' : query_aliases})
        file_name_list = file_name_list['text']
        print(f"Total Tokens for query_alias listing: {cb.total_tokens}")
        print(f"Prompt Tokens for query_alias listing: {cb.prompt_tokens}")
        print(f"Completion Tokens for query_alias listing: {cb.completion_tokens}")
        print(f"Total Cost (USD) for query_alias listing: ${cb.total_cost}")
     # file_name_list = [file_name.strip() for file_name in file_name_list if file_name.strip()]
     print(file_name_list)
     return file_name_list

    def extract_metrics(self, question):
     few_shot_examples = [
         {"question":"Point the reasons for the poor performance of the query?", "response":"[ 'Thread_duration_max', 'inputRowsPerThread_0_50_75_90_100', 'thread_Duration_0_50_75_90_100', 'Cost_percent', 'totalClientQueryTime', 'TotalParquetReadingTime', 'PartitionPruningDurationMs', 'executionQueueingTime', 'ReadIOTime', 'fileListingDurationInMs', 'Parsing_time', 'SubmitTasksDurationMs', 'totalOpenDuration', 'Seek_io_time/count', 'taskInitializationDuration', 'readColumnChunkStreamsDuration', 'fileReaderOpenTime', 'task_rowsInCount_0_50_75_90_100', 'pageReadFromChunkDuration', 'Read_io_time_percent', 'totalRowGroupReadTimeMillis', 'totalRowGroupFilteringTime', 'filtering_cost_percent', 'InMemoryAggregationOperator_max', 'SinkOperator_max', 'TableScanOperator_max', 'PartInMemoryAggregationOperator_max', 'totalPartitionBeforePruning', 'totalFilesBeforePruning', 'Page_filter_creation_time_max', 'Open_time_percent', 'Stream_close_time','Max_memory', 'Query_max_memory','Files','Tasks','Partitions','skipped_pages','skipped_row_groups','row_count_in','row_count_out','table_name','num_chunks_in','num_chunks_out','join_type']"},
         {"question":"What are the most expensive operators?","response":"['Operator', 'query_alias', 'cost_percent', 'cost_percent_str', 'Thread_duration_max', 'query_execution_time', 'Parallelism', 'Max_memory', 'Tasks', 'Files', 'inputRowsPerThread_0_50_75_90_100', 'row_count_in','thread_Duration_0_50_75_90_100', 'table_name', 'TotalParquetReadingTime' ]"},
         {"question":"Compare the both files whether pruning happend.", "response":"['query_alias','operator','CacheHits',''Files','Tasks','Num_chunks_in','Num_chunks_out','Partitions','row_count_in','Total_row_groups','TotalFIlesBeforePruning','TotalPartitionBeforePruning','table_name']"}
     ]
     metric_extractor_template = PromptTemplate(
         input_variables=['question','document'],
         template="You are a metric extractor and can extract the necessary metrics from the user question : {question} based on the context provided which is a documentation containing definitions of all metrics i.e; the response: {response}"
     )
    
     few_shot_prompt_template = FewShotPromptTemplate(
      example_prompt=metric_extractor_template,
      examples=few_shot_examples,
      prefix="This is the context of documentation that contains all the definitions of metrics. \ncontext : \n{document}\n . Instructions: \n1) You need to just return the list of names of metrics whose values will be needed to answer the question.\n2) Always return 'Operator','query_alias','operator_id' and do not return 'queryId' into the list regardless of the need.\n3) If the question is specifically related to strict metrics which will be occured only once in our data like 'ReadIOTime','Total_query_time' etc then strictly do not include 'Operator' in the list. \n4)  If you identify metrics that might not be directly requested but could significantly contribute to a comprehensive analysis, include them in the list. This approach ensures a broader perspective and may reveal insights not immediately apparent from the question.\n 5) Factors affecting pruning are 'CacheHits',''Files','Tasks','Num_chunks_in','Num_chunks_out','Partitions','row_count_in','Total_row_groups','TotalFIlesBeforePruning','TotalPartitionBeforePruning','table_name'.\nNote: only return the list of operators and do not write a single token extra apart from names",
      suffix="Question: {question}\nResponse:",
      input_variables=metric_extractor_template.input_variables,
      example_separator="\n\n"
     )
     metric_extractor = LLMChain(llm=self.GPT4, prompt=few_shot_prompt_template, verbose=False)

     # Extract metrics using the LLMChain
     with get_openai_callback() as ab:
        res = metric_extractor({'question': question, 'document': self.document})
        print(res)
        metrics_output = res['text'].split('\n')  # Split the text into a list of metrics
        print(f"Total Tokens for extraction: {ab.total_tokens}")
        print(f"Prompt Tokens for extraction: {ab.prompt_tokens}")
        print(f"Completion Tokens for extraction: {ab.completion_tokens}")
        print(f"Total Cost (USD) for extraction: ${ab.total_cost}")
    
     if metrics_output and any(metrics_output):
        try:
            # Attempt to evaluate the string as a literal expression
            print(f"generated metric list is in string format : {metrics_output}")
            metrics_list = ast.literal_eval(metrics_output[0])
        except (ValueError, SyntaxError):
            # If the conversion fails, it means metrics_output was already a list
            metrics_list = metrics_output
     else:
        metrics_list = []  
     # Remove any empty strings or unnecessary whitespace
     metrics_list = [metric.strip() for metric in metrics_list if metric.strip()]
    
     print(metrics_list)
     return metrics_list

    def clean_json(self, json_data, metric_list):
     if isinstance(json_data, dict):
        keys_to_delete = []
        for key in list(json_data.keys()):
            value = json_data[key]
            if isinstance(value, list):
                 if all(isinstance(item, (int, float)) for item in value) and key.lower() not in [metric.lower() for metric in metric_list]:
                     keys_to_delete.append(key)
                 self.clean_json(value, metric_list)
            else:
                if key.lower() not in [metric.lower() for metric in metric_list]:
                        keys_to_delete.append(key)
        for key in keys_to_delete:
            del json_data[key]
     elif isinstance(json_data, list):
        for item in json_data:
                self.clean_json(item, metric_list)
     return json_data
    
    def process_json_files(self,input_folder, output_folder,question, filenames_list):
     if (question == "Can you give me pruning analysis.") :
         metrics_list = ['query_alias','cost_percent','operator','operator_id','Files','Tasks','Num_chunks_in','Num_chunks_out','Partitions','Total_row_groups','TotalFIlesBeforePruning','TotalPartitionBeforePruning','table_name','skipped_row_groups']
     elif(question == "point out the main reasons for the queries poor performance.") :
         metrics_list = ['query_alias', 'operator', 'operator_id', 'Thread_duration_max', 'inputRowsPerThread_0_50_75_90_100', 'thread_Duration_0_50_75_90_100', 'cost_percent', 'totalClientQueryTime', 'TotalParquetReadingTime', 'PartitionPruningDurationMs', 'executionQueueingTime', 'ReadIOTime', 'fileListingDurationInMs', 'Parsing_time', 'SubmitTasksDurationMs', 'totalOpenDuration', 'Seek_io_time/count', 'taskInitializationDuration', 'readColumnChunkStreamsDuration', 'fileReaderOpenTime', 'task_rowsInCount_0_50_75_90_100', 'pageReadFromChunkDuration', 'Read_io_time_percent', 'totalRowGroupReadTimeMillis', 'totalRowGroupFilteringTime', 'filtering_cost_percent', 'InMemoryAggregationOperator_max', 'SinkOperator_max', 'TableScanOperator_max', 'PartInMemoryAggregationOperator_max', 'totalPartitionBeforePruning', 'totalFilesBeforePruning', 'Page_filter_creation_time_max', 'Open_time_percent', 'Stream_close_time','Max_memory', 'Query_max_memory','Files','Tasks','Partitions','skipped_pages','skipped_row_groups','row_count_in','row_count_out','table_name','num_chunks_in','num_chunks_out','join_type']
     else:
         metrics_list = self.extract_metrics(question)
     # Iterate over each JSON file in the required files list
     for filename in filenames_list:
        input_path = os.path.join(input_folder, f"{filename}.json")
        output_path = os.path.join(output_folder, f"{filename}.json")
        try:
            # Load the JSON data
            with open(input_path, 'r') as f:
                json_data = json.load(f)
            query_Id = json_data['queryId']

            # Perform cleanup process 
            modified_json_data = self.clean_json(json_data, metrics_list)
            modified_json_data['queryId'] = query_Id

                # Save the modified JSON data to a separate file
            output_file = os.path.splitext(output_path)[0] + f"_{question.replace(' ', '_')}.json"
            with open(output_file, 'w') as f:
                json.dump(modified_json_data, f, indent=4)
        except FileNotFoundError:
                print(f"File '{filename}.json' is not present in the data.")

    def clean_folder(self,folder):
     # Iterate over each file in the output folder
     for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        # Check if the file is a regular file (not a directory)
        if os.path.isfile(file_path):
            # Remove the file
            os.remove(file_path)            
        
    def load_document(self, filepath):
        # Load documentation or any text file
        with open(filepath, 'r') as file:
            return file.read()

        
    def get_response(self, question):
     start_time = time.time()
     self.clean_folder(self.output_folder)
     req_files_list = []
     if self.mode == 'CSV_Tool_Mode':
        req_files_list = self.required_file_extractor(question,self.query_aliases)
        if not req_files_list:
            raise FileNotFoundError("Please kindly check your question and ask again with proper aliases if needed")
     else: 
        req_files_list = [os.path.splitext(filename)[0] for filename in os.listdir(self.input_folder) if filename.endswith('.json')]    
     if self.mode == 'Multi_Json_Mode' and not req_files_list:
         raise FileNotFoundError("PLease check the folder path you provided and make sure that it contains EAO jsons")
     if req_files_list and any(req_files_list):
        try : 
            req_files_list = ast.literal_eval(req_files_list)
        except : 
            pass
     self.process_json_files(self.input_folder,self.output_folder,question,req_files_list)
    #  loader = DirectoryLoader(self.output_folder, glob="**/*.json", show_progress=True, loader_cls=JSONLoader, loader_kwargs={'jq_schema': '.', 'text_content': False})
    #  modified_data = loader.load()
     for filename in req_files_list:
        output_path = os.path.join(self.output_folder, f"{filename}_{question.replace(' ', '_')}.json")
        with open(output_path,"r") as j:
            modified_data = json.load(j)

     with open("extra_processed_metadata.json","r") as f :
         schema_n_metadata = json.load(f)
     with get_openai_callback() as cb:
        response = self.ultimate_response(question, modified_data, schema_n_metadata)
        response = response['text']
        # Print token usage details
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
     print(f"time taken for the question: {question} is {time.time()-start_time}")
     self.clean_folder(self.input_folder)   
     print(response)
     return response
    
    def ultimate_response(self,question,JSON_data,schema_n_metadata):
     memory = ConversationBufferWindowMemory(input_key='question',k=10)
     if (question == "Can you give me pruning analysis."):
         with open("prompts/pruning_analysis_V1.txt","r") as f:
             JSON_Analyser_template = f.read()
         pruningAndPartitioningReport = self.get_pruningAndPartitioningReport(JSON_data,schema_n_metadata)
         new_prompt_template = PromptTemplate.from_template(template=JSON_Analyser_template)
         QnA_teller = LLMChain(llm=self.GPT4,prompt= new_prompt_template,verbose= True,memory=memory)
         response = QnA_teller({'question': question, 'document': self.document, 'data': JSON_data, 'query_text' : self.query, 'pruningAndPartitioningReport' : pruningAndPartitioningReport})
     else: 
         with open("prompts/gemini_pro.txt","r") as f:
             JSON_Analyser_template = f.read()
         pruningAndPartitioningReport,distributionReport,badJoinReport = self.get_all_reports(JSON_data,schema_n_metadata)
         new_prompt_template = PromptTemplate.from_template(template=JSON_Analyser_template)
         QnA_teller = LLMChain(llm=self.GPT4,prompt= new_prompt_template,verbose= True,memory=memory)
         response = QnA_teller({'question': question, 'document': self.document, 'data': JSON_data, 'query_text' : self.query, 'pruningAndPartitioningReport' : pruningAndPartitioningReport, 'BadDistributionReport' : distributionReport, 'badJoinReport' : badJoinReport})
     return response
    
    def get_all_reports(self,query_stats_data,schema_n_metadata):
        analyzer = Pruning_Partitioning_distribution_badJoin_Analyzer(query_stats_data,schema_n_metadata)
        analyzer.perform_overall_analysis()
        pruning_report_message = analyzer.generate_pruning_report()
        partitioning_report_message = analyzer.generate_partitioning_report()
        combined_report_of_Pruning_n_partitioning = pruning_report_message + partitioning_report_message
        distribution_report_message = analyzer.generate_distribution_report_for_TS_operators()
        bad_join_report_message = analyzer.generate_bad_joins_report()
        return combined_report_of_Pruning_n_partitioning,distribution_report_message,bad_join_report_message
    def get_pruningAndPartitioningReport(self,query_stats_data,schema_n_metadata):
        analyzer = Pruning_Partitioning_distribution_badJoin_Analyzer(query_stats_data,schema_n_metadata)
        analyzer.perform_pruning_analysis()
        pruning_report_message = analyzer.generate_pruning_report()
        partitioning_report_message = analyzer.generate_partitioning_report()
        combined_report_of_Pruning_n_partitioning = pruning_report_message + partitioning_report_message
        return combined_report_of_Pruning_n_partitioning

class Pruning_Partitioning_distribution_badJoin_Analyzer:
    def __init__(self, query_stats_data, partitioning_metadata):
        self.query_stats_data = query_stats_data
        self.partitioning_metadata = partitioning_metadata
        self.pruning_results = []
        self.unpruned_tables = []
        self.involved_tables = set()
        self.focusTables = []
        self.unevenDistributedTSOperators = []
        self.bad_join_orders = []
        self.exploded_joins = []

    def analyze_table(self, node):        
        files_pruned = node["totalFilesBeforePruning"] - node["files"]
        partitions_pruned = node["totalPartitionBeforePruning"] - node["partitions"]
        table_name = node["table_name"]
        operator_id = node["operator_id"]
        skipped_row_groups = node["skipped_row_groups"]
        cost_percent = node["cost_percent"]
        if cost_percent > 15 :
            self.focusTables.append((table_name,operator_id,cost_percent))            
        self.pruning_results.append({
                "operator_id" : operator_id,
                "Table": table_name,
                "Files_Pruned": files_pruned,
                "Partitions_Pruned": partitions_pruned,
                "skipped_row_groups" : skipped_row_groups,
                "cost_percent" : cost_percent,
            })
        if files_pruned == 0 and partitions_pruned == 0 and skipped_row_groups == 0: 
            self.unpruned_tables.append((table_name, operator_id, cost_percent))

    def analyse_tableScan_forDistribution(self,node):
        operator_id = node["operator_id"]
        inputRowsPerThread_0_50_75_90_100_str = node["inputRowsPerThread_0_50_75_90_100"]
        inputRowsPerThread_0_50_75_90_100_str = inputRowsPerThread_0_50_75_90_100_str.strip('[]')
        inputRowsPerThread_0_50_75_90_100 = [int(x) for x in inputRowsPerThread_0_50_75_90_100_str.split(',')]  
        value_at_50 = inputRowsPerThread_0_50_75_90_100[1]
        value_at_90 = inputRowsPerThread_0_50_75_90_100[3]
        if value_at_50 > 1000000 and value_at_90 > 1000000 and value_at_90 > 1.5*value_at_50:
            self.unevenDistributedTSOperators.append((operator_id,value_at_50,value_at_90))

    def traverse_json_for_pruning_analysis(self, node):
        if isinstance(node, dict):
            if "table_name" in node:
                self.analyze_table(node)
            for key, value in node.items():
                if key == "childOperators":
                    for child_node in value:
                        self.traverse_json_for_pruning_analysis(child_node)

    def traverse_json(self, node):
        if isinstance(node, dict):
            if "table_name" in node:
                self.analyze_table(node)
            if node["operator"] == "TableScanOperator" : 
                self.analyse_tableScan_forDistribution(node)
            if node["operator"] == "JoinOperator" :
                self.detect_bad_join_order_n_join_explosions(node)
            for key, value in node.items():
                if key == "childOperators":
                    for child_node in value:
                        self.traverse_json(child_node)

    def detect_bad_join_order_n_join_explosions(self,node):
        join_type = node["join_type"]
        operator_id = node["operator_id"]
        cost_percent = node["cost_percent"]
        row_count_in_main_join = node["row_count_in"]
        row_count_out_main_join = node["row_count_out"]
        if row_count_out_main_join > 2.5*row_count_in_main_join :
            self.exploded_joins.append((join_type,operator_id,row_count_in_main_join,row_count_out_main_join,cost_percent))
        row_count_out_ColumnarLookupTableBuildOperator_BUILDSIDE = 0
        row_count_out_otherChildOperator_PROBESIDE = 0
        for key,value in node.items():
            if key == "childOperators" :
                for child_node in value:
                    if child_node["operator"] == "ColumnarLookupTableBuildOperator" :
                        row_count_out_ColumnarLookupTableBuildOperator_BUILDSIDE = child_node["row_count_out"]
                    elif child_node["operator"] != "ColumnarLookupTableBuildOperator" :
                        row_count_out_otherChildOperator_PROBESIDE = child_node["row_count_out"]
        if row_count_out_ColumnarLookupTableBuildOperator_BUILDSIDE>1000000 and row_count_out_otherChildOperator_PROBESIDE > 1000000 and row_count_out_ColumnarLookupTableBuildOperator_BUILDSIDE > 5*row_count_out_otherChildOperator_PROBESIDE :
            self.bad_join_orders.append((join_type,operator_id,row_count_out_ColumnarLookupTableBuildOperator_BUILDSIDE,row_count_out_otherChildOperator_PROBESIDE,cost_percent))
   

    def perform_overall_analysis(self):
        self.traverse_json(self.query_stats_data)
        return self.pruning_results, self.unpruned_tables
    
    def perform_pruning_analysis(self):
        self.traverse_json_for_pruning_analysis(self.query_stats_data)
        return self.pruning_results, self.unpruned_tables

    def generate_pruning_report(self):
        pruning_report_message = "Pruning Report:\n"
        pruned_tables =[]
        if self.unpruned_tables:
            unpruned_tables_message = "Tables that are not pruned:\n"
            for table_name, operator_id,cost_percent in self.unpruned_tables:
                unpruned_tables_message += f"{table_name} at operator_ID: {operator_id} with cost percent of {cost_percent}% of total time.\n"
            pruning_report_message += unpruned_tables_message

        if self.pruning_results:
            pruned_tables_message = "Tables that are pruned:\n"
            for result in self.pruning_results:
                operator_Id = result["operator_id"]
                table_name = result["Table"]
                files_pruned = result["Files_Pruned"]
                partitions_pruned = result["Partitions_Pruned"]
                skipped_row_groups = result["skipped_row_groups"]
                cost_percent = result["cost_percent"]
                self.involved_tables.add(table_name)
                if files_pruned > 0 and partitions_pruned > 0:
                    pruned_tables_message += f"Table {table_name} at operator_id: {operator_Id}, having cost percent of {cost_percent}% of total time, had file pruning with {files_pruned} files pruned and partition pruning with {partitions_pruned} partitions pruned.\n"
                    pruned_tables.append(table_name)
                    continue
                elif files_pruned > 0:
                    pruned_tables_message += f"Table {table_name} at operator_id: {operator_Id}, having cost percent of {cost_percent}% of total time, had file pruning with {files_pruned} files pruned.\n"
                    pruned_tables.append(table_name)
                elif partitions_pruned > 0:
                    pruned_tables_message += f"Table {table_name} at operator_id: {operator_Id}, having cost percent of {cost_percent}% of total time, had partition pruning with {partitions_pruned} partitions pruned.\n"
                    pruned_tables.append(table_name)
                elif skipped_row_groups > 0:
                    pruned_tables_message += f"Table {table_name} at operator_id: {operator_Id}, having cost percent of {cost_percent}% of total time, had {skipped_row_groups} skipped row groups.\n"
                    pruned_tables.append(table_name)
            if not pruned_tables:
                pruned_tables_message += "No tables involved in the query are pruned.\n"
            pruning_report_message += pruned_tables_message
        if self.focusTables :
            focusTables_message = "Focusable Tables:\n"
            for table_name,operator_id,cost_percent_p in self.focusTables:
                focusTables_message += f"Table {table_name} at operator ID: {operator_id} is with relatively high cost_percent of {cost_percent_p}%.\n"
            pruning_report_message += focusTables_message
            pruning_report_message += "\n"
        return pruning_report_message
        
    def generate_partitioning_report(self):
        partitioning_report_message = "Partitioning Report:\n"
        # Load the JSON data if it's a string, otherwise assume it's already a dict
        if isinstance(self.partitioning_metadata, str):
            data = json.loads(self.partitioning_metadata)
        else:
            data = self.partitioning_metadata

        # Check if data is a list of tables; if not, make it a list for uniform processing
        if not isinstance(data, list):
            data = [data]

        partitioned_tables_message = ""
        for table_metadata in data:
            actual_table_name = table_metadata['tableName'].split('.')[-1]
            if actual_table_name in self.involved_tables:
                # Check if the table is partitioned
                if table_metadata.get("isPartitioned", False):
                    partitioned_columns = []
                    # Check for partition column stats to find partitioned columns
                    if "partitionColumnStats" in table_metadata and table_metadata["partitionColumnStats"]:
                        for partition in table_metadata["partitionColumnStats"]:
                            # Accumulate the partition column names
                            partitioned_columns.append(partition['partitionColumnName'])
                    # Accumulate the report for the table
                        partitioned_tables_message += f"{actual_table_name} is partitioned on column {', '.join(partitioned_columns)}\n"
                    else:
                        partitioned_tables_message += f"{actual_table_name} is partitioned but no partition column information available.\n"

        if partitioned_tables_message:
            partitioning_report_message += partitioned_tables_message
        else:
            partitioning_report_message += "No partitioned tables found.\n"

        return partitioning_report_message
    
    def generate_distribution_report_for_TS_operators(self):
        distribution_report_message = "Distribution Report:\n"
        if self.unevenDistributedTSOperators :
            uneven_distributed_TS_message = "Table scan operators with uneven distribution of inputRowsPerThread :\n"
            for result in self.unevenDistributedTSOperators :
                operator_id = result[0]
                value_at_50 = result[1]
                value_at_90 = result[2]
                uneven_distributed_TS_message += f"TableScanOperator at operator_id:{operator_id} has uneven distribution as {value_at_90}(value at 90%) is a sudden high compared to {value_at_50}(value at 50%)\n"
            distribution_report_message += uneven_distributed_TS_message
        else:
            distribution_report_message += "No unevenly distributed table scan operators found.\n"
        distribution_report_message += "\n"
        return distribution_report_message
    
    def generate_bad_joins_report(self):
        join_report_message = "bad Joins report:\n"
        if self.bad_join_orders:
            bad_join_order_message = "Joins with bad_join ordering:\n"
            for result in self.bad_join_orders:
                join_type = result[0]
                operator_id = result[1]
                row_count_out_ColumnarLookupTableBuildOperator_BUILDSIDE = result[2]
                row_count_out_otherChildOperator_PROBESIDE = result[3]
                cost_percent = result[4]
                bad_join_order_message += f"Join operator of {join_type} type at operator_id: {operator_id, }with cost percent {cost_percent}% of total time, is having a bad join order with {row_count_out_ColumnarLookupTableBuildOperator_BUILDSIDE} row_count_out on BUILD SIDE which is way higher than {row_count_out_otherChildOperator_PROBESIDE} row_count_out on PROBESIDE\n"
            join_report_message += bad_join_order_message
        if not self.bad_join_orders :
            join_report_message += "no bad join ordering detected\n"
        if self.exploded_joins:
            exploded_joins_message = "Exploded Joins:\n"
            for result in self.exploded_joins:
                join_type = result[0]
                operator_id = result[1]
                row_count_in_mainJoin = result[2]
                row_count_out_mainJoin = result[3]
                cost_percent = result[4]
                exploded_joins_message += f"Join operator of {join_type} type at operator_id: {operator_id}, with cost percent {cost_percent}% of total time, is an exploded join with {row_count_out_mainJoin} rows out which is comparably high with {row_count_in_mainJoin} rows in.\n"
            join_report_message += exploded_joins_message
        if not self.exploded_joins :
            join_report_message += "no exploded joins reported.\n"
        return join_report_message
 
