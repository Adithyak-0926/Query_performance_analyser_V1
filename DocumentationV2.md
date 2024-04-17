**Metric definitions**  
**Common Metrics**  
|Metric Name | Definition |  
|------------|------------|
|QueryId | This is the unique identifier for each executed query. No two queries will have the same queryId ever.|  
|Operator|Operator is the metric for any particular operation run inside the query. 
Example: Joinoperator is the operator for join operations executed within the query. Other examples are  SinkOperator, LogicalValueOperator, PartInMemoryAggregationOperator, TableScanOperator|  
|Operator_id|Identifier for each operator within the execution plan.|  
|planType|In SQL, an execution plan is a detailed roadmap that the database engine creates to execute a query efficiently. This plan includes information about the order in which tables will be accessed, the types of joins that will be performed, the indexes that will be utilized, and any other operations such as sorting or aggregating data, etc.|  
|Thread_duration_max|Thread_duration_max represents the maximum time taken amongst all threads to perform a specific operation like joining.|  
|Cost_percent_str|Refers to the cost percentage represented as string.|  
|inputRowsPerThread_0_50_75_90_100|This is an array of 5 values, which shows us how many number of rows are input per thread at different percentiles (with respect to how much data is read)|  
|row_count_in/out|nRefers to the number of rows entering or exiting the operator during an operation.|  
|Partitioned|Refers to whether the data was partitioned or not. The metric is in the form of a boolean.|  
|Parallelism|Parallelism means performing certain operations  parallelly on different threads at a time generally on different files of data to increase the speed/decrease the time of execution of the particular operation. Parallelism as a metric represents the degree to which it is being employed in executing a specific operator. |  
|Max_memory|Refers to the maximum memory used while performing a certain operation|  
|num_chunks_in/out|number of chunks taken in or generated out for an operator. In the context of SQL query execution, "chunks" typically refer to portions or fragments of data being processed or manipulated by the database engine. |  
|thread_Duration_0_50_75_90_100|Array of 5 values, provides information about thread execution duration at different percentiles (with respect to how much data is read)|  
|Cost_percent|Refers to the percentage of total time spent on performing a certain operation.|  
|join_type|The type of join that particular operator is. Ex: INNER, OUTER, EQUI, etc.|  

**Operator Specific Metrics**  
|Metric Name | Definition |  
|------------|------------|
|Total_query_time | Refers to the total time taken for query execution.|  
|Was_distributed|Boolean indicating if the query execution/data in database was distributed across multiple systems.|  
|table_name|table_name represents the name of the table on which the particular operation is executed. |  
|Stats time|This is the total time recorded for gathering all the statistics such as: planType|  
|totalClientQueryTime|It refers to the total amount of time taken to execute a query from the perspective of the client application. It includes the time spent by the client application sending the query to the database server, waiting for the database server to process the query, and receiving the results back from the server.|  
|TotalParquetReadingTime|It refers to the total time taken to read the parquet files in the database. In case of parallelism, it refers to the collective time taken to read the parquet files across all threads.|  
|PartitionPruningDurationMs|Pruning means removing or skipping unwanted partitions or files and only read the necessary data.
PartitionPruningDurationMs refers to the time taken (in miliseconds) for partition pruning for the operation.|  
|executionQueueingTime|Refers to the total time taken to queue the tasks in the desired order of execution.|  
|ReadIOTime|Refers to the time taken for input/output (I/O) operations during the reading of data from storage devices by the database system. (IO - network(S3))|  
|totalBytes|Refers to the total size of the data in the output.|  
|fileListingDurationInMs|Refers to the time spent on listing which files to be read after pruning during a particular TableScanOperation.|  
|Query_max_memory|maximum of all max_memorys' used for query for all operators. max_memory refers to the maximum memory used by a certain operator.|  
|Queue_blocked_time|Refers to the time for which the queue is blocked with the assigned tasks for execution in % and ms.|  
|TotalTablescanFilteringTime|Refers to the total time taken for table scan filtering. If parallelism is there, it is the collective table scan filtering time of all threads. Table scan filtering refers to the process of filtering rows during a table scan operation in a database query.|  
|Parsing_time|Refers to the total time spent on parsing, planning, and giving out the plan. |  
|SubmitTasksDurationMs|Refers to the total time taken for coordinating executor to submit/assign tasks to other executors|  
|totalOpenDuration|Refers to the total time taken to initialise all of the operators in the operator tree. An "operator tree" is a hierarchical representation of the logical and physical plan obtained from the planner.|  
|skipped_pages/skipped_row_groups|Refers to the total count of the pages/rows_groups that are skipped while reading the parquet files.|  
|File_name_max_read_time|Name of the file which took maximum time for reading in the entire execution.|  
|Parquet_task_cost_percent|Refers tot the cost percentage associated specifically with parquet reading.|  
|Seek_io_time/count|Refers to the time spent on IO seeks and number of IO seeks.Seek means moving the header/pointer to another chunck's top to skip the reading of unnecessary chunk. A pointer points to the top of a chunk or a column of rowgroup in parquet.|  
|read_io_bytes/count|Refers to the number of IO bytes read and the read_IO count.|  
|taskInitializationDuration|Refers to the maximum time taken to start the task on all threads in case of parallelism.|  
|Tasks|Refers to the total number of tasks to be executed during a particular TableScanOperation.|  
|Partitions|Partitioning refers to the process of dissecting large datasets to ignore unnecessary reading of some files. "Partitions" metric defines the number of partitions used in a particular TableScanOperation.|  
|Files|Refers to the number of files processed after pruning during a particular TableScanOperation.|  
|cacheHIts|We store data in cache in order to ignore unnecessary loading of same data from s3. if we run the query which uses the same data, it checks in cache for reading, if it is matched it is considered as cacheHit.|  
|Stream_close_time|Refers to the close time of stream (that is pipeline) from s3 to engine|  
|Page_filter_creation_time_max|Refers to the maximum time taken for filtering all pages|  
|Open_time_percent|Refers to the percentage of time spent in open state|  
|readColumnChunkStreamsDuration|time taken to read column chunks|  
|fileReaderOpenTime|total duration for which file reading is happen|  
|task_rowsInCount_0_50_75_90_100|count of rows that are processed per task across various percentiles|  
|Total_row_groups|total number of row_groups are involved|  
|pageReadFromChunkDuration|Refers to the total time taken to read the column chunks.|  
|Read_io_time_percent|Refers to the percentage of time spent in reading from IO(network) from total_query_time|  
|totalRowGroupReadTimeMillis|Refers to the total time taken to read all involved row_groups in milliseconds|  
|totalRowGroupFilteringTime|Refers to the total time taken to filter all involved row_groups|  
|filtering_cost_percent|Refers to the percentage time spent on filtering during a particular TableScanOperation|  
|InMemoryAggregationOperator_max|In case of parallelism, this metric refers to the maximum time taken among all threads to do aggregation operation individually |  
|SinkOperator_max|In case of parallelism, this metric refers to the maximum time taken among all threads to do sink operation individually |  
|TableScanOperator_max|In case of parallelism, this metric refers to the maximum time taken among all threads to do Table scan operation individually |  
|PartInMemoryAggregationOperator_max|In case of parallelism, this metric refers to the maximum time taken among all threads to do aggregation operation individually |  
|totalPartitionBeforePruning|Refers to the total number of partitions that are available before pruning during the particular TableScanOperation|  
|totalFilesBeforePruning|Refers to the total number of files that are available before pruning during the particular TableScanOperation|  
|query_alias|Alias name for query which is unique for a query|