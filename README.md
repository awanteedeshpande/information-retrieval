# Information Retrieval System

## Description
Solution for Information Retrieval project of Statistical Natural Language Processing class 2020. <br/>

The task is to develop and evaluate a two-stage information retrieval model that given a query returns the `n` most relevant documents and then ranks the sentences within the documents. </br>

The baseline is a tf-idf based document retriever. The results are then improved using the Okapi BM25 model. The third part extends this model to ranked sentences.

## Requirements
Python 3.5 or above

## Directory Structure
- src - main directory
    1. dataset - contains the corpus and query files <br/>
	-> generated - intermediate and final generated files
	(tf.json, idf.json, ranking.json, results.txt, plots)
	2. baseline.py - baseline model implementation
	3. bm25.py - bm25 ranking implementation
	4. bm25_sentence.py - sentence based bm25 model implementation
	5. run.py - main function to execute the code
- analysis.pdf - Analysis and obervations with explanation 
- How to run the code - Execute command <br/>
    -> python run.py

---