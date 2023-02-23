# Databricks notebook source
# MAGIC %md This solution accelerator can also be found at https://github.com/databricks-industry-solutions/jsl-suicide-prediction.

# COMMAND ----------

# MAGIC %md
# MAGIC # Annotating Data & Training Custom NER model for Suicide Risk Prediction
# MAGIC 
# MAGIC In this tutorial, we'll see how we can easily pre-annotate data using pre-defined vocabulary / key-word matching, and upload them as pre-annotations to NLPLab.
# MAGIC 
# MAGIC We'll be using the AnnotationLab module to create, configure, export and import projects with minimal code.
# MAGIC 
# MAGIC Note: The Annotation Lab module is available in Spark NLP for Healthcare 4.2.2+.
# MAGIC 
# MAGIC ## Following are the main steps in this exercise:
# MAGIC 
# MAGIC ### 1. Using string matching and existing off-the-shelf vocabularies, create a simple pipeline to get rudimentary results.
# MAGIC ### 2. Upload the initial results to Annotation Lab, annotate, and download annotations.
# MAGIC ### 3. Train an NER model on the annotated data to achieve better performance.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Initial Configurations

# COMMAND ----------

# MAGIC %pip install tensorflow_addons protobuf==3.20.*

# COMMAND ----------

import pandas as pd
import os

import sparknlp
import sparknlp_jsl
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *

from pyspark.ml import Pipeline, PipelineModel
from sparknlp.training import CoNLL

print('sparknlp.version : ',sparknlp.version())
print('sparknlp_jsl.version : ',sparknlp_jsl.version())

spark

# COMMAND ----------

# MAGIC %md
# MAGIC For this exercise, we define two different layers of solutions:
# MAGIC 1. Bronze: Text Matcher based rudimentary results - that will be uploaded in annotation lab and refined.
# MAGIC 2. Silver: After annotating the documents properly in Annotation Lab, train an NER model, get results

# COMMAND ----------

delta_bronze_path='/FileStore/HLS/nlp/delta/bronze/'

dbutils.fs.mkdirs(delta_bronze_path)

os.environ['delta_bronze_path']=f'/dbfs{delta_bronze_path}'

delta_silver_path='/FileStore/HLS/nlp/delta/silver/'

dbutils.fs.mkdirs(delta_silver_path)

os.environ['delta_silver_path']=f'/dbfs{delta_silver_path}'


# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Using string matching and existing off-the-shelf vocabularies, create a simple pipeline to get rudimentary results.
# MAGIC 
# MAGIC First, we'll rely on existing vocabularies comprising of key words e.g: "kill my self", "want to die" etc to get preliminary results.
# MAGIC 
# MAGIC The vocbulary in this exercise is obtained from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4690390/

# COMMAND ----------

# MAGIC %md
# MAGIC **About the dataset:** The dataset is a collection of reddit posts, taken from the work done by: 
# MAGIC - http://users.umiacs.umd.edu/~resnik/umd_reddit_suicidality_dataset.html
# MAGIC - https://github.com/hesamuel/goodbye_world

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.1. Download resources and explore existing vocabularies

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC cd $delta_bronze_path
# MAGIC 
# MAGIC wget -O suicide_pred_vocab_data.zip https://github.com/JohnSnowLabs/spark-nlp-workshop/raw/master/databricks/python/healthcare_case_studies/data/suicide_pred_vocab_data.zip
# MAGIC 
# MAGIC unzip -o suicide_pred_vocab_data.zip

# COMMAND ----------

# checking for files

data_vocab_path = f"{delta_bronze_path}suicide_pred_vocab_data/"

dbutils.fs.ls(data_vocab_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Taking a look at the vocabulary list

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC head /dbfs/FileStore/HLS/nlp/delta/bronze/suicide_pred_vocab_data/suicide_behavior_vocab.txt

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2. Creating a Spark NLP pipeline using textmatchers to find entities in the data

# COMMAND ----------

documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
  .setInputCols("sentence")\
  .setOutputCol("token")

text_matcher1 = TextMatcher().setInputCols("sentence","token").setOutputCol("textmatch_chunk1").setEntityValue("SUICIDE_BEHAVIOR")\
    .setEntities(data_vocab_path+"suicide_behavior_vocab.txt").setCaseSensitive(False).setMergeOverlapping(True)\
    .setBuildFromTokens(True)

text_matcher2 = TextMatcher().setInputCols("sentence","token").setOutputCol("textmatch_chunk2").setEntityValue("SUICIDE_PSYCHACHE")\
    .setEntities(data_vocab_path+"suicide_psychache_vocab.txt").setCaseSensitive(False).setMergeOverlapping(True)\
    .setBuildFromTokens(True)

text_matcher3 = TextMatcher().setInputCols("sentence","token").setOutputCol("textmatch_chunk3").setEntityValue("PAST_TRAUMA")\
    .setEntities(data_vocab_path+"suicide_trauma_vocab.txt").setCaseSensitive(False).setMergeOverlapping(True)\
    .setBuildFromTokens(True)

chunk_merger = ChunkMergeApproach()\
    .setInputCols(['textmatch_chunk1', 'textmatch_chunk2', 'textmatch_chunk3'])\
    .setOutputCol("all_chunks")

pipeline =  Pipeline(
    stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        text_matcher1, text_matcher2, text_matcher3,
        chunk_merger
    ]
)

p_model = pipeline.fit(spark.createDataFrame(pd.DataFrame({'text': ['']})))

l_model = LightPipeline(p_model)

# COMMAND ----------

# Load the target data.

data = spark.read.csv(data_vocab_path+'suicide_det_data.csv').withColumnRenamed('_c0', 'text')
print (data.count())
data.show(3)


# COMMAND ----------

results = p_model.transform(data).collect()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.4. Analyzing Results
# MAGIC 
# MAGIC Even the basic textmatching approach works in general, but has the following challenges:
# MAGIC 1. Lack of context - leading to too many false positives.
# MAGIC 2. Less meaningful / incomplete chunks.

# COMMAND ----------

# random check
from sparknlp_display import NerVisualizer

displayHTML(NerVisualizer().display(results[500], 'all_chunks', return_html=True))

# COMMAND ----------

# Checking on a piece of text
text = """My ex attempted suicide - she got way too attached and when i ghosted her i wanted to hate her , but i couldn’t after her attempt . so i hated myself ."""

results_single = l_model.fullAnnotate(text)[0]

from sparknlp_display import NerVisualizer

displayHTML(NerVisualizer().display(results_single, 'all_chunks', return_html=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Now, we can use these results as pre-annotations and uploade on the Annotation Lab. Pre-annotations help reduce manual annotation time as the annotator does not need to annotate everything, but rather make corrections.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC For this exercise, we are using John Snow Lab's Annotation Lab tool.
# MAGIC 
# MAGIC The Annotation Lab is a stand-alone web interface designed to be used and installed inside any organization's environment to protect data privacy. It can be easily installed on a single VM.
# MAGIC 
# MAGIC More details and instructions can be found here: https://nlp.johnsnowlabs.com/docs/en/alab/install#aws-marketplace. 
# MAGIC 
# MAGIC While the tasks and pre-annotations can be uploaded directly via web interface as well, we are leveraging the API module for convenience.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1. Generate Pre-annotations using the Alab Module.

# COMMAND ----------

# MAGIC %md
# MAGIC Initialize Alab module

# COMMAND ----------

# annotation lab module
from sparknlp_jsl.alab import AnnotationLab

alab = AnnotationLab()

# COMMAND ----------

# NOTE: "all_results" is the result of the pipeline after running on sample docs.
pre_annotations, summary = alab.generate_preannotations(all_results = results, document_column = 'document', ner_columns = ['all_chunks'])
pre_annotations[:10]

# COMMAND ----------

## What is summary? 
# While a user would know all the entities comming out of the NLP pipeline, listing all of them manually is laborious.
# The summary object helps identify how many types of entities, assertions, and relations are present.
# This saves the user from listing all labels individually; we'll use this object while setting project configuration.
summary

# COMMAND ----------

# MAGIC %md
# MAGIC ## <font color=#FF0000>**Note: If you don't have credentials for the Annotation Lab, we have provided the annotations for ~100 tasks. Jump to section 3.2 to directly download the exported JSON file, and start training**</font>.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2. Set Annotation Lab credentials and Create a New Project

# COMMAND ----------

# Set Credentials
username = 'admin'
password = dbutils.secrets.get("solution-accelerator-cicd", "alab-password")
client_secret = dbutils.secrets.get("solution-accelerator-cicd", "alab-client-secret")  # see https://nlp.johnsnowlabs.com/docs/en/alab/api#get-client-secret
annotationlab_url = dbutils.secrets.get("solution-accelerator-cicd", "alab-url") # your alab instance URL (could even be internal URL if deployed on-prem).

alab.set_credentials(

  # required: username
  username=username,

  # required: password
  password=password,

  # required: secret for you alab instance (every alab installation has a different secret)
  client_secret=client_secret, 

  # required: http(s) url for you annotation lab
  annotationlab_url=annotationlab_url
)

# COMMAND ----------

# create new project
alab.create_project('suicide_detection') # Shows 400 if the project already exists. This error can be ignored

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3. Set project configuration (NER label tags, assertion classes, and relation tags)

# COMMAND ----------

# set configuration

## either manually define labels:
# alab.set_project_config(
#   project_name = 'suicide_detection',
#   ner_labels = ['SUICIDE_BEHAVIOR', 'SUICIDE_TRAUMA', 'SUICIDE_OTHERS', 'SUICIDE_PSYCHACHE']
# )

# OR use the summary object which already has all details

alab.set_project_config(
  project_name = 'suicide_detection',
  ner_labels = summary['ner_labels'],
  assertion_labels = summary['assertion_labels'],
  relations_labels = summary['re_labels']
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4. Upload pre-annotations to the newly created project
# MAGIC #### <font color=#FF0000>Note: You can upload all the tasks and annotate. For demo purpose, we are only uploading 5 tasks.</font>

# COMMAND ----------

# Upload documents to Alab

alab.upload_preannotations(
  project_name = 'suicide_detection',
  preannotations = pre_annotations[:5]) # testing with 5 annotations

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.5. Annotate documents on Annotation Lab and make necessary corrections
# MAGIC 
# MAGIC **2.5.1 The first step for annotations is developing, and adhering to some guidelines, which are crucial for controlling the flow of annotations and avoid confusions between entitiy types.**
# MAGIC 
# MAGIC **An example Annotation Guideline (AG) is available [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/databricks/python/healthcare_case_studies/data/suicide_pred_AG.docx).**
# MAGIC 
# MAGIC **2.5.2 Once annotation guildelines have been finalized, annotations can be started.**
# MAGIC 
# MAGIC **Since we have already uploaded pre-anntations to the alab, we can get started.**
# MAGIC 
# MAGIC 1.       Go to the Projects -> suicide_detection -> tasks
# MAGIC 
# MAGIC 2.       Select the first task
# MAGIC 
# MAGIC 3.       Click Edit -> select NER type -> select corresponding text, as defined in Annotation Guidelines.
# MAGIC 
# MAGIC 4.       Click Save -> Submit.
# MAGIC 
# MAGIC 5.       Go to the next task, until all tasks are completed.
# MAGIC 
# MAGIC In the tasks overview, you should see all 5 tasks submitted.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Now, we can train an NER model using the annotations performed on the Annotation Lab

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1. Import the annotations from Annotation Lab and save as a JSON file.

# COMMAND ----------

exported_json = alab.get_annotations(
  project_name = 'suicide_detection', 
  output_name='result',
  save_dir=f"/dbfs/{delta_silver_path}")


# COMMAND ----------

print (delta_silver_path)
dbutils.fs.ls(delta_silver_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2. Convert the JSON file to CoNLL format for training an NER model

# COMMAND ----------

# MAGIC %md
# MAGIC #### <font color=#FF0000>Note: For demo purpose, we are downloading the annotated tasks from a public source (instead of annotation lab).</font>

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC cd $delta_silver_path
# MAGIC 
# MAGIC wget -O suicide_pred_annotations.json https://github.com/JohnSnowLabs/spark-nlp-workshop/raw/master/databricks/python/healthcare_case_studies/data/suicide_pred_annotations.json

# COMMAND ----------

print (delta_silver_path)
dbutils.fs.ls(delta_silver_path)

# COMMAND ----------

from sparknlp_jsl.alab import AnnotationLab
alab = AnnotationLab()
alab.get_conll_data(spark, f"/dbfs/{delta_silver_path}suicide_pred_annotations.json", output_name='conll_demo', save_dir=f"/dbfs/{delta_silver_path}")

# COMMAND ----------

dbutils.fs.ls(delta_silver_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.3. Train NER Model

# COMMAND ----------

# MAGIC %md
# MAGIC Load Data

# COMMAND ----------

conll_data = CoNLL().readDataset(spark, f"{delta_silver_path}conll_demo.conll")

conll_data.show(3)

# COMMAND ----------

# MAGIC %md
# MAGIC Look at label distribution

# COMMAND ----------


from pyspark.sql import functions as F

conll_data.select(F.explode(F.arrays_zip(conll_data.token.result,
                                         conll_data.label.result)).alias("cols")) \
          .select(F.expr("cols['0']").alias("token"),
                  F.expr("cols['1']").alias("ground_truth"))\
          .groupBy('ground_truth')\
          .count()\
          .orderBy('count', ascending=False)\
          .show(100,truncate=False)



# COMMAND ----------

# MAGIC %md
# MAGIC Select Embeddings

# COMMAND ----------

clinical_embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical', "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

# COMMAND ----------

# MAGIC %md
# MAGIC Graph Builder to automatically generate a TensorFlow graph for training.

# COMMAND ----------

graph_folder_path = "/dbfs/ner/medical_ner_graphs"

ner_graph_builder = TFGraphBuilder()\
    .setModelName("ner_dl")\
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setLabelColumn("label")\
    .setGraphFolder(graph_folder_path)\
    .setGraphFile("auto")\
    .setHiddenUnitsNumber(20)

# COMMAND ----------

# MAGIC %md
# MAGIC Ner Model with hyper-parameters

# COMMAND ----------

nerTagger = MedicalNerApproach()\
  .setInputCols(["sentence", "token", "embeddings"])\
  .setLabelColumn("label")\
  .setOutputCol("ner")\
  .setMaxEpochs(35)\
  .setBatchSize(8)\
  .setRandomSeed(0)\
  .setVerbose(1)\
  .setLr(0.001)\
  .setEvaluationLogExtended(True) \
  .setEnableOutputLogs(True)\
  .setOutputLogsPath('dbfs:/ner/ner_logs')\
  .setUseBestModel(True)\
  .setGraphFolder('dbfs:/ner/medical_ner_graphs')\
  .setValidationSplit(0.2)
  # .setEnableMemoryOptimizer(True) #>> if you have a limited memory and a large conll file, you can set this True to train batch by batch       

ner_pipeline = Pipeline(stages=[
          clinical_embeddings,
          ner_graph_builder,
          nerTagger
 ])

# COMMAND ----------

# MAGIC %md
# MAGIC Train

# COMMAND ----------

model = ner_pipeline.fit(conll_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Check logs

# COMMAND ----------

ls -l /dbfs/ner/ner_logs/

# COMMAND ----------

import fnmatch
import os
filename = fnmatch.filter(os.listdir('/dbfs/ner/ner_logs/'), 'MedicalNerApproach_*.log')[0]

with open(f'/dbfs/ner/ner_logs/{filename}', 'r') as f_:
  lines = ''.join(f_)
print (lines)

# COMMAND ----------

# MAGIC %md
# MAGIC Save the model to disk, load and test on the new model

# COMMAND ----------

# save model
model.stages[-1].write().overwrite().save(delta_silver_path+'ner_model')

# COMMAND ----------

# use model in a prediction pipeline

documentAssembler = DocumentAssembler()\
            .setInputCol("text")\
            .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel().pretrained('sentence_detector_dl_healthcare', 'en', 'clinical/models')\
           .setInputCols(["document"])\
           .setOutputCol("sentence")

tokenizer = Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")
embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical', 'en', 'clinical/models') \
      .setInputCols("sentence", "token") \
      .setOutputCol("embeddings")

#load the trained ner model
ner_model =MedicalNerModel().load(delta_silver_path+'ner_model')\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner_tags")

#ner converter ner jsl enriched
ner_chunk = NerConverterInternal()\
    .setInputCols(['sentence', 'token', 'ner_tags']) \
    .setOutputCol('ner_chunk')


pipeline=Pipeline(stages=[
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        ner_model,
        ner_chunk,
        
         ])
empty_data = spark.createDataFrame([[""]]).toDF("text")
model = pipeline.fit(empty_data)
light_model = LightPipeline(model)

# COMMAND ----------

# Checking on a piece of text
text = """My ex attempted suicide - she got way too attached and when i ghosted her i wanted to hate her , but i couldn’t after her attempt . so i hated myself ."""

results_single = light_model.fullAnnotate(text)[0]

from sparknlp_display import NerVisualizer

displayHTML(NerVisualizer().display(results_single, 'ner_chunk', return_html=True))

# COMMAND ----------


