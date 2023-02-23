# Detecting suicidal behaviour in text.

Suicide is the 13th leading cause of death globally, accounting for 5-6% of all deaths (https://pubmed.ncbi.nlm.nih.gov/23245604/). The risk of completed suicide varies worldwide by various sociodemographic characteristics, with young adults, teens, and males facing the highest risks of suicide completion.

Given the signifcant role of social media, it has been observed that a substantial amount of suicidal behaviour is reported on such mediums which can be useful to study various aspects of suicide.

While we can collect a lot of data from social media websites and forums, there is a lack of annotated datasets that can be leveraged to train ML models to deal with the challenging nature of social media texts.


# Annotating Data & Training Custom NER model for Suicide Risk Prediction

In this tutorial, we'll see how we can easily pre-annotate data using pre-defined vocabulary / key-word matching, and upload them as pre-annotations to NLPLab.

We'll be using the AnnotationLab module to create, configure, export and import projects with minimal code.

Note: The Annotation Lab module is available in Spark NLP for Healthcare 4.2.2+.

## Following are the main steps in this exercise:

### 1. Using string matching and existing off-the-shelf vocabularies, create a simple pipeline to get rudimentary results.
### 2. Upload the initial results to Annotation Lab, annotate, and download annotations.
### 3. Train an NER model on the annotated data to achieve better performance.


### Following is the life cycle of an annotatioin project - from developing Annotation Guidelines to Pre-Annotating to training, and testing.

![Annotation Life Cycle](https://www.johnsnowlabs.com/wp-content/uploads/2022/12/chart-Annotation_Lab_2.png)


## License
Copyright / License info of the notebook. Copyright [2021] the Notebook Authors.  The source in this notebook is provided subject to the [Apache 2.0 License](https://spdx.org/licenses/Apache-2.0.html).  All included or referenced third party libraries are subject to the licenses set forth below.
|Library Name|Library License|Library License URL|Library Source URL|
| :-: | :-:| :-: | :-:|
|Spark NLP Display|Apache License 2.0|https://github.com/JohnSnowLabs/spark-nlp-display/blob/main/LICENSE|https://github.com/JohnSnowLabs/spark-nlp-display|
|Spark NLP |Apache License 2.0| https://github.com/JohnSnowLabs/spark-nlp/blob/master/LICENSE | https://github.com/JohnSnowLabs/spark-nlp|
|Spark NLP for Healthcare|[Proprietary license - John Snow Labs Inc.](https://www.johnsnowlabs.com/spark-nlp-health/) |NA|NA|
|Author|
|-|
|Databricks Inc.|
|John Snow Labs Inc.|


## Disclaimers
Databricks Inc. (“Databricks”) does not dispense medical, diagnosis, or treatment advice. This Solution Accelerator (“tool”) is for informational purposes only and may not be used as a substitute for professional medical advice, treatment, or diagnosis. This tool may not be used within Databricks to process Protected Health Information (“PHI”) as defined in the Health Insurance Portability and Accountability Act of 1996, unless you have executed with Databricks a contract that allows for processing PHI, an accompanying Business Associate Agreement (BAA), and are running this notebook within a HIPAA Account.  Please note that if you run this notebook within Azure Databricks, your contract with Microsoft applies.

## Instruction
To run this accelerator, set up JSL Partner Connect [AWS](https://docs.databricks.com/integrations/ml/john-snow-labs.html#connect-to-john-snow-labs-using-partner-connect), [Azure](https://learn.microsoft.com/en-us/azure/databricks/integrations/ml/john-snow-labs#--connect-to-john-snow-labs-using-partner-connect) and navigate to **My Subscriptions** tab. Make sure you have a valid subscription for the workspace you clone this repo into, then **install on cluster** as shown in the screenshot below, with the default options. You will receive an email from JSL when the installation completes.

<br>
<img src="https://raw.githubusercontent.com/databricks-industry-solutions/oncology/main/images/JSL_partner_connect_install.png" width=65%>

Once the JSL installation completes successfully, clone this repo into a Databricks workspace. Attach the `RUNME` notebook to any cluster and execute the notebook via `Run-All`. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. Execute the multi-step-job to see how the pipeline runs.