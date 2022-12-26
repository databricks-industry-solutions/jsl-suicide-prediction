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