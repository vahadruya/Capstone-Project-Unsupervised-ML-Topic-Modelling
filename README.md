# Topic Modeling of BBC News Articles
This project is a Capstone Project done as part of Unsupervised Machine Learning. A set of 2225 BBC News Articles are analysed to identify the underlying themes and topics within them.

<details>
<summary>Table of Contents</summary>

1. [About the Project](#about-the-project)
2. [Data Reading and Description](#data-reading-and-description)
3. [Data Pre-Processing](#data-pre-processing)
4. [Model Implementation](#model-implementation)
    + [LDA Model](#1-lda-model)
    + [LSA Model](#2-lsa-model)
5. [Model Evaluation](#model-evaluation)
6. [Results](#results)
7. [Conlusion](#conclusion)
8. [Challenges Faced](#challenges-faced)
9. [Libraries Used](#libraries-used)
10. [Contact](#contact)
</details>

## About The Project

Topic modelling is a widely used technique in natural language processing that helps to extract latent topics from a large collection of documents. In the context of News Articles, it categorises these documents into various categories of requirement, which is very helpful for organisations to manage their content and for the readers as well, to easily find articles of interest.

It can also help in content summarisation by breaking down the lengthy articles into keywords and themes to briefly summarise the content in a concise manner, without loss of information.

This Project focuses on the former application, to determine the underlying topics within the corpus of News Articles. The original category of each article is provided as an input for evaluation of the topic modeling algorithm. It should be noted that these original categories are not considered as an input for modeling and is in no way influences the algorithm metholody.
<div align = "right">    
  <a href="#topic-modeling-of-bbc-news-articles">(back to top)</a>
</div>

## Data Reading and Description

The Dataset was available as individual txt files for each article, with their original category/topic provided as an input. It was read into the Python Notebook using **re** and **glob** libraries and converted into a single DataFrame with the following columns:
*   **Title**: Title of the article
*   **Description**: Content of the article
*   **Category**: Original Category of the article

<div align = "right">    
  <a href="#topic-modeling-of-bbc-news-articles">(back to top)</a>
</div>

## Data Pre-Processing

Before applying the topic modeling algorithms, the textual data was preprocessed to expand contractions, remove punctuations, digits, whitespaces and stop words, and to lemmatize the remaining words. The resulting corpus was then vectorized using both the Count and TFIDF vectorizers, with each row in the vectorized data representing a document and each column representing a unique term in the corpus

<div align = "right">    
  <a href="#topic-modeling-of-bbc-news-articles">(back to top)</a>
</div>

## Model Implementation

Three models were applied on the vectorized data, with the first two being variations of **Latent Dirichlet Allocation (LDA)** algorithm and the third one using the **Latent Semantic Analysis (LSA)**.
### 1. LDA Model

The Latent Dirichlet Allocation (LDA) model was trained on the preprocessed data using the Scikit-learn library. The model was optimized using GridSearchCV to determine the optimal number of topics. The model was trained on two different vectorized data inputs - one in which the vectorization method was using CountVectorizer, and the other with TFIDF-Vectorizer.

### 2. LSA Model

The LSA model was trained on the pre-processed data using the TruncatedSVD class in sciki-learn library. Similar to LDA model, both TFIDF and CountVectorized inputs were given to the model, and each topic was analysed.

<div align = "right">    
  <a href="#topic-modeling-of-bbc-news-articles">(back to top)</a>
</div>

## Model Evaluation

The Log-Likelihood and Perplexity scores were evaluated for each of the models. Since the original category of each article is provided as an input, the metrics of evaluation for a Classification problem - Precision, Recall and F1 Score - are utilised here. F1 Score is given priority because of the equality and non-hierarchy between each topic.

<div align = "right">    
  <a href="#topic-modeling-of-bbc-news-articles">(back to top)</a>
</div>

## Results

It was found that the LDA models outperformed the LSA models. The LSA model identified primarily only two topics, with it over-predicting on one of them. In contrast, the LDA models identified all the five topics corresponding closely with the original article categories.

Comparing the two LDA models, the model with CountVectorized data as the input well outperformed the other model. The model accuracy was close to 93%, with the latter only scoring about 85%. It fared better in the other metrics as well, except in Log-likelihood score.

<div align = "right">    
  <a href="#topic-modeling-of-bbc-news-articles">(back to top)</a>
</div>

## Conclusion

Overall, the LDA model with CountVectorizer proved to be a more effective approach to topic modeling of the BBC News articles dataset, producing results which closely corresponded with the original article categories. This project demonstrates the usefulness of topic modeling techniques for understanding large text datasets and the importance of selecting an appropriate algorithm for the task at hand.

<div align = "right">    
  <a href="#topic-modeling-of-bbc-news-articles">(back to top)</a>
</div>

## Challenges Faced

*   The first challenge faced in the project was right at the start: reading the text files to form a consolidated, tabular dataset. It took some bit of searching and reading of documentations to attain the knowledge and apply it to code. Some encoding errors also needed to be tackled in the process.
*   While pre-processing was mostly a breeze, the choice of vectorization methods for specific models was another challenge. It was (rightly) expected that the CountVectorizer would be suitable for LDA, so the same was considered a Hypothesis which was tested by deploying the LDA on both the Bag-of-Words model and TFIDF Vectorized dataset. The Hypotheses statement finally turned to be true, atleast in this context.
*   Finally, the choice of evaluation metrics: Apart from Perplexity and Log-Likelihood, another set of metrics were needed which is more interpretable to evaluate the models. So, the metrics usually used for Classification Problems - Precision, Recall and F1 Score, was chosen. It seemed unconventional at first, but proved very fruitful.

<div align = "right">    
  <a href="#topic-modeling-of-bbc-news-articles">(back to top)</a>
</div>

## Libraries Used

For reading, handling and manipulating data

* |[glob](https://docs.python.org/3/library/glob.html)|
  |---|
* |[re](https://docs.python.org/3/library/re.html)|
  |---|
* |[pandas](https://pandas.pydata.org)|
  |---|
* |[numpy](https://numpy.org)|
  |---|
* |[random](https://docs.python.org/3/library/random.html)|
  |---|

For Visualisation
* |[matplotlib](https://matplotlib.org)|
  |---|
* |[seaborn](https://seaborn.pydata.org)|
  |---|
* |[pyLDAvis](https://pyldavis.readthedocs.io/en/latest/readme.html)|
  |---|
* |[wordcloud](https://pypi.org/project/wordcloud/)|
  |---|

For Textual Pre-processing and Model Building
* |[string](https://docs.python.org/3/library/string.html)|
  |---|
* |[nltk](https://nltk.org)|
  |---|
* |[contractions](https://pypi.org/project/pycontractions/)|
  |---|
* |[sklearn](https://scikit-learn.org/stable/)|
  |---|

<div align = "right">    
  <a href="#topic-modeling-of-bbc-news-articles">(back to top)</a>
</div>

## Contact

|[Gmail](mailto:apaditya96@gmail.com)|[Linkedin](https://www.linkedin.com/in/aditya-a-p-507b1b239)|
|---|---|

<div align = "right">    
  <a href="#topic-modeling-of-bbc-news-articles">(back to top)</a>
</div>
