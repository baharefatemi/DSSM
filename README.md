# DSSM
An implementation of the paper: Learning Deep Structured Semantic Models for Web Search using Clickthrough Data


This is an implementation of a latent semantic model that intend to map a query to its relevant document where keyword-based approaches often fails. This is a model with deep structure that project queries and documents into a common low-dimensional space where the relevance of a document given a query is readily computed as the distance between them.

This model can be used as a search engine that helps people find out their desired document even with searching a query that:
1. has different words than the document
2. is abbreviation of the document words
3. changed the order of the words in the document
4. shortened words in the document
5. has typos
5. has spacing issues
...

## Dependencies

* `Python` version 2.7 or higher
* `Numpy` version 1.13.1 or higher
* `Tensorflow` version 1.1.0 or higher
