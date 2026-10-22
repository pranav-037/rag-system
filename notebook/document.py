#!/usr/bin/env python
# coding: utf-8

# # Data Ingestion

# In[ ]:


###Document Structure

from langchain_core.documents.base import Document


# In[ ]:


doc=Document(
    page_content="this is a test document",
    metadata={
        "source": "example",
        "pages":1,
        "author": "Pranav Thakur",
        "date_created": "2026-09-15"
    } 
)
doc


# In[ ]:


###Create a simple txt file

import os
os.makedirs("../data/text_files", exist_ok=True)


# In[ ]:


sample_texts={
    "../data/text_files/artificial_intelligence.txt":"""
Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making.
It is a field of research in engineering, mathematics, and computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximise their chances of achieving defined goals.
High-profile applications of AI include advanced web search engines, chatbots, virtual assistants, autonomous vehicles, play and analysis in strategy games (e.g., chess and Go), and content generation (e.g. images, audio, and videos).
""",
"../data/text_files/machine_learning.txt":"""
Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from  data and generalize to unseen data, and thus perform tasks without being explicitly programmed.
Statistics and mathematical optimisation methods compose the foundations of machine learning. Data mining is a related field of study, focusing on exploratory data analysis (EDA) through unsupervised learning.
From a theoretical viewpoint, probably approximately correct learning provides a mathematical and statistical framework for describing machine learning. Most traditional machine learning and deep learning algorithms can be described as empirical risk minimisation under this framework.
Advances in the field of deep learning have allowed neural networks, a class of statistical algorithms, to surpass many previous machine learning approaches in performance. 
The resulting dominance of machine learning within the field of AI from the late 2000s and early 2010s onwards was such that the terms "machine learning" and "artificial intelligence" are now often used interchangeably, even though they are not synonymous."""
}

for filepath,content in sample_texts.items():
    with open(filepath,'w',encoding="utf-8") as f:
        f.write(content)
print("Sample text files created successfully.")        


# In[ ]:


###TextLoader

from langchain_community.document_loaders import TextLoader

loader=TextLoader("../data/text_files/artificial_intelligence.txt",encoding="utf-8")

document=loader.load()
print(document)


# In[ ]:


###DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader

###load all text files in the directory
dir_loader=DirectoryLoader(
    "../data/text_files",
    glob="**/*.txt", ##Pattern to match files
    loader_cls=TextLoader, ##loader class to use for loading files
    loader_kwargs={'encoding': 'utf-8'},
    show_progress=False

)

documents=dir_loader.load()
documents


# In[ ]:


from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader

###load all text files in the directory
dir_loader=DirectoryLoader(
    "../data/pdf",
    glob="**/*.pdf", ##Pattern to match files
    loader_cls=PyMuPDFLoader, ##loader class to use for loading files
    show_progress=False

)

pdf_documents=dir_loader.load()
pdf_documents


# In[ ]:


type(pdf_documents[0])


# In[ ]:




