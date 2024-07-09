from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader("example/method_statement/2209.01390.pdf")
# pages = loader.load_and_split()

# print(pages[00])

import pypdf

pdfs = pypdf.PdfReader('example/method_statement/1701-W-000-CSC-760-000047.pdf')
pdf = pdfs.loader.load_and_split()
pdf[00]