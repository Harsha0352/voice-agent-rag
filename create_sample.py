# If you don't have a PDF library like fpdf, we can just name a text file as .pdf for a crude test
# But pypdf expects a real PDF. Let's use reportlab if possible or just assume they have one.
# Since I can't be sure of reportlab, I'll just provide a text-based sample they can 'print to PDF' from Word/Notepad.

content = """
DATA SCIENCE PROJECT INFO
-------------------------
The RAG system is a Retrieval-Augmented Generation implementation.
It uses Whisper for Speech-to-Text and VITS for Text-to-Speech.
The LLM used is Llama 3.2 1B Instruct.
The vector database is FAISS.
It now supports PDF uploads for custom knowledge bases.
"""

with open("sample_content.txt", "w") as f:
    f.write(content)

print("Created sample_content.txt. You can 'Save as PDF' from any editor to test the upload feature!")
