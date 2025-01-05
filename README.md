# NLP Spell-Checker System For Cancer Research

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-DistilBERT-yellow)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains an advanced spelling correction system that detects and corrects both non-word and real-word errors. Designed for domain-specific use, such as medical text correction, the system integrates traditional NLP techniques with machine learning models to ensure high accuracy and usability. The project is implemented with a user-friendly Graphical User Interface (GUI) for seamless interaction.

---

## 🚀 Project Overview

### ✨ **System Features**

1. **Error Detection**  
   🛠️ **Non-word Errors**:  
   - Detects words not present in the corpus using edit distance algorithms.  
   - Provides ranked suggestions based on similarity to valid words.  

   🔍 **Real-word Errors**:  
   - Identifies words used out of context with the help of bigram models and DistilBERT.  

2. **Error Correction**  
   - 🧠 Generates ranked correction suggestions using:  
     - 🖍️ Edit Distance for spelling similarity.  
     - 📊 Probabilistic models for context-aware ranking.  

3. **Graphical User Interface (GUI)**  
   - 🎨 Interactive error highlighting with:  
     - **Red** for non-word errors.  
     - **Blue** for real-word errors.  
     - **Green** for corrected words (underlined).  
   - 💡 Features include real-time corrections, word search, and alphabetical sorting.  

4. **Performance Metrics**  
   - ⚡ Fine-tuned DistilBERT achieved a validation accuracy of **99.88%**.  
   - ⏱️ Optimized for real-time feedback with minimal latency.  

---

## 🛠️ Technologies and Libraries

### **Algorithms**  
- 📏 **Edit Distance**:  
  Levenshtein and Damerau-Levenshtein distance for error detection and correction.  
- 🔗 **Bigram Analysis**:  
  Context-aware word pairing probabilities.  

### **Machine Learning**  
- 🤖 **DistilBERT**:  
  Fine-tuned transformer model for real-word error detection.  

### **Python Libraries**  
- 🔥 **PyTorch**: Model training and optimization.  
- 🦗 **Hugging Face Transformers**: Pretrained models and tokenizers.  
- 📖 **NLTK**: Bigram implementation and corpus processing.  
- 🖥️ **Tkinter & TtkBootstrap**: GUI development and styling.  
- ⚙️ **Regex**: Text preprocessing.  
- 📊 **Collections (Counter)**: Word frequency analysis.  

---

## ✍️ Usage

### **Graphical User Interface**  

1. **Input Area**:  
   - Paste or type text into the editor.  
   - Word limit: **500 words**.  

2. **Error Detection**:  
   - Errors flagged in real-time:  
     - **Red**: Non-word errors.  
     - **Blue**: Real-word errors.  
   - Corrected words are shown in **green** and underlined.  

3. **Suggestions**:  
   - Right-click on flagged words for correction suggestions.  
   - Select a suggestion or ignore the flagged word.  

### **Word Management**  
- 🔍 Search words in the corpus using the search box.  
- 🐂️ Alphabetically sort words to navigate through the corpus.  

---

## 🛠️ System Design

### **Workflow**  

1. 🧹 **Corpus Preprocessing**:  
   - Clean text by removing special characters.  
   - Tokenize into words and calculate word frequencies.  
   - Store frequencies for ranking correction candidates.  

2. 🔎 **Error Detection**:  
   - 🛠️ Non-word Errors:  
     - Use Levenshtein distance to identify misspellings.  
   - 🔗 Real-word Errors:  
     - Employ bigram analysis to detect contextual anomalies.  

3. ✨ **Suggestions**:  
   - Rank corrections by:  
     - Edit distance similarity.  
     - Probabilistic context-based ranking.  

4. 🎨 **GUI Integration**:  
   - Features real-time highlighting, user interactions, and corpus management tools.  

---

## 📊 Results

### **Strengths**  
- High accuracy in domain-specific text correction.  
- Efficient handling of both simple and complex spelling errors.  
- User-friendly interface with intuitive controls.  

### **Limitations**  
- Does not support grammar corrections.  
- Scalability challenges with larger corpora.  

---

## 🚀 Future Enhancements  

1. Expand to include grammar correction.  
2. Enable batch text processing.  
3. Optimize for longer corpora and complex datasets.  

---

## 📚 References  

Refer to the [report](./NLP_ASSIGNMENT_REPORT_EDITED.pdf) for detailed citations and explanations of algorithms and methodologies used.
