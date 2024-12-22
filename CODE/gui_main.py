import tkinter as tk
from tkinter import messagebox, Menu
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

import string
import regex as re
import torch
from transformers import DistilBertForMaskedLM, DistilBertTokenizer
from difflib import SequenceMatcher

from preprocess_corpus import word_counter, data
from edit_distance import word_candidates, P

# Load the fine-tuned BERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('./fine_tuned_distilbert1')
model = DistilBertForMaskedLM.from_pretrained('./fine_tuned_distilbert1')

# Set of common words to ignore
common_words = set([
    "i", "the", "a", "an", "and", "or", "is", "are",
    "to", "for", ".", ",", ":", ";", "-", "(", ")",
    "\"", "'", "in", "on", "at", "by", "with", "from"
])

# Set to store ignored words
ignored_words = set()

# Predict masked word using distilBERT
def predict_masked_word(text, masked_index):
    inputs = tokenizer.encode_plus(text, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Adjust masked_index to map to tokenized form
    tokenized_text = tokenizer.convert_ids_to_tokens(input_ids[0])
    word_tokens = tokenizer.tokenize(text)
    
    # Find the actual token index that corresponds to the masked word
    # This might involve mapping from the plain text to the tokenized text.
    tokens[masked_index] = '[MASK]'
    masked_input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids[0, masked_index] = masked_input_ids[masked_index]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = outputs.logits

    masked_index_prediction = predictions[0, masked_index].topk(10).indices
    predicted_tokens = [tokenizer.decode([idx]).strip() for idx in masked_index_prediction]
    return predicted_tokens

# Find misused words in the input text
def find_misused_words(text):
    inputs = tokenizer.encode_plus(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    misused_words = []

    for i, word in enumerate(tokens):
        if (word not in tokenizer.all_special_tokens and 
            word.isalpha() and 
            word.lower() not in common_words and 
            word.lower() not in ignored_words):
            predicted_words = predict_masked_word(text, i)
            top_predictions = [pred.lower() for pred in predicted_words[:3]]
            if word.lower() not in top_predictions:
                misused_words.append((word, i, predicted_words))
    return misused_words

# Function to update the word count display and handle word limit
def update_word_count(event=None):
    text_content = input_text_widget.get("1.0", tk.END).strip()
    words = re.findall(r'\b\w+\b', text_content)
    word_count = len(words)
    
    if word_count > 500:
        word_count_label.config(foreground="red")
        word_count_label.config(text=f"Word count: {word_count} / 500 (Word limit exceeded!)")
    else:
        word_count_label.config(foreground="black")
        word_count_label.config(text=f"Word count: {word_count} / 500")

# Function to correct the input text
def correct_input():
    input_text = input_text_widget.get("1.0", tk.END).strip()
    if input_text:
        input_text_widget.delete("1.0", tk.END)
        misused_words = find_misused_words(input_text)
        words = re.findall(r'\b\w+\b', input_text)
        error_found = False

        for word in words:
            if word.lower() in common_words or word.lower() in word_counter or word.lower() in ignored_words:
                input_text_widget.insert(tk.END, word + " ")
            else:
                error_found = True
                input_text_widget.insert(tk.END, word + " ", "misspelled")

        for word, idx, predicted_words in misused_words:
            error_found = True
            # Highlight the misused word
            start_index = f"1.0+{find_word_start_position(input_text, word)}c"
            end_index = f"1.0+{find_word_start_position(input_text, word) + len(word)}c"
            input_text_widget.tag_add("real-word-error", start_index, end_index)
            
        if error_found:
            status_var.set(f"Errors Found: {len(misused_words)}")
        else:
            messagebox.showinfo("No Errors", "No spelling errors found in the input text.")
    else:
        messagebox.showwarning("Input Error", "Please enter a word or sentence to correct.")
    update_word_count()

# Helper function to find the start position of a word in the text
def find_word_start_position(text, word):
    match = re.search(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE)
    if match:
        return match.start()
    return 0

# Function to ignore a selected word
def ignore_word():
    try:
        selected_word = input_text_widget.get(tk.SEL_FIRST, tk.SEL_LAST).strip()
        if selected_word:
            ignored_words.add(selected_word.lower())
            messagebox.showinfo("Word Ignored", f"The word '{selected_word}' has been ignored.")
            status_var.set(f"Ignored Word: '{selected_word}'") #Status bar shows ignore words
            correct_input()  # Re-run correction to update the text highlighting
    except tk.TclError:
        messagebox.showwarning("Selection Error", "Please select a word to ignore.")

# Function to calculate similarity percentage between two words
def calculate_similarity(word1, word2):
    return int(SequenceMatcher(None, word1, word2).ratio() * 100)

# Show suggestions for a word (with similarity percentage)
def show_suggestions(event):
    try:
        index = input_text_widget.index(tk.CURRENT)
        word_start = input_text_widget.index(f"{index} wordstart")
        word_end = input_text_widget.index(f"{index} wordend")
        misspelled_word = input_text_widget.get(word_start, word_end).strip()

        if ('misspelled' in input_text_widget.tag_names(word_start) or
            'real-word-error' in input_text_widget.tag_names(word_start)):

            # Suggestions from edit distance model (non-word errors)
            edit_distance_suggestions = sorted(word_candidates(misspelled_word)[1] | word_candidates(misspelled_word)[2], key=P, reverse=True)
            
            # Find the masked index (convert position to a word index)
            text_before_misspelled_word = input_text_widget.get("1.0", word_start).strip()
            tokenized_text_before = tokenizer.tokenize(text_before_misspelled_word)
            masked_index = len(tokenized_text_before)

            # Suggestions from BERT (real-word errors)
            bert_suggestions = predict_masked_word(input_text_widget.get("1.0", tk.END).strip(), masked_index)

            # Combine all suggestions (limit to top 5 from each)
            all_suggestions = list(set(edit_distance_suggestions + bert_suggestions))

            # Calculate similarity for each suggestion
            suggestions_with_similarity = [(suggestion, calculate_similarity(misspelled_word, suggestion)) for suggestion in all_suggestions]

            # Sort suggestions by similarity in descending order and take top 5
            sorted_suggestions = sorted(suggestions_with_similarity, key=lambda x: x[1], reverse=True)[:5]

            # If no suggestions, return
            if not sorted_suggestions:
                return
            
            suggestions_menu = Menu(input_text_widget, tearoff=0)

            # Add each suggestion with its similarity percentage
            for suggestion, similarity in sorted_suggestions:
                suggestions_menu.add_command(
                    label=f"{suggestion} ({similarity}%)",
                    command=lambda suggestion=suggestion: replace_word(word_start, word_end, suggestion)
                )

            suggestions_menu.tk_popup(event.x_root, event.y_root)  # Popup at cursor location
            suggestions_menu.grab_release()
    except Exception as e:
        print(f"Error: {e}")

        
# Replace word with a suggestion
def replace_word(start, end, new_word):
    input_text_widget.delete(start, end)
    input_text_widget.insert(start, new_word, "corrected")
    status_var.set(f"Replaced with: '{new_word}'")

#------------------------- Tkinter GUI Setup ----------------------------------------
root = tk.Tk()
root.title("Spell Checker for Cancer")

# Input Text Widget
input_text_widget = tk.Text(root, width=65, height=15)
input_text_widget.grid(row=0, column=0, columnspan=3, padx=10, pady=5)
input_text_widget.tag_config("misspelled", foreground="red")
input_text_widget.tag_config("real-word-error", foreground="blue")
input_text_widget.tag_config("corrected", foreground="green", underline=True)
input_text_widget.bind("<Button-1>", show_suggestions)  # Right-click for suggestions
input_text_widget.bind("<KeyRelease>", update_word_count)  # Bind key release to update word count

# Word Count Label (below the text widget)
word_count_label = tk.Label(root, text="Word count: 0 / 500")
word_count_label.grid(row=1, column=2, columnspan=3, padx=10, pady=0)

# Correct button
correct_button = ttk.Button(root, text="Check", bootstyle=SUCCESS, command=lambda: [correct_input(), print("Checking...")])
correct_button.grid(row=2, column=0, padx=10, pady=5)

# Ignore button
ignore_button = ttk.Button(root, text="Ignore", bootstyle=WARNING, command=lambda: [ignore_word(), print("Ignoring...")])
ignore_button.grid(row=2, column=1, padx=10, pady=5)

# Exit button
exit_button = ttk.Button(root, text="Exit", bootstyle=DANGER, command=root.quit)
exit_button.grid(row=2, column=2,padx=10, pady=5)

# Word search and Listbox Frame
listbox_frame = tk.Frame(root)
listbox_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky='w')

# Search Word Label
search_label = tk.Label(listbox_frame, text="Search Word:")
search_label.pack(side=tk.LEFT, padx=(0, 5))

# Search Entry
word_search_entry = ttk.Entry(listbox_frame)
word_search_entry.pack(side=tk.LEFT, padx=5)

# Word List Label
word_list_label = tk.Label(listbox_frame, text="Word List:")
word_list_label.pack(side=tk.LEFT, padx=(20, 5))

# Word Listbox
word_listbox = tk.Listbox(listbox_frame, width=30, height=10)
word_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

#---------------------------- Dictionary ------------------------------------

# Update Listbox based on search
def update_listbox():
    search_term = word_search_entry.get().strip().lower()
    word_listbox.delete(0, tk.END)
    for word in sorted(word_counter):
        if search_term in word:
            word_listbox.insert(tk.END, word)

word_search_entry.bind("<KeyRelease>", lambda event: update_listbox())

# Function to scroll to the first word starting with the selected letter
def scroll_to_letter(event):
    selected_letter = alphabet_combobox.get()
    for index, word in enumerate(sorted(word_counter)):  # Loop through words in sorted order
        if word.startswith(selected_letter.lower()):
            word_listbox.see(index)  # Scroll to the first matching word
            word_listbox.selection_clear(0, tk.END)  # Clear any previous selections
            word_listbox.selection_set(index)  # Highlight the selected word
            word_listbox.activate(index)  # Set focus on the selected word
            break

# Frame for Alphabet Dropdown
alphabet_frame = tk.Frame(root)
alphabet_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=5)

# Alphabet Dropdown (Combobox)
alphabet_combobox = ttk.Combobox(alphabet_frame, values=list(string.ascii_uppercase), state="readonly", width=5)
alphabet_combobox.set("Alphabet")  # Default text
alphabet_combobox.pack(side=tk.LEFT, padx=5)

# Bind the dropdown selection to the scroll function
alphabet_combobox.bind("<<ComboboxSelected>>", scroll_to_letter)

# Example function to populate listbox
def populate_listbox():
    word_listbox.delete(0, tk.END)
    for word in sorted(word_counter):  # Sorting the words alphabetically
        word_listbox.insert(tk.END, word)

# Call to populate the listbox with words
populate_listbox()

# Function to delete a selected word from the word list and save changes
def delete_selected_word():
    selected = word_listbox.curselection()
    if selected:
        word = word_listbox.get(selected[0])
        word_listbox.delete(selected[0])  # Remove from Listbox
        if word in word_counter:
            del word_counter[word]  # Remove from word_counter
            
            # Save the updated word list back to the corpus file
            with open('cancer_new.txt', 'w', encoding='utf-8') as file:
                for word, count in word_counter.items():
                    file.write(f"{word} " * count)  # Write each word based on its frequency
            status_var.set(f"Deleted word: '{word}' and updated the corpus file.")
    else:
        messagebox.showwarning("Selection Error", "Please select a word to delete.")

# Allow selecting a word from the listbox to insert into the text widget
def insert_selected_word(event):
    selected = word_listbox.curselection()
    if selected:
        word = word_listbox.get(selected[0])
        input_text_widget.insert(tk.INSERT, word + " ")

# Insert Button
word_listbox.bind("<Double-Button-1>", insert_selected_word)

# Delete Button
delete_button = ttk.Button(listbox_frame, text="Delete", bootstyle=DANGER, command=delete_selected_word)
delete_button.pack(side=tk.LEFT, padx=10)

# Bind Return key to correct input
root.bind('<Return>', lambda event: correct_input())

# ------------------ Status Bar ------------------

# Create a StringVar to hold the status message
status_var = tk.StringVar()
status_var.set("Ready.")

# Status Bar Label
status_bar = tk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor='w')
status_bar.grid(row=5, column=0, columnspan=3, sticky='we')

# Configure grid weights to ensure the status bar stretches horizontally
root.grid_rowconfigure(3, weight=0)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)

# Run the Tkinter event loop
root.mainloop()
