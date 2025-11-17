# Model 5 Implementation
Core architecture: **TF-IDF** embedding model with **Naive Bayes** classifier model<br>
Author: *Marcus KWAN TH*<br>

## Prerequisite (Same as Getting Started):
1. Cloned this GitHub repository into the local PC
2. Please run the following command to start playing with this project.

```bash
cd twitter-sentiment-analysis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Before running the code, please ensure the file `util.zip` is present in the root directory. If not, zip the util folder using the following command: 
```bash
cd twitter-sentiment-analysis
zip -r util.zip util
```

## Usage: TF-IDF with Naive Bayes
1. Run `tfidf_nb_program.ipynb` from the current folder.
2. Follow the instruction inside the Jupyter Notebook to play around with it.

## Optional: TF-IDF Word-embedding Exporter
1. Run `tfidf_word_embed.ipynb` from the current folder.
2. Follow the instruction inside the Jupyter Notebook to play around with it.
3. You may check the `output` folder for the exported TF-IDF vectors after running the last cell of the file.