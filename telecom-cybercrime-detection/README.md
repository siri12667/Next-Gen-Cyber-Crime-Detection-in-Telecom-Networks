# Next-Gen Cyber Crime Detection in Telecom Networks

Machine-learning system that analyses network-traffic records and flags suspicious
activity / cyber attacks in telecom networks. Built with **Logistic Regression** and
**Random Forest**, with a **Tkinter desktop GUI**.

Capstone project - Audisankara College of Engineering & Technology (JNTUA).

## Pipeline (matches the System Architecture slide)

`Upload dataset -> Preprocess -> Feature extraction -> Train/Test split -> Logistic Regression / Random Forest -> Prediction -> Comparison graph`

| Step | What it does |
|------|--------------|
| Preprocess | drops ID / leaky columns, removes duplicates, fills missing values, label-encodes `proto`, `service`, `state` |
| Data splitting | 80 / 20 stratified train-test split, scaling fitted on training data only |
| Logistic Regression | scaled features, reports precision / recall / F1 / accuracy + confusion matrix |
| RFC Classifier | 100-tree Random Forest, same metrics |
| Prediction | classifies test rows as *Cyber attack detected* / *No Cyber Attack detected* with the better model |
| Comparison Graph | bar chart of both models |

## Project structure

```
telecom-cybercrime-detection/
├── main.py                      # start the GUI
├── requirements.txt
├── requirements-dev.txt         # pytest
├── data/
│   └── sample_telecom_traffic.csv   # synthetic demo data (UNSW-NB15 layout)
├── src/
│   ├── gui.py                   # Tkinter interface
│   ├── ml_pipeline.py           # all ML logic
│   ├── run_cli.py               # run everything without the GUI
│   └── generate_sample_data.py  # regenerate demo data
├── tests/test_pipeline.py
├── models/  outputs/            # generated files (models, graphs)
└── .vscode/launch.json          # F5 run configs
```

## Run in VS Code (Windows)

```bash
# 1. open the folder in VS Code  (File > Open Folder)
# 2. open a terminal (Ctrl + `) and create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Linux/Mac: source .venv/bin/activate

# 3. install packages
pip install -r requirements.txt

# 4. start the app
python main.py
```

In the GUI click the buttons in order:
**Upload Dataset -> Preprocess Dataset -> Data splitting -> Logistic Regresssion -> RFC Classifier -> Prediction -> Comparison Graph**.

No GUI? Run `python -m src.run_cli` to see the full output in the terminal and save
`outputs/comparison_graph.png`.

> Linux only: if Tkinter is missing, `sudo apt install python3-tk`.

## Using the real dataset (recommended for submission)

The bundled CSV is **synthetic demo data** so the project works immediately. The review
slides' screenshot (35,069 test rows) matches the **UNSW-NB15** training set:

1. Download `UNSW_NB15_training-set.csv` from the UNSW Canberra Cyber page / Kaggle.
2. Put it in `data/` and choose it with **Upload Dataset**.

Requirement: the CSV needs a `label` column (0 = normal, 1 = attack). `id` and
`attack_cat` are dropped automatically. Report the metrics you get on the real data in
your final report - results on the synthetic file are only for demonstration.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: telecom cyber crime detection"
git branch -M main
# create an EMPTY repo on github.com first (no README), then:
git remote add origin https://github.com/<your-username>/telecom-cybercrime-detection.git
git push -u origin main
```

## Future scope
Real-time traffic analysis, deep-learning models, cloud / big-data scaling, automated response.
