"""Tkinter desktop GUI - matches the screenshots in the project review PPT."""
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from src.ml_pipeline import CyberCrimePipeline


class CyberAttackApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.pipe = CyberCrimePipeline()
        root.title("CYBER ATTACK DETECTION")
        root.geometry("1100x650")
        root.configure(bg="#c8e0ff")

        title = tk.Label(
            root, text="DETECTION OF CYBER ATTACKS IN NETWORK USING MACHINE LEARNING TECHNIQUES",
            font=("Times New Roman", 15, "bold"), fg="navy", bg="#808080", pady=14)
        title.pack(fill="x")

        frame = tk.Frame(root, bg="#c8e0ff")
        frame.pack(pady=20)
        buttons = [
            ("Upload Dataset", self.upload), ("Preprocess Dataset", self.preprocess),
            ("Data splitting", self.split), ("Logistic Regresssion", self.run_lr),
            ("RFC Classifier", self.run_rf), ("Prediction", self.predict),
            ("Comparison Graph", self.compare), ("Exit", root.destroy),
        ]
        for i, (text, cmd) in enumerate(buttons):
            tk.Button(frame, text=text, command=cmd, width=22,
                      font=("Times New Roman", 12, "bold")).grid(
                row=i // 2, column=i % 2, padx=60, pady=8)

        self.out = ScrolledText(root, width=130, height=20, font=("Times New Roman", 11))
        self.out.pack(padx=20, pady=10, fill="both", expand=True)
        self.status = tk.Label(root, text="Ready", anchor="w", bg="#c8e0ff")
        self.status.pack(fill="x", padx=20)

    # helpers -----------------------------------------------------------
    def show(self, text: str, clear: bool = True):
        if clear:
            self.out.delete("1.0", tk.END)
        self.out.insert(tk.END, text + "\n")
        self.out.see(tk.END)

    def run(self, label, func):
        self.status.config(text=f"{label} ...")
        self.root.update_idletasks()
        try:
            result = func()
            self.status.config(text=f"{label} - done")
            return result
        except Exception as e:  # show friendly error instead of crashing
            self.status.config(text=f"{label} - failed")
            messagebox.showerror("Error", str(e))

    # button handlers ---------------------------------------------------
    def upload(self):
        path = filedialog.askopenfilename(title="Select dataset (CSV)",
                                          filetypes=[("CSV files", "*.csv")])
        if path:
            msg = self.run("Loading dataset", lambda: self.pipe.load_dataset(path))
            if msg:
                self.show(msg)

    def preprocess(self):
        msg = self.run("Preprocessing", self.pipe.preprocess)
        if msg:
            self.show(msg)

    def split(self):
        msg = self.run("Splitting data", self.pipe.split)
        if msg:
            self.show(msg)

    def run_lr(self):
        res = self.run("Training Logistic Regression", self.pipe.train_logistic_regression)
        if res:
            self.show(res.summary())

    def run_rf(self):
        res = self.run("Training Random Forest", self.pipe.train_random_forest)
        if res:
            self.show(res.summary())

    def predict(self):
        msg = self.run("Predicting", lambda: self.pipe.predict_rows(10))
        if msg:
            self.show(msg)

    def compare(self):
        fig = self.run("Building graph", self.pipe.comparison_figure)
        if not fig:
            return
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        win = tk.Toplevel(self.root)
        win.title("Comparison Graph")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


def launch():
    root = tk.Tk()
    CyberAttackApp(root)
    root.mainloop()
