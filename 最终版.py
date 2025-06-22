# ✅ 集成更新：仅显示用户勾选的轴（x/y/z）
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.signal.windows import gaussian
import os

class PlotWindow(tk.Toplevel):
    def __init__(self, master, datasets):
        super().__init__(master)
        self.title("叠加对比图像窗口")
        self.datasets = datasets

        self.figure = plt.Figure(figsize=(10, 8), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack()

        self.plot_all_data()

        save_btn = tk.Button(self, text="保存图像", command=self.save_figure)
        save_btn.pack(pady=5)

    def apply_filter(self, data, method):
        if method == "不滤波":
            return data
        elif method == "Savitzky-Golay":
            return savgol_filter(data, window_length=11, polyorder=2)
        elif method == "Butterworth":
            b, a = butter(N=2, Wn=0.1)
            return filtfilt(b, a, data)
        elif method == "Gaussian":
            kernel = gaussian(11, std=2)
            kernel /= kernel.sum()
            return np.convolve(data, kernel, mode='same')
        return data

    def plot_all_data(self):
        all_axes = set()
        for dataset in self.datasets:
            all_axes.update(dataset['options']['axes'])

        axis_order = {'x': 0, 'y': 1, 'z': 2}
        selected_axes = sorted(list(all_axes), key=lambda a: axis_order[a])

        self.figure.clf()
        self.axes = self.figure.subplots(len(selected_axes), 1, sharex=True)
        if len(selected_axes) == 1:
            self.axes = [self.axes]

        for dataset in self.datasets:
            df = dataset['data'].copy()
            opt = dataset['options']
            df["Time (s)"] -= df["Time (s)"].iloc[0]
            time = df["Time (s)"]
            mask = (time >= opt['start']) & (time <= opt['end'])
            time = time[mask]

            for i, axis in enumerate(selected_axes):
                if axis in opt['axes']:
                    raw = df[f"Linear Acceleration {axis} (m/s^2)"][mask]
                    filtered = self.apply_filter(raw, opt['filter'])
                    self.axes[i].plot(time, filtered, label=opt['label'], color=opt['color'], linewidth=opt['linewidth'])
                    self.axes[i].set_ylabel(f"加速度 {axis.upper()} (m/s²)")
                    self.axes[i].grid(True)

        self.axes[-1].set_xlabel("时间 (s)")
        for ax in self.axes:
            ax.legend()
        self.canvas.draw()

    def save_figure(self):
        save_dir = filedialog.askdirectory(title="选择保存目录")
        if not save_dir:
            return
        base_name = "combined_acceleration_plot"
        file_path = os.path.join(save_dir, f"{base_name}.png")
        i = 1
        while os.path.exists(file_path):
            file_path = os.path.join(save_dir, f"{base_name}_{i}.png")
            i += 1
        self.figure.savefig(file_path, dpi=300)
        messagebox.showinfo("保存成功", f"图像已保存至 {file_path}")

class ControlPanel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("多组加速度数据分析")
        self.datasets = []
        self.build_ui()

    def build_ui(self):
        self.frames = []
        for i in range(5):
            frame = self.create_file_frame(i + 1)
            self.frames.append(frame)
        tk.Button(self, text="生成叠加对比图", command=self.plot_all).pack(pady=10)

    def create_file_frame(self, index):
        frame = tk.LabelFrame(self, text=f"数据组 {index}", padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=5)

        file_var = tk.StringVar()
        label_var = tk.StringVar(value=f"组{index}")
        color_var = tk.StringVar(value="blue")
        linewidth_var = tk.DoubleVar(value=0.5)
        filter_var = tk.StringVar(value="不滤波")
        axis_vars = {axis: tk.BooleanVar(value=True) for axis in ['x', 'y', 'z']}
        start_var = tk.StringVar()
        end_var = tk.StringVar()

        tk.Button(frame, text="选择CSV文件", command=lambda: self.load_file(file_var)).grid(row=0, column=0, columnspan=2)
        tk.Label(frame, textvariable=file_var, wraplength=300).grid(row=1, column=0, columnspan=2)

        ttk.Label(frame, text="标签").grid(row=2, column=0)
        ttk.Entry(frame, textvariable=label_var).grid(row=2, column=1)

        ttk.Label(frame, text="颜色").grid(row=3, column=0)
        ttk.Combobox(frame, textvariable=color_var, values=["blue", "red", "green", "orange", "purple", "cyan", "black", "gray"]).grid(row=3, column=1)

        ttk.Label(frame, text="粗细").grid(row=4, column=0)
        ttk.Scale(frame, from_=0.1, to=3.0, variable=linewidth_var, orient=tk.HORIZONTAL).grid(row=4, column=1)

        ttk.Label(frame, text="滤波").grid(row=5, column=0)
        ttk.Combobox(frame, textvariable=filter_var, values=["不滤波", "Savitzky-Golay", "Butterworth", "Gaussian"]).grid(row=5, column=1)

        axis_frame = tk.Frame(frame)
        axis_frame.grid(row=6, column=0, columnspan=2)
        for i, axis in enumerate(['x', 'y', 'z']):
            tk.Checkbutton(axis_frame, text=axis.upper(), variable=axis_vars[axis]).pack(side=tk.LEFT)

        time_frame = tk.Frame(frame)
        time_frame.grid(row=7, column=0, columnspan=2)
        tk.Label(time_frame, text="起始时间").pack(side=tk.LEFT)
        tk.Entry(time_frame, textvariable=start_var, width=10).pack(side=tk.LEFT)
        tk.Label(time_frame, text="结束时间").pack(side=tk.LEFT)
        tk.Entry(time_frame, textvariable=end_var, width=10).pack(side=tk.LEFT)

        frame.vars = {
            'file': file_var, 'label': label_var, 'color': color_var, 'linewidth': linewidth_var,
            'filter': filter_var, 'axes': axis_vars, 'start': start_var, 'end': end_var
        }
        return frame

    def load_file(self, var):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            var.set(file_path)

    def plot_all(self):
        self.datasets.clear()
        for frame in self.frames:
            vars = frame.vars
            path = vars['file'].get()
            if not path:
                continue
            try:
                df = pd.read_csv(path)
                df["Time (s)"] -= df["Time (s)"].iloc[0]
                time = df["Time (s)"]
                start = float(vars['start'].get()) if vars['start'].get() else time.min()
                end = float(vars['end'].get()) if vars['end'].get() else time.max()
                options = {
                    'label': vars['label'].get(),
                    'color': vars['color'].get(),
                    'linewidth': vars['linewidth'].get(),
                    'filter': vars['filter'].get(),
                    'axes': [k for k, v in vars['axes'].items() if v.get()],
                    'start': start,
                    'end': end
                }
                if options['axes']:  # 至少选中一个轴
                    self.datasets.append({'data': df, 'options': options})
            except Exception as e:
                messagebox.showerror("错误", f"无法加载 {path}: {e}")
        if self.datasets:
            PlotWindow(self, self.datasets)

if __name__ == "__main__":
    ControlPanel().mainloop()
