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
    def __init__(self, master, data, options):
        super().__init__(master)
        self.title("加速度图像窗口")
        self.data = data
        self.options = options

        self.figure, self.axes = plt.subplots(len(options['axes']), 1, figsize=(10, 6), sharex=True, constrained_layout=True)
        if len(options['axes']) == 1:
            self.axes = [self.axes]

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack()

        self.plot_data()

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
        else:
            return data

    def plot_data(self):
        df = self.data
        time = df["Time (s)"]
        start_time = self.options['start']
        end_time = self.options['end']
        mask = (time >= start_time) & (time <= end_time)
        time = time[mask]

        selected_axes = self.options['axes']
        color = self.options['color']
        linewidth = self.options['linewidth']
        filt = self.options['filter']

        for i, axis in enumerate(selected_axes):
            ax = self.axes[i]
            raw_data = df[f"Linear Acceleration {axis} (m/s^2)"][mask]
            filtered = self.apply_filter(raw_data, filt)
            ax.plot(time, filtered, color=color, linewidth=linewidth)
            ax.set_title(f"加速度 {axis.upper()} 轴")
            ax.set_ylabel("加速度 (m/s²)")
            ax.grid(True)

        self.axes[-1].set_xlabel("时间 (s)")
        self.canvas.draw()

    def save_figure(self):
        save_dir = filedialog.askdirectory(title="选择保存目录")
        if not save_dir:
            return

        base_name = "acceleration_plot"
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
        self.title("加速度数据分析设置面板")

        self.color_options = ["blue", "red", "green", "orange", "purple", "cyan", "magenta", "black", "brown", "gray"]
        self.filter_options = ["不滤波", "Savitzky-Golay", "Butterworth", "Gaussian"]

        self.data = None
        self.file_path = None
        self.build_ui()

    def build_ui(self):
        tk.Button(self, text="选择CSV文件", command=self.load_file).pack(pady=5)

        tk.Label(self, text="线条颜色").pack()
        self.color_var = tk.StringVar(value=self.color_options[0])
        ttk.Combobox(self, textvariable=self.color_var, values=self.color_options).pack()

        tk.Label(self, text="线条粗细").pack()
        self.linewidth_var = tk.DoubleVar(value=0.5)
        tk.Scale(self, from_=0.1, to=3.0, resolution=0.1, variable=self.linewidth_var, orient=tk.HORIZONTAL).pack()

        tk.Label(self, text="滤波方式").pack()
        self.filter_var = tk.StringVar(value=self.filter_options[0])
        ttk.Combobox(self, textvariable=self.filter_var, values=self.filter_options).pack()

        tk.Label(self, text="选择要分析的轴").pack()
        self.axis_vars = {axis: tk.BooleanVar(value=True) for axis in ['x', 'y', 'z']}
        for axis in self.axis_vars:
            tk.Checkbutton(self, text=axis.upper(), variable=self.axis_vars[axis]).pack()

        tk.Label(self, text="输入时间范围 (单位: 秒，留空为全选)").pack()
        frame = tk.Frame(self)
        frame.pack()
        tk.Label(frame, text="起始时间:").grid(row=0, column=0)
        self.start_time_entry = tk.Entry(frame, width=10)
        self.start_time_entry.grid(row=0, column=1)
        tk.Label(frame, text="结束时间:").grid(row=0, column=2)
        self.end_time_entry = tk.Entry(frame, width=10)
        self.end_time_entry.grid(row=0, column=3)

        self.redraw_btn = tk.Button(self, text="重新生成图像", command=self.open_plot_window)
        self.redraw_btn.pack(pady=5)

    def load_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.file_path:
            df = pd.read_csv(self.file_path)
            required_cols = ["Time (s)", "Linear Acceleration x (m/s^2)",
                             "Linear Acceleration y (m/s^2)", "Linear Acceleration z (m/s^2)"]
            if not all(col in df.columns for col in required_cols):
                messagebox.showerror("错误", "CSV 文件缺少必要的列")
                return

            self.data = df
            messagebox.showinfo("成功", f"已加载数据文件: {self.file_path}")
            self.open_plot_window()

    def open_plot_window(self):
        if self.data is None:
            messagebox.showerror("错误", "请先加载CSV文件")
            return

        time = self.data["Time (s)"]
        try:
            start = float(self.start_time_entry.get()) if self.start_time_entry.get() else time.min()
            end = float(self.end_time_entry.get()) if self.end_time_entry.get() else time.max()
        except ValueError:
            messagebox.showerror("错误", "请输入有效的时间范围")
            return

        options = {
            'color': self.color_var.get(),
            'linewidth': self.linewidth_var.get(),
            'filter': self.filter_var.get(),
            'axes': [k for k, v in self.axis_vars.items() if v.get()],
            'start': start,
            'end': end
        }

        PlotWindow(self, self.data, options)


if __name__ == "__main__":
    app = ControlPanel()
    app.mainloop()
