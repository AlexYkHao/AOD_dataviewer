import tkinter as tk
from tkinter import filedialog, ttk
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageTk
import os
from scipy.signal import savgol_filter  # Import for Savitzky-Golay filter

class TDMSViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("TDMS Data Viewer")
        self.root.geometry("1600x800")

        # Create main frames
        self.left_frame = ttk.Frame(root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10)

        self.right_frame = ttk.Frame(root)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame components
        self.load_button = ttk.Button(self.left_frame, text="Load Data", command=self.load_data, style='Custom.TButton')
        self.load_button.pack(pady=10)

        # Add folder path label
        self.folder_label = ttk.Label(self.left_frame, text="Current folder:", wraplength=380)
        self.folder_label.pack(pady=5)
        self.path_label = ttk.Label(self.left_frame, text=os.path.dirname(os.path.abspath(__file__)), 
                                  wraplength=380, font=('Arial', 8))
        self.path_label.pack(pady=5)

        self.zoom_button = ttk.Button(self.left_frame, text="Zoom In", command=self.enable_zoom, style='Custom.TButton')
        self.zoom_button.pack(pady=10)
        
        self.reset_zoom_button = ttk.Button(self.left_frame, text="Reset Zoom", command=self.reset_zoom, style='Custom.TButton')
        self.reset_zoom_button.pack(pady=10)

        # Add smooth controls to left frame
        self.smooth_frame = ttk.Frame(self.left_frame)
        self.smooth_frame.pack(pady=10)
        
        # Filter type selection
        self.filter_type_frame = ttk.Frame(self.left_frame)
        self.filter_type_frame.pack(pady=5)
        
        self.filter_type_var = tk.StringVar(value="moving_avg")
        self.moving_avg_radio = ttk.Radiobutton(self.filter_type_frame, text="Moving Average", 
                                               variable=self.filter_type_var, value="moving_avg",
                                               command=self.toggle_filter_options)
        self.moving_avg_radio.pack(side=tk.LEFT, padx=5)
        
        self.savgol_radio = ttk.Radiobutton(self.filter_type_frame, text="Savitzky-Golay", 
                                          variable=self.filter_type_var, value="savgol",
                                          command=self.toggle_filter_options)
        self.savgol_radio.pack(side=tk.LEFT, padx=5)
        
        # Window size input (common for both filters)
        self.window_label = ttk.Label(self.smooth_frame, text="Window Size:")
        self.window_label.pack(side=tk.LEFT, padx=5)
        
        self.window_size_var = tk.StringVar(value="5")
        self.window_entry = ttk.Entry(self.smooth_frame, textvariable=self.window_size_var, width=8)
        self.window_entry.pack(side=tk.LEFT, padx=5)
        
        # Order input (for Savitzky-Golay only)
        self.order_frame = ttk.Frame(self.left_frame)
        self.order_frame.pack(pady=5)
        
        self.order_label = ttk.Label(self.order_frame, text="Polynomial Order:")
        self.order_label.pack(side=tk.LEFT, padx=5)
        
        self.order_var = tk.StringVar(value="2")
        self.order_entry = ttk.Entry(self.order_frame, textvariable=self.order_var, width=8)
        self.order_entry.pack(side=tk.LEFT, padx=5)
        
        # Initially hide the order input
        self.order_frame.pack_forget()
        
        self.smooth_button = ttk.Button(self.left_frame, text="Smooth", 
                                      command=self.smooth_data, 
                                      style='Custom.TButton')
        self.smooth_button.pack(pady=5)

        # Add navigation buttons
        self.nav_frame = ttk.Frame(self.left_frame)
        self.nav_frame.pack(pady=10)
        
        self.left_button = ttk.Button(self.nav_frame, text="←", 
                                    command=self.shift_left, 
                                    style='Custom.TButton')
        self.left_button.pack(side=tk.LEFT, padx=5)
        
        self.right_button = ttk.Button(self.nav_frame, text="→", 
                                     command=self.shift_right, 
                                     style='Custom.TButton')
        self.right_button.pack(side=tk.LEFT, padx=5)

        # Create custom style
        style = ttk.Style()
        style.configure('Custom.TButton', 
                       background='blue',
                       foreground='black',
                       font=('Arial', 12, 'bold'),
                       padding=10)
        
        # Configure Combobox colors and font
        style.configure('TCombobox', 
                       selectbackground='#0078D7',
                       selectforeground='black',
                       fieldbackground='black',
                       background='white')
        
        style.map('TCombobox',
                 fieldbackground=[('readonly', 'black')],
                 selectbackground=[('readonly', '#0078D7')],
                 selectforeground=[('readonly', 'black')])

        # Image canvas (left frame)
        self.image_canvas = tk.Canvas(self.left_frame, width=400, height=400)
        self.image_canvas.pack(pady=10)

        # Right frame components
        self.plot_frame = ttk.Frame(self.right_frame)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create dropdown frame
        self.dropdown_frame = ttk.Frame(self.right_frame, width=150)  # Fixed width for dropdown area
        self.dropdown_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        self.dropdown_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Create figure and canvas first
        self.fig = Figure(figsize=(8, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        
        # Add initial text
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Please load data using the "Load Data" button', 
                ha='center', va='center', fontsize=12)
        ax.set_axis_off()
        
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        
        # Create dropdown menus
        self.channel_vars = []
        self.channel_dropdowns = []
        
        # Add spacing for sync plot
        ttk.Label(self.dropdown_frame, text="").pack(pady=65)
        
        for i in range(5):
            var = tk.StringVar()
            self.channel_vars.append(var)
            dropdown = ttk.Combobox(self.dropdown_frame, textvariable=var, state='readonly', width=15)
            dropdown.pack(pady=45)  # Adjust this value to match plot spacing
            dropdown.bind('<<ComboboxSelected>>', lambda e, idx=i: self.on_channel_select(idx))
            self.channel_dropdowns.append(dropdown)

        # Initialize variables
        self.analog_file = None
        self.pmt_file = None
        self.current_plots = []
        self.original_data = None
        self.sync = None
        self.sync_time = None
        self.pmt_data = None
        self.pmt_time = None
        
        # Add root directory tracker
        self.root_directory = os.path.dirname(os.path.abspath(__file__))

        # Add zoom variables
        self.zoom_enabled = False
        self.zoom_rect = None
        self.zoom_press = None
        self.zoom_background = None
        self.zoom_ax = None

        # Add time window tracker
        self.shown_time_window = None  # Will store (x_min, x_max)

        # Add smoothing state tracker
        self.is_smoothed = False
        self.current_window_size = None
        self.current_order = None
        self.current_filter_type = "moving_avg"

    def load_data(self):
        initial_dir = self.root_directory if self.root_directory else "/"
        folder_path = filedialog.askdirectory(title="Select Data Folder", 
                                            initialdir=initial_dir)
        
        if folder_path:
            # Update folder path label
            self.path_label.configure(text=folder_path)
            
            # Set root directory as parent of selected folder
            self.root_directory = os.path.dirname(folder_path)
            
            # Reset smoothing state
            self.is_smoothed = False
            self.current_window_size = None
            self.current_order = None
            self.current_filter_type = "moving_avg"
            
            # Reset zoom window
            self.shown_time_window = None
            
            # Look for specific TDMS files            
            for f in os.listdir(folder_path):
                if f.endswith('AnalogIN.tdms'):
                    self.analog_file = os.path.join(folder_path, f)
                elif f.endswith('pmt1.tdms_index'):
                    # Remove '_index' from filename to get the actual tdms file
                    self.pmt_file = os.path.join(folder_path, f.replace('_index', ''))
                
            if self.analog_file:
                self.sync, self.sync_time = self.get_analog_data(self.analog_file)
            
            if self.pmt_file:
                self.pmt_data, self.pmt_time = self.get_pmt_data(self.pmt_file)
                if not self.analog_file:  # if no sync signal, just show 0s with the same length as the PMT data
                    self.sync_time = self.pmt_time  
                    self.sync = np.zeros(len(self.pmt_time))
                
                # Update dropdown menus with PMT channels
                if self.pmt_data:
                    channel_list = list(self.pmt_data.keys())
                    # Calculate width based on longest channel name
                    max_width = max(len(ch) for ch in channel_list)
                    for dropdown in self.channel_dropdowns:
                        dropdown['values'] = channel_list
                        dropdown.set(channel_list[0] if channel_list else '')
                        dropdown.configure(width=max_width + 2)  # Add padding
                
                self.plot_data()

            # Look for PNG files
            png_files = [f for f in os.listdir(folder_path) if f.endswith('pmt1.png')]
            if png_files:
                png_path = os.path.join(folder_path, png_files[0])
                self.load_image(png_path)

    def load_image(self, image_path):
        # Load and display PNG image
        image = Image.open(image_path)
        image = image.resize((400, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.image_canvas.image = photo

    def plot_data(self):
        self.fig.clear()
        self.current_plots = []
        self.original_data = []
        
        # Plot sync signal (first subplot)
        ax = self.fig.add_subplot(6, 1, 1)
        try:
            if self.sync is not None:
                line, = ax.plot(self.sync_time, self.sync, linewidth=0.5)
                self.current_plots.append(line)
                self.original_data.append(self.sync)
                ax.set_title('Flicker signal')
                ax.grid(False)
                
                # Initialize time window if not set
                if self.shown_time_window is None:
                    self.shown_time_window = (min(self.sync_time), max(self.sync_time))
                ax.set_xlim(self.shown_time_window)
            else:
                raise IndexError
        except (IndexError, AttributeError):
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')

        # Plot PMT channels
        channel_list = []
        if self.pmt_data:
            channel_list = list(self.pmt_data.keys())
            # Update dropdown values first
            for dropdown in self.channel_dropdowns:
                dropdown['values'] = channel_list
            
        for i in range(5):
            ax = self.fig.add_subplot(6, 1, i+2)
            try:
                if self.pmt_data is not None and channel_list:
                    # Always set initial channel cycling through the list
                    channel_idx = i % len(channel_list)
                    selected_channel = channel_list[channel_idx]
                    self.channel_vars[i].set(selected_channel)
                    self.channel_dropdowns[i].set(selected_channel)
                    
                    data = self.pmt_data[selected_channel]
                    line, = ax.plot(self.pmt_time, data, linewidth=0.5)
                    self.current_plots.append(line)
                    self.original_data.append(data)
                    ax.set_title(selected_channel)
                    ax.grid(False)
                    ax.set_xlim(self.shown_time_window)  # Apply consistent time window
                else:
                    raise IndexError
            except (IndexError, AttributeError):
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')

        self.fig.tight_layout()
        self.canvas.draw()

    def toggle_filter_options(self):
        filter_type = self.filter_type_var.get()
        if filter_type == "savgol":
            self.order_frame.pack(after=self.smooth_frame, pady=5)
        else:
            self.order_frame.pack_forget()
    
    def smooth_data(self):
        if self.pmt_data is None or not self.current_plots:
            return

        # Get the filter type
        filter_type = self.filter_type_var.get()
        self.current_filter_type = filter_type

        try:
            window_size = int(self.window_size_var.get())
            # Ensure window size is odd for Savitzky-Golay
            if filter_type == "savgol" and window_size % 2 == 0:
                window_size += 1
                self.window_size_var.set(str(window_size))
        except ValueError:
            window_size = 5
            self.window_size_var.set("5")

        self.is_smoothed = True
        self.current_window_size = window_size
        
        # Get the polynomial order for Savitzky-Golay
        if filter_type == "savgol":
            try:
                order = int(self.order_var.get())
                # Ensure order is less than window size
                if order >= window_size:
                    order = window_size - 1
                    self.order_var.set(str(order))
            except ValueError:
                order = 2
                self.order_var.set("2")
            self.current_order = order
        
        for i, line in enumerate(self.current_plots):
            data = self.original_data[i]
            if filter_type == "moving_avg":
                smoothed_data = self.moving_average(data, window_size)
            else:  # savgol
                smoothed_data = self.savitzky_golay(data, window_size, order)
            line.set_ydata(smoothed_data)

        self.canvas.draw()

    @staticmethod
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')
    
    @staticmethod
    def savitzky_golay(data, window_size, order):
        try:
            # Ensure window_size is odd and order is less than window_size
            if window_size % 2 == 0:
                window_size += 1
            if order >= window_size:
                order = window_size - 1
                
            return savgol_filter(data, window_size, order)
        except Exception as e:
            print(f"Error applying Savitzky-Golay filter: {e}")
            # Fall back to moving average if there's an error
            return np.convolve(data, np.ones(window_size)/window_size, mode='same')

    @staticmethod
    def get_analog_data(analog_file):
        with TdmsFile.open(analog_file) as sync_file:
            group = sync_file['external analogIN']
            sync_ = group['AnalogGPIOBoard/ai0']
            sync = sync_[:]
        sync_time = np.arange(0, len(sync))
        sync_time = sync_time / 10000
        return sync, sync_time

    @staticmethod
    def get_pmt_data(pmt_file):
        channel_dic = {}
        with TdmsFile.open(pmt_file) as tdms_file:
            green_group = tdms_file['PMT1']
            green_sum = None
            for channel in green_group.channels():
                channel_name = channel.name
                if 'time' in channel_name:
                    channel_time = channel[:].copy()
                else:
                    channel_dic[channel_name + '_green'] = channel[:]
                    if green_sum is None:
                        green_sum = channel[:].copy()
                    else:
                        green_sum += channel[:].copy()
            channel_dic['green_sum'] = green_sum.copy()
            if 'PMT2' in tdms_file:
                red_group = tdms_file['PMT2']
                red_sum = None
                for channel in red_group.channels():
                    channel_name = channel.name
                    if 'time' not in channel_name:
                        channel_dic[channel_name+ '_red'] = channel[:]
                        if red_sum is None:
                            red_sum = channel[:].copy()
                        else:
                            red_sum += channel[:].copy()
                channel_dic['red_sum'] = red_sum.copy()
        acquisision_rate = 1000/np.gradient(channel_time).mean()
        channel_time = np.arange(0, len(list(channel_dic.values())[0]))/acquisision_rate
        return channel_dic, channel_time

    def on_channel_select(self, subplot_idx):
        if self.pmt_data is None:
            return
            
        selected_channel = self.channel_vars[subplot_idx].get()
        self.channel_dropdowns[subplot_idx].set(selected_channel)
        self.channel_dropdowns[subplot_idx].configure(foreground='black')
        
        if selected_channel:
            ax = self.fig.get_axes()[subplot_idx + 1]
            ax.clear()
            data = self.pmt_data[selected_channel]
            
            # Update original data for smoothing
            self.original_data[subplot_idx + 1] = data.copy()
            
            # Apply smoothing if active
            if self.is_smoothed:
                if self.current_filter_type == "moving_avg":
                    data = self.moving_average(data, self.current_window_size)
                else:  # savgol
                    data = self.savitzky_golay(data, self.current_window_size, self.current_order)
            
            # Update plot with linewidth
            line, = ax.plot(self.pmt_time, data, linewidth=0.5)
            self.current_plots[subplot_idx + 1] = line
            ax.set_title(selected_channel)
            ax.grid(False)
            ax.set_xlim(self.shown_time_window)
            
            self.fig.tight_layout()
            self.canvas.draw()

    def enable_zoom(self):
        if not self.zoom_enabled:
            self.zoom_enabled = True
            self.zoom_button.configure(text="Disable Zoom")
            self.canvas.mpl_connect('button_press_event', self.on_zoom_press)
            self.canvas.mpl_connect('button_release_event', self.on_zoom_release)
            self.canvas.mpl_connect('motion_notify_event', self.on_zoom_motion)
        else:
            self.zoom_enabled = False
            self.zoom_button.configure(text="Zoom In")

    def on_zoom_press(self, event):
        if not self.zoom_enabled or event.button != 1:
            return
        
        self.zoom_press = (event.xdata, event.ydata)
        self.zoom_ax = event.inaxes
        if self.zoom_ax is None:
            return
            
        # Store background for blitting
        self.zoom_background = self.canvas.copy_from_bbox(self.fig.bbox)
        
        # Create rectangle
        self.zoom_rect = plt.Rectangle(
            (event.xdata, event.ydata),
            0, 0,
            facecolor='gray',
            alpha=0.3
        )
        self.zoom_ax.add_patch(self.zoom_rect)

    def on_zoom_motion(self, event):
        if not self.zoom_enabled or self.zoom_press is None:
            return
            
        if event.inaxes is None:
            return
            
        # Restore background
        self.canvas.restore_region(self.zoom_background)
        
        # Update rectangle
        width = event.xdata - self.zoom_press[0]
        height = event.ydata - self.zoom_press[1]
        self.zoom_rect.set_width(width)
        self.zoom_rect.set_height(height)
        
        # Redraw
        self.zoom_ax.draw_artist(self.zoom_rect)
        self.canvas.blit(self.fig.bbox)

    def on_zoom_release(self, event):
        if not self.zoom_enabled or self.zoom_press is None:
            return
            
        if event.inaxes is None:
            return
            
        x_start = min(self.zoom_press[0], event.xdata)
        x_end = max(self.zoom_press[0], event.xdata)
        y_start = min(self.zoom_press[1], event.ydata)
        y_end = max(self.zoom_press[1], event.ydata)
        
        # Update time window for all subplots
        self.shown_time_window = (x_start, x_end)
        
        # Apply zoom
        for ax in self.fig.get_axes():
            # Apply x zoom to all subplots
            ax.set_xlim(self.shown_time_window)
            
            # Apply y zoom only to the selected subplot
            if ax == self.zoom_ax:
                ax.set_ylim(y_start, y_end)
        
        # Clean up
        if self.zoom_rect:
            self.zoom_rect.remove()
        self.zoom_rect = None
        self.zoom_press = None
        self.zoom_background = None
        self.zoom_ax = None
        
        self.canvas.draw()

    def reset_zoom(self):
        if hasattr(self, 'sync_time') and self.sync_time is not None:
            # Update time window to full range
            self.shown_time_window = (min(self.sync_time), max(self.sync_time))
            
            for i, ax in enumerate(self.fig.get_axes()):
                # Reset x limits for all plots
                ax.set_xlim(self.shown_time_window)
                
                # Reset y limits based on data
                if i < len(self.original_data):
                    data = self.original_data[i]
                    margin = (max(data) - min(data)) * 0.05  # 5% margin
                    ax.set_ylim(min(data) - margin, max(data) + margin)
            
            self.canvas.draw()

    def shift_left(self):
        if self.shown_time_window is None:
            return
            
        window_width = self.shown_time_window[1] - self.shown_time_window[0]
        shift_amount = window_width * 0.1  # Shift by 10% of window width
        
        new_start = self.shown_time_window[0] - shift_amount
        new_end = self.shown_time_window[1] - shift_amount
        
        # Ensure we don't go before the start of data
        if new_start >= min(self.sync_time):
            self.shown_time_window = (new_start, new_end)
            
            for ax in self.fig.get_axes():
                ax.set_xlim(self.shown_time_window)
            
            self.canvas.draw()

    def shift_right(self):
        if self.shown_time_window is None:
            return
            
        window_width = self.shown_time_window[1] - self.shown_time_window[0]
        shift_amount = window_width * 0.1  # Shift by 10% of window width
        
        new_start = self.shown_time_window[0] + shift_amount
        new_end = self.shown_time_window[1] + shift_amount
        
        # Ensure we don't go past the end of data
        if new_end <= max(self.sync_time):
            self.shown_time_window = (new_start, new_end)
            
            for ax in self.fig.get_axes():
                ax.set_xlim(self.shown_time_window)
            
            self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = TDMSViewer(root)
    root.mainloop() 