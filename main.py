import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pathlib import Path
import logging
from typing import Optional, Tuple
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """A class to analyze CPU and GPU performance data from MSI Afterburner logs."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df_final = None
        self.cpu_core_num = 0
        
    def load_data(self) -> bool:
        """Load and preprocess the performance data."""
        try:
            # Get column names
            df_get_col_name = pd.read_csv(
                self.data_path, 
                sep=',', 
                encoding_errors='ignore', 
                skiprows=lambda x: x in np.append([0, 1], np.arange(3, 100))
            )
            
            new_col_names = [col.strip() for col in df_get_col_name.columns]
            
            # Load main data
            self.df_final = pd.read_csv(
                self.data_path, 
                sep=',',
                encoding_errors='ignore',
                skiprows=lambda x: x in np.arange(2)
            )
            
            # Filter data
            self.df_final = self.df_final[self.df_final.iloc[:, 0] == 80]
            
            # Convert numeric columns
            for col in self.df_final.columns[2:]:
                try:
                    self.df_final.loc[:, col] = pd.to_numeric(self.df_final.loc[:, col], errors='coerce')
                except ValueError:
                    continue
            
            # Set column names
            self.df_final.columns = new_col_names
            
            # Process datetime
            self._process_datetime()
            
            # Process framerate
            self.df_final['Framerate'] = pd.to_numeric(
                self.df_final['Framerate'].replace('N/A', np.nan), 
                errors='coerce'
            ).fillna(60.0)
            
            # Determine CPU core count
            self._determine_cpu_cores()
            
            # Clean and process CPU data
            self._process_cpu_data()
            
            logger.info(f"Data loaded successfully. CPU cores detected: {self.cpu_core_num}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def _process_datetime(self):
        """Process datetime column."""
        aft_format_data = "%d/%m/%Y %H:%M:%S"
        self.df_final['format_time_aft'] = (
            self.df_final.iloc[:, 1]
            .astype(str)
            .str.strip()
            .str.replace('-', '/')
            .apply(lambda x: ' '.join(x.split()))
        )
        self.df_final['format_time_aft'] = pd.to_datetime(
            self.df_final['format_time_aft'], 
            format=aft_format_data, 
            errors='coerce'
        )
    
    def _determine_cpu_cores(self):
        """Determine the number of CPU cores from column names."""
        core_gen = [
            col.split(' ')[0].split('CPU')[1] 
            for col in self.df_final.columns 
            if col.startswith("CPU") and col.split(' ')[0].split('CPU')[1].isnumeric()
        ]
        self.cpu_core_num = max([int(i) for i in core_gen]) if core_gen else 0
    
    def _process_cpu_data(self):
        """Process CPU-related data columns."""
        # Remove rows with invalid CPU usage
        invalid_rows = self.df_final['CPU usage'] == 'N/A'
        self.df_final = self.df_final[~invalid_rows]
        
        # Process main CPU columns
        cpu_cols = ['CPU clock', 'CPU usage', 'CPU power']
        for col in cpu_cols:
            if col in self.df_final.columns:
                self.df_final[col] = pd.to_numeric(
                    self.df_final[col].astype(str).str.strip(), 
                    errors='coerce'
                )
        
        # Process individual CPU core columns
        for i in range(1, self.cpu_core_num + 1):
            for metric in ['clock', 'usage']:
                col_name = f'CPU{i} {metric}'
                if col_name in self.df_final.columns:
                    self.df_final[col_name] = pd.to_numeric(
                        self.df_final[col_name].astype(str).str.strip(), 
                        errors='coerce'
                    )
    
    def plot_cpu_temperatures(self):
        """Plot CPU temperature data."""
        if self.df_final is None:
            logger.error("Data not loaded. Call load_data() first.")
            return
        
        cpu_temp_cols = [f'CPU{i} temperature' for i in range(1, self.cpu_core_num + 1)]
        existing_temp_cols = [col for col in cpu_temp_cols if col in self.df_final.columns]
        
        if not existing_temp_cols:
            logger.warning("No CPU temperature columns found.")
            return
        
        # Set up plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot individual core temperatures
        for col in existing_temp_cols:
            ax.plot(self.df_final['format_time_aft'], self.df_final[col], 
                   linestyle='--', linewidth=0.8, alpha=0.7)
        
        # Plot overall CPU temperature if available
        if 'CPU temperature' in self.df_final.columns:
            ax.plot(self.df_final['format_time_aft'], self.df_final['CPU temperature'], 
                   linewidth=2, color='red', label='Overall CPU Temp')
        
        ax.set_title('CPU Temperature Analysis', fontsize=14, fontweight='bold')
        ax.set_ylabel('Temperature (°C)')
        ax.set_xlabel('Time')
        ax.legend([col.replace('CPU', 'Core').replace(' temperature', '') for col in existing_temp_cols] + 
                 (['Overall'] if 'CPU temperature' in self.df_final.columns else []),
                 bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_cpu_power_analysis(self):
        """Plot both CPU power percentage and power consumption in watts."""
        if self.df_final is None:
            logger.error("Data not loaded. Call load_data() first.")
            return
        
        # Check for available power columns
        power_percent_cols = ['CPU power', 'CPU Power %', 'CPU power percent']
        power_watt_cols = ['Power', 'CPU power (W)', 'Power consumption', 'CPU Power Consumption']
        
        power_percent_col = None
        power_watt_col = None
        
        # Find power percentage column
        for col in power_percent_cols:
            if col in self.df_final.columns:
                power_percent_col = col
                break
        
        # Find power watts column
        for col in power_watt_cols:
            if col in self.df_final.columns:
                power_watt_col = col
                break
        
        if not power_percent_col and not power_watt_col:
            logger.warning("No CPU power columns found.")
            return
        
        # Set up plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        
        # Create subplots - 2 rows, 1 column
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot CPU Power Percentage
        if power_percent_col:
            power_percent_data = pd.to_numeric(self.df_final[power_percent_col], errors='coerce')
            
            axes[0].plot(self.df_final['format_time_aft'], power_percent_data, 
                        linewidth=1.5, color='orange', alpha=0.8, label=f'{power_percent_col}')
            
            # Add average line
            avg_power_percent = power_percent_data.mean()
            axes[0].axhline(y=avg_power_percent, color='darkorange', linestyle='--', alpha=0.7, 
                           label=f'Average: {avg_power_percent:.1f}%')
            
            # Add performance zones
            axes[0].axhspan(0, 25, alpha=0.1, color='green', label='Low Load')
            axes[0].axhspan(25, 75, alpha=0.1, color='yellow', label='Medium Load')
            axes[0].axhspan(75, 220, alpha=0.1, color='red', label='High Load')
            
            axes[0].set_title('CPU Power Consumption (%)', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('CPU Power (%)')
            axes[0].set_ylim(0, 220)
            axes[0].set_yticks(np.arange(0, 230, 10))
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc='upper left')
        else:
            axes[0].text(0.5, 0.5, 'CPU Power (%) data not available', 
                        transform=axes[0].transAxes, ha='center', va='center', 
                        fontsize=12, color='gray')
            axes[0].set_title('CPU Power Consumption (%) - No Data', fontsize=14, fontweight='bold')
        
        # Plot Power in Watts
        if power_watt_col:
            power_watt_data = pd.to_numeric(self.df_final[power_watt_col], errors='coerce')
            
            axes[1].plot(self.df_final['format_time_aft'], power_watt_data, 
                        linewidth=1.5, color='red', alpha=0.8, label=f'{power_watt_col}')
            
            # Add average line
            avg_power_watt = power_watt_data.mean()
            axes[1].axhline(y=avg_power_watt, color='darkred', linestyle='--', alpha=0.7, 
                           label=f'Average: {avg_power_watt:.1f}W')
            
            # Add power efficiency zones (typical for modern CPUs)
            max_power = power_watt_data.max()
            if max_power > 200:  # High-end CPU
                axes[1].axhspan(0, 65, alpha=0.1, color='green', label='Efficient')
                axes[1].axhspan(65, 150, alpha=0.1, color='yellow', label='Normal')
                axes[1].axhspan(150, max_power * 1.1, alpha=0.1, color='red', label='High Power')
                tick_interval = 25
            else:  # Mid-range CPU
                axes[1].axhspan(0, 35, alpha=0.1, color='green', label='Efficient')
                axes[1].axhspan(35, 85, alpha=0.1, color='yellow', label='Normal')
                axes[1].axhspan(85, max_power * 1.1, alpha=0.1, color='red', label='High Power')
                tick_interval = 20
            
            axes[1].set_title('CPU Power Consumption (Watts)', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Power (Watts)')
            axes[1].set_yticks(np.arange(0, max_power + tick_interval, tick_interval))
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc='upper left')
        else:
            axes[1].text(0.5, 0.5, 'CPU Power (Watts) data not available', 
                        transform=axes[1].transAxes, ha='center', va='center', 
                        fontsize=12, color='gray')
            axes[1].set_title('CPU Power Consumption (Watts) - No Data', fontsize=14, fontweight='bold')
        
        # Common settings
        axes[1].set_xlabel('Time')
        plt.xticks(rotation=45)
        # fig.suptitle('CPU Power Analysis', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.show()
        
        # Print power statistics
        print("\nCPU Power Statistics:")
        print("=" * 25)
        
        if power_percent_col:
            percent_stats = pd.to_numeric(self.df_final[power_percent_col], errors='coerce').agg(['mean', 'min', 'max', 'std']).round(2)
            print(f"\nCPU Power Percentage ({power_percent_col}):")
            for stat, value in percent_stats.items():
                print(f"  {stat.capitalize()}: {value}%")
        
        if power_watt_col:
            watt_stats = pd.to_numeric(self.df_final[power_watt_col], errors='coerce').agg(['mean', 'min', 'max', 'std']).round(2)
            print(f"\nCPU Power Consumption ({power_watt_col}):")
            for stat, value in watt_stats.items():
                print(f"  {stat.capitalize()}: {value}W")
            
            # Calculate energy consumption estimate
            if not watt_stats.isna().any():
                duration_hours = (self.df_final['format_time_aft'].max() - 
                                self.df_final['format_time_aft'].min()).total_seconds() / 3600
                avg_power = watt_stats['mean']
                energy_kwh = (avg_power * duration_hours) / 1000
                
                print(f"\nSession Duration: {duration_hours:.2f} hours")
                print(f"Estimated Energy Consumption: {energy_kwh:.4f} kWh")
                print(f"Estimated Cost (at TL 2.59/kWh): TL {energy_kwh * 2.59:.4f}")
    
    def plot_gpu_memory_only(self):
        """Plot only GPU memory (VRAM) usage for GPU analysis."""
        if self.df_final is None:
            return
        
        gpu_memory_cols = ['Memory usage', 'GPU memory', 'VRAM usage']
        gpu_col = None
        
        for col in gpu_memory_cols:
            if col in self.df_final.columns:
                gpu_col = col
                break
        
        if not gpu_col:
            return
        
        # Set up plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        gpu_data = pd.to_numeric(self.df_final[gpu_col], errors='coerce')
        
        ax.plot(self.df_final['format_time_aft'], gpu_data, 
                linewidth=1.5, color='blue', alpha=0.8)
        
        # Add average line
        avg_gpu_mem = gpu_data.mean()
        ax.axhline(y=avg_gpu_mem, color='darkblue', linestyle='--', alpha=0.7, 
                   label=f'Average: {avg_gpu_mem:.0f} MB')
        
        ax.set_title('GPU Memory Usage (VRAM)', fontsize=14, fontweight='bold')
        ax.set_ylabel('VRAM Usage (MB)')
        ax.set_xlabel('Time')
        
        # Set appropriate y-axis range
        max_gpu = gpu_data.max()
        if max_gpu > 8000:
            ax.set_yticks(np.arange(0, max_gpu + 1000, 1000))
        else:
            ax.set_yticks(np.arange(0, max_gpu + 500, 500))
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_cpu_usage(self):
        """Plot CPU usage data."""
        if self.df_final is None:
            logger.error("Data not loaded. Call load_data() first.")
            return
        
        cpu_usage_cols = [f'CPU{i} usage' for i in range(1, self.cpu_core_num + 1)]
        existing_usage_cols = [col for col in cpu_usage_cols if col in self.df_final.columns]
        
        # Set up plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot individual core usage
        for col in existing_usage_cols:
            ax.plot(self.df_final['format_time_aft'], self.df_final[col], 
                   linestyle='--', linewidth=0.8, alpha=0.7)
        
        # Plot overall CPU usage if available
        if 'CPU usage' in self.df_final.columns:
            ax.scatter(self.df_final['format_time_aft'], self.df_final['CPU usage'], 
                      s=20, marker='x', c='red', label='Overall CPU Usage')
        
        ax.set_title('CPU Usage Analysis', fontsize=14, fontweight='bold')
        ax.set_ylabel('Usage (%)')
        ax.set_xlabel('Time')
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 110, 10))
        ax.legend([col.replace('CPU', 'Core').replace(' usage', '') for col in existing_usage_cols] + 
                 (['Overall'] if 'CPU usage' in self.df_final.columns else []),
                 bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_gpu_analysis(self):
        """Plot comprehensive GPU analysis."""
        if self.df_final is None:
            logger.error("Data not loaded. Call load_data() first.")
            return
        
        gpu_metrics = ['GPU temperature', 'GPU usage', 'Power percent', 'Power', 'Fan speed']
        available_metrics = [metric for metric in gpu_metrics if metric in self.df_final.columns]
        
        if not available_metrics:
            logger.warning("No GPU metrics found.")
            return
        
        # Set up plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        
        fig, axes = plt.subplots(len(available_metrics), 1, figsize=(14, 3*len(available_metrics)), 
                                sharex=True)
        
        if len(available_metrics) == 1:
            axes = [axes]
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, metric in enumerate(available_metrics):
            axes[i].plot(self.df_final['format_time_aft'], self.df_final[metric], 
                        linewidth=1.2, color=colors[i % len(colors)])
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend([metric], loc='upper right')
        
        # Handle fan speed 2 if available
        if 'Fan speed 2' in self.df_final.columns and 'Fan speed' in available_metrics:
            fan_idx = available_metrics.index('Fan speed')
            axes[fan_idx].plot(self.df_final['format_time_aft'], self.df_final['Fan speed 2'], 
                              linewidth=1.2, linestyle='--', color='cyan')
            axes[fan_idx].legend(['Fan speed', 'Fan speed 2'], loc='upper right')
        
        axes[-1].set_xlabel('Time')
        fig.suptitle('GPU Performance Analysis', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Plot FPS separately
        self.plot_framerate()
        
        # Plot memory usage (GPU VRAM only for GPU analysis)
        self.plot_gpu_memory_only()
        
        # Show statistics
        print("\nGPU Performance Statistics:")
        print("=" * 40)
        stats = self.get_statistics(['GPU usage', 'GPU temperature', 'Power percent', 'Power', 'Fan speed', 'Framerate'])
        print(stats)
    
    def get_statistics(self, metrics: Optional[list] = None) -> pd.DataFrame:
        """Get statistical summary of specified metrics."""
        if self.df_final is None:
            logger.error("Data not loaded. Call load_data() first.")
            return pd.DataFrame()
        
        if metrics is None:
            metrics = ['GPU usage', 'GPU temperature', 'Power percent', 'Power', 'Fan speed', 'Framerate']
        
        available_metrics = [metric for metric in metrics if metric in self.df_final.columns]
        
        if not available_metrics:
            logger.warning("No specified metrics found in data.")
            return pd.DataFrame()
        
        stats = self.df_final[available_metrics].agg(['mean', 'min', 'max', 'median', 'std']).round(2)
        return stats
    
    def save_data(self, filename: str, output_dir: str = "output"):
        """Save processed data to files."""
        if self.df_final is None:
            logger.error("Data not loaded. Call load_data() first.")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save as CSV
        csv_path = output_path / f"{filename}.csv"
        self.df_final.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Save as Excel
        xlsx_path = output_path / f"{filename}.xlsx"
        self.df_final.to_excel(xlsx_path, index=False)
        
        logger.info(f"Data saved to {csv_path} and {xlsx_path}")
    
    def plot_framerate(self):
        """Plot framerate (FPS) data."""
        if self.df_final is None:
            logger.error("Data not loaded. Call load_data() first.")
            return
        
        if 'Framerate' not in self.df_final.columns:
            logger.warning("Framerate column not found.")
            return
        
        # Set up plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.df_final['format_time_aft'], self.df_final['Framerate'], 
               linewidth=1.5, color='green', alpha=0.8)
        
        # Add average line
        avg_fps = self.df_final['Framerate'].mean()
        ax.axhline(y=avg_fps, color='red', linestyle='--', alpha=0.7, 
                  label=f'Average: {avg_fps:.1f} FPS')
        
        ax.set_title('Framerate (FPS) Analysis', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frames Per Second (FPS)')
        ax.set_xlabel('Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_memory_usage(self):
        """Plot both GPU memory and system RAM usage."""
        if self.df_final is None:
            logger.error("Data not loaded. Call load_data() first.")
            return
        
        # Check for available memory columns
        gpu_memory_cols = ['Memory usage', 'GPU memory', 'VRAM usage']
        ram_cols = ['RAM usage', 'System RAM', 'Memory']
        
        gpu_col = None
        ram_col = None
        
        # Find GPU memory column
        for col in gpu_memory_cols:
            if col in self.df_final.columns:
                gpu_col = col
                break
        
        # Find RAM column  
        for col in ram_cols:
            if col in self.df_final.columns:
                ram_col = col
                break
        
        if not gpu_col and not ram_col:
            logger.warning("No memory usage columns found.")
            return
        
        # Set up plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        
        # Determine subplot layout
        num_plots = sum([bool(gpu_col), bool(ram_col)])
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6*num_plots), sharex=True)
        
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot GPU Memory (VRAM)
        if gpu_col:
            gpu_data = pd.to_numeric(self.df_final[gpu_col], errors='coerce')
            
            axes[plot_idx].plot(self.df_final['format_time_aft'], gpu_data, 
                              linewidth=1.5, color='blue', alpha=0.8, label=f'{gpu_col}')
            
            # Add average line
            avg_gpu_mem = gpu_data.mean()
            axes[plot_idx].axhline(y=avg_gpu_mem, color='darkblue', linestyle='--', alpha=0.7, 
                                  label=f'Average: {avg_gpu_mem:.0f} MB')
            
            axes[plot_idx].set_title('GPU Memory Usage (VRAM)', fontsize=14, fontweight='bold')
            axes[plot_idx].set_ylabel('VRAM Usage (MB)')
            
            # Set appropriate y-axis range for GPU memory
            max_gpu = gpu_data.max()
            if max_gpu > 8000:
                axes[plot_idx].set_yticks(np.arange(0, max_gpu + 1000, 1000))
            else:
                axes[plot_idx].set_yticks(np.arange(0, max_gpu + 500, 500))
                
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].legend()
            plot_idx += 1
        
        # Plot System RAM
        if ram_col:
            # Process RAM data
            if ram_col not in self.df_final.columns:
                logger.warning(f"Column '{ram_col}' not found, trying to process...")
                return
                
            ram_data = pd.to_numeric(self.df_final[ram_col].astype(str).str.strip(), errors='coerce')
            
            axes[plot_idx].plot(self.df_final['format_time_aft'], ram_data, 
                              linewidth=1.5, color='green', alpha=0.8, label=f'{ram_col}')
            
            # Add average line
            avg_ram = ram_data.mean()
            axes[plot_idx].axhline(y=avg_ram, color='darkgreen', linestyle='--', alpha=0.7, 
                                  label=f'Average: {avg_ram:.0f} MB')
            
            axes[plot_idx].set_title('System RAM Usage', fontsize=14, fontweight='bold')
            axes[plot_idx].set_ylabel('RAM Usage (MB)')
            
            # Set appropriate y-axis range for system RAM
            max_ram = ram_data.max()
            if max_ram > 16000:
                tick_interval = 2048
            elif max_ram > 8000:
                tick_interval = 1024
            else:
                tick_interval = 512
                
            axes[plot_idx].set_yticks(np.arange(0, max_ram + tick_interval, tick_interval))
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].legend()
        
        # Set common x-axis label
        axes[-1].set_xlabel('Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Print memory statistics
        print("\nMemory Usage Statistics:")
        print("=" * 30)
        if gpu_col:
            gpu_stats = pd.to_numeric(self.df_final[gpu_col], errors='coerce').agg(['mean', 'min', 'max', 'std']).round(1)
            print(f"\nGPU Memory ({gpu_col}):")
            for stat, value in gpu_stats.items():
                print(f"  {stat.capitalize()}: {value} MB")
        
        if ram_col:
            ram_stats = pd.to_numeric(self.df_final[ram_col].astype(str).str.strip(), errors='coerce').agg(['mean', 'min', 'max', 'std']).round(1)
            print(f"\nSystem RAM ({ram_col}):")
            for stat, value in ram_stats.items():
                print(f"  {stat.capitalize()}: {value} MB")
    
    def plot_cpu_frequencies(self):
        """Plot CPU frequency data."""
        if self.df_final is None:
            logger.error("Data not loaded. Call load_data() first.")
            return
        
        cpu_freq_cols = [f'CPU{i} clock' for i in range(1, self.cpu_core_num + 1)]
        existing_freq_cols = [col for col in cpu_freq_cols if col in self.df_final.columns]
        
        if not existing_freq_cols:
            logger.warning("No CPU frequency columns found.")
            return
        
        # Set up plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot individual core frequencies
        for col in existing_freq_cols:
            ax.plot(self.df_final['format_time_aft'], self.df_final[col], 
                   linestyle='--', linewidth=1, alpha=0.7)
        
        # Plot overall CPU frequency if available
        if 'CPU clock' in self.df_final.columns:
            ax.scatter(self.df_final['format_time_aft'], self.df_final['CPU clock'], 
                      marker='x', c='red', s=10, label='Overall CPU Clock')
        
        ax.set_title('CPU Frequencies Analysis', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency (MHz)')
        ax.set_xlabel('Time')
        ax.set_yticks(np.arange(0, 6000, 250))
        ax.legend([col.replace('CPU', 'Core').replace(' clock', '') for col in existing_freq_cols] + 
                 (['Overall'] if 'CPU clock' in self.df_final.columns else []),
                 bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    """Main function with improved menu system."""
    print("Performance Analysis Tool")
    print("=" * 30)
    
    # Get data path
    data_path = input("Enter path to Afterburner log file: ").strip()
    if not data_path:
        data_path = r"C:\Python_Works\Extensions\texts\afterburner_intel_logs\afterburner\MHW.txt"
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(data_path)
    
    if not analyzer.load_data():
        print("Failed to load data. Exiting.")
        return
    
    menu_options = {
        '1': ('CPU Temperature Analysis', analyzer.plot_cpu_temperatures),
        '2': ('CPU Usage Analysis', analyzer.plot_cpu_usage),
        '3': ('CPU Frequency Analysis', analyzer.plot_cpu_frequencies),
        '4': ('CPU Power Analysis (% + Watts)', analyzer.plot_cpu_power_analysis),
        '5': ('GPU Analysis (Complete)', analyzer.plot_gpu_analysis),
        '6': ('Framerate Analysis', analyzer.plot_framerate),
        '7': ('Memory Analysis (GPU + System RAM)', analyzer.plot_memory_usage),
        '8': ('Show Statistics', lambda: print(analyzer.get_statistics())),
        '9': ('Save Data', lambda: analyzer.save_data(input("Enter filename: ").strip() or "performance_data")),
        'q': ('Quit', None)
    }
    
    while True:
        print("\nMenu Options:")
        for key, (description, _) in menu_options.items():
            print(f"{key} - {description}")
        
        choice = input("\nEnter your choice: ").strip().lower()
        
        if choice == 'q':
            print("Exiting...")
            break
        
        if choice in menu_options and choice != 'q':
            try:
                menu_options[choice][1]()
            except Exception as e:
                logger.error(f"Error executing option {choice}: {e}")
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()