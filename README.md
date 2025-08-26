# MSI Afterburner Performance Analysis Tool

A comprehensive Python-based tool for analyzing CPU and GPU performance data from MSI Afterburner logs. This tool provides detailed visualizations and statistics for system monitoring data, perfect for gamers, overclockers, and system performance enthusiasts.

## Features

###  CPU Analysis
- **Temperature Monitoring**: Individual core temperatures and overall CPU temperature
- **Usage Analysis**: Per-core and overall CPU utilization with visual load indicators
- **Frequency Tracking**: CPU clock speeds for all cores with frequency analysis
- **Power Consumption**: Both percentage-based and watts-based power monitoring with efficiency zones

###  GPU Analysis (Discrete GPUs)
- **Performance Metrics**: Temperature, usage, power consumption, and fan speeds
- **Memory Monitoring**: VRAM usage analysis with statistical summaries
- **Clock Speeds**: Core and memory clock frequency tracking
- **Gaming Performance**: Framerate analysis with average FPS indicators

###  Memory Analysis
- **Dual Memory Tracking**: Separate analysis for GPU VRAM and System RAM
- **Usage Patterns**: Memory consumption over time with average usage lines
- **Smart Scaling**: Automatic y-axis scaling based on memory capacity

### Advanced Features
- **Energy Consumption**: Calculate power usage and estimated electricity costs
- **Statistical Analysis**: Mean, min, max, median, and standard deviation for all metrics
- **Performance Zones**: Visual indicators for idle, normal, and high-load states
- **Data Export**: Save processed data in CSV and Excel formats

## Installation

### Prerequisites
```bash
pip install pandas numpy matplotlib scikit-learn seaborn pathlib
```

### Supported Python Versions
- Python 3.7+
- Compatible with Windows, macOS, and Linux

## Usage

### Quick Start
1. **Generate MSI Afterburner logs**:
   - Open MSI Afterburner
   - Go to Settings → Monitoring
   - Enable desired metrics (CPU temp, GPU temp, usage, etc.)
   - MSI Afterburner takes logs in .hml format but users simply need to convert it to `.txt`. 

2. **Run the analysis tool**:
   ```bash
   python performance_analyzer.py
   ```

3. **Enter your log file path** when prompted or use the default path

4. **Choose analysis options** from the interactive menu

### Menu Options
```
1 - CPU Temperature Analysis       Individual core and overall CPU temperatures
2 - CPU Usage Analysis            Per-core and overall CPU utilization
3 - CPU Frequency Analysis        CPU clock speeds and frequency patterns
4 - CPU Power Analysis            Power consumption (% and Watts) with energy costs
5 - GPU Analysis (Complete)       Comprehensive GPU performance metrics
6 - Framerate Analysis            Gaming performance and FPS analysis
7 - Memory Analysis               GPU VRAM and System RAM usage
8 - Show Statistics              Detailed statistical summaries
9 - Save Data                    Export processed data
q - Quit                         Exit the application
```

### Example Workflow
```python
from performance_analyzer import PerformanceAnalyzer

# Initialize analyzer
analyzer = PerformanceAnalyzer("path/to/your/afterburner_log.txt")

# Load and process data
if analyzer.load_data():
    # Generate CPU temperature analysis
    analyzer.plot_cpu_temperatures()
    
    # Analyze GPU performance
    analyzer.plot_gpu_analysis()
    
    # Get statistical summary
    stats = analyzer.get_statistics()
    print(stats)
```

## Data Format Support

### MSI Afterburner Log Format
The tool automatically processes MSI Afterburner `.txt` log files with the following supported metrics:

**CPU Metrics:**
- CPU temperature, CPU1-CPUn temperature
- CPU usage, CPU1-CPUn usage
- CPU clock, CPU1-CPUn clock
- CPU power, Power consumption

![Description](https://github.com/Zodijackyl98/afterburner-data-analyzer/blob/main/examples/graphs/cpu_temperature.png)

![Description](https://github.com/Zodijackyl98/afterburner-data-analyzer/blob/main/examples/graphs/cpu_usage.png)

![Description](https://github.com/Zodijackyl98/afterburner-data-analyzer/blob/main/examples/graphs/cpu_freqs.png)

**GPU Metrics:**
- GPU temperature, GPU usage
- Memory usage (VRAM), Core clock, Memory clock
- Power, Power percent, Fan speed
- Framerate (FPS)

![Description](https://github.com/Zodijackyl98/afterburner-data-analyzer/blob/main/examples/graphs/gpu_stats.png)

![Description](https://github.com/Zodijackyl98/afterburner-data-analyzer/blob/main/examples/graphs/gpu_framerate.png)


**System Metrics:**
- RAM usage, Memory usage
- Timestamps with automatic date/time parsing

![Description](https://github.com/Zodijackyl98/afterburner-data-analyzer/blob/main/examples/graphs/gpu_vram_ram.png)


## Configuration

### Default Paths
```python
# Default Afterburner log path (Windows)
DEFAULT_PATH = r"C:\Python_Works\Extensions\texts\afterburner_intel_logs\afterburner\MHW.txt"

# Output directory for saved data
OUTPUT_DIR = "output/"
```

### Customization Options
- **Plot Styles**: Automatic fallback between seaborn and matplotlib styles
- **Color Schemes**: Professional color palettes with performance zone indicators
- **Export Formats**: CSV and Excel export with UTF-8 encoding
- **Statistical Metrics**: Configurable statistical analysis parameters

## Output Examples

### Visual Analysis
- **Multi-panel Dashboards**: Comprehensive performance overviews
- **Time-series Plots**: Performance metrics over time with trend analysis
- **Performance Zones**: Color-coded efficiency and load indicators
- **Statistical Overlays**: Average lines and performance thresholds

### Data Export
```
performance_data.csv      Comma-separated values for data analysis
performance_data.xlsx     Excel format with formatted columns
```

### Statistical Output
```
GPU Performance Statistics:
========================================
                GPU usage  GPU temperature  Power  Framerate
mean               65.34            72.45  145.67      89.23
min                12.45            45.67   78.90      45.67
max                98.76            84.32  198.45     144.50
median             67.89            71.23  142.34      92.15
std                18.45             8.92   25.67      15.78
```

```
CPU Power Statistics:
=========================

CPU Power Percentage (CPU power):
  Mean: 106.79%
  Min: 0.0%
  Max: 174.18%
  Std: 19.33%

CPU Power Consumption (Power):
  Mean: 158.92W
  Min: 52.78W
  Max: 179.63W
  Std: 10.85W

Session Duration: 0.32 hours
Estimated Energy Consumption: 0.0501 kWh
Estimated Cost (at TL 2.59/kWh): TL 0.1298

```

![Description](https://github.com/Zodijackyl98/afterburner-data-analyzer/blob/main/examples/graphs/cpu_power_consumption.png)


## Hardware Compatibility

### Supported GPUs
- **NVIDIA**: RTX 40/30/20/10 series, GTX 16/10 series
- **AMD**: RX 7000/6000/5000 series, Radeon VII, Vega series
- **Focus**: Discrete GPUs only (integrated graphics ignored)
- All feedbacks are welcome especially with unique setups.

### Supported CPUs
- **Intel**: Core i3/i5/i7/i9 (all generations)
- **AMD**: Ryzen 3/5/7/9, Threadripper series
- **Multi-core**: Automatic detection of core count (up to 32+ cores)

## Troubleshooting

### Common Issues

**"No data found" error:**
- Ensure MSI Afterburner logging is enabled
- Check that the log file path is correct
- Verify metrics are being monitored in Afterburner

**Missing columns warning:**
- Enable desired metrics in MSI Afterburner monitoring tab
- Restart logging to generate complete data
- Check hardware compatibility

**Plot style errors:**
- Script automatically handles matplotlib/seaborn version differences
- Falls back to default matplotlib style if needed

**Memory errors:**
- Consider analyzing smaller time windows
- Ensure sufficient system RAM

### Performance Tips
- **Log Duration**: 1-4 hours of logging provides good analysis data
- **Sampling Rate**: 1-second intervals balance detail and file size
- **Metrics Selection**: Enable only needed metrics to reduce processing time

## Contributing

### Development Setup
```bash
git clone https://github.com/your-repo/msi-afterburner-analyzer
cd msi-afterburner-analyzer
pip install -r requirements.txt
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function parameters
- Include docstrings for all methods
- Add error handling for data processing

### Feature Requests
- GPU comparison tools
- Real-time monitoring integration
- Advanced statistical analysis
- Custom performance benchmarks

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **MSI Afterburner**: For providing comprehensive hardware monitoring
- **Pandas/NumPy**: For efficient data processing capabilities  
- **Matplotlib/Seaborn**: For powerful visualization tools

## Version History

### v1.0.0 (Current)
- Complete object-oriented rewrite
- Enhanced error handling and logging
- Dual memory analysis (GPU + System RAM)
- CPU power analysis with energy calculations
- Professional visualizations with performance zones


