# DFG_activation_maps

A Streamlit web application for comparing neuroimaging activation maps.

## How to Edit Files

### Repository Structure

```
DFG_activation_maps/
├── README.md                           # This documentation file
├── app_compare_activation_maps.py     # Main Streamlit application
└── requirements.txt                   # Python dependencies
```

### Editing the Application (`app_compare_activation_maps.py`)

This is the main Streamlit application file. Here's how to edit it:

1. **Open the file in your preferred editor:**
   ```bash
   # Using nano
   nano app_compare_activation_maps.py
   
   # Using vim
   vim app_compare_activation_maps.py
   
   # Using VS Code
   code app_compare_activation_maps.py
   ```

2. **Key sections to edit:**
   - **Configuration (near the top)**: Update Google Drive folder IDs, data paths
   - **Data loading functions**: Modify how data is downloaded and processed
   - **UI components**: Change the Streamlit interface elements
   - **Visualization settings**: Adjust plot parameters and layouts

3. **Common edits:**
   - Change Google Drive folder ID: Edit `DRIVE_FOLDER_ID` variable
   - Modify data directory: Edit `DATA_ROOT` variable
   - Update page layout: Edit `st.set_page_config()` parameters

### Editing Dependencies (`requirements.txt`)

To add or modify Python packages:

1. **Open the file:**
   ```bash
   nano requirements.txt
   ```

2. **Add new packages:**
   - Add one package per line
   - Specify versions for reproducibility: `package_name==1.2.3` (recommended)
   - Using exact versions ensures consistent environments across installations

3. **After editing, reinstall dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Development Workflow

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Orizehn/DFG_activation_maps.git
   cd DFG_activation_maps
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Make your edits** to any file using your preferred editor

4. **Test your changes:**
   ```bash
   streamlit run app_compare_activation_maps.py
   ```

5. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Description of your changes"
   git push
   ```

### Quick Edit Examples

#### Example 1: Change the app title
```python
# Find this line (near the top of the file)
st.set_page_config(layout="wide", page_title="Compare Activation Maps")

# Change to:
st.set_page_config(layout="wide", page_title="Your New Title")
```

#### Example 2: Update Google Drive folder ID
```python
# Find this line (in the CLOUD DATA SETUP section)
DRIVE_FOLDER_ID = "1Vr4QPF4-Vb7tUAB6cvJTcWiDNX3OD5QT"

# Change to your folder ID:
DRIVE_FOLDER_ID = "your_folder_id_here"
```

#### Example 3: Add a new Python package
```bash
# Edit requirements.txt with your preferred editor:
nano requirements.txt
```

Add the package name on a new line. For example, if your requirements.txt currently contains:
```
streamlit==1.31.0
numpy==1.24.3
pandas==2.0.1
```

Add your new package with version pinning:
```
streamlit==1.31.0
numpy==1.24.3
pandas==2.0.1
new-package-name==1.0.0
```

Then install the updated dependencies:
```bash
pip install -r requirements.txt
```

### Tips

- Always test your changes locally before committing
- Use version control (git) to track your edits
- Keep dependencies in `requirements.txt` up to date
- Comment your code changes for better maintainability
- The app uses Streamlit caching (`@st.cache_resource`) for performance.
  If you modify data loading functions, you may need to restart the app or
  clear the cache to see your changes

### Getting Help

- Streamlit documentation: https://docs.streamlit.io/
- Nilearn documentation: https://nilearn.github.io/
- For issues with this repository, create a GitHub issue