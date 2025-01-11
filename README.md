
# Chest-Cancer-Classification-Using-CNN

- Uses cnn to classify images.

# **How to Install Conda** 

Setting up Conda in a GitHub Codespace for data science is straightforward. Follow these steps to get started:

### Step 1: Open Your Codespace

1. Navigate to your repository on GitHub.
2. Click the green "Code" button and select "Open with Codespaces."
3. Create a new codespace or open an existing one.

### Step 2: Install Miniconda

1. **Access the Terminal**: Open the terminal in your Codespace (you can find it in the bottom panel or use the shortcut `Ctrl + `).

2. **Download Miniconda**:
   Run the following command to download the Miniconda installer script:

   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```

3. **Install Miniconda**:
   Execute the installer script:

   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

   Follow the prompts during installation. You can press `Enter` to accept the default options. 

4. **Initialize Conda**:
   After installation, run:

   ```bash
   source ~/miniconda3/bin/activate
   ```

5. Next step will be **setting up Project** and **Virtual Environment**.
   
    - Let's First create a **Virtual Environment**. In my case I am using **conda**. 
      - First command will be to **create** a virtual environment with **specific python version** and **-y flag mean yes to all comming steps**.

          ```bash
          conda create -p venv python=3.10 -y 
          ```
      - Now we need to activate the **Created Virtual Environment** with this command.        
          ```bash
          conda activate venv/
          ```
6. But i will install libraries using **requirements.txt** with this command. This will install all mentioned libraries in **requirements.txt** and set up project.
        
    ```bash
    pip install -r requirements.txt 
    ```   
7. Now in order to run the whole pipeline run this command;

    ```bash
    python main.py
    ```

8. Now in order to run streamlit application use this command

    ```bash
    streamlit run streamlit.py
    ```
