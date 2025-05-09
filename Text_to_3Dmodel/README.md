🧠 Text-to-3D Model Generator using Shap-E
=========================================

Introduction
------------
This project explores AI-powered 3D model generation using OpenAI’s Shap-E model. 
It allows you to create realistic 3D object meshes from natural language prompts 
(e.g., "a cute dog") and save them as `.obj` files.

The project combines text-to-3D generation with mesh exporting and basic visualization.

Objective
---------
1. Convert text prompts into 3D models using OpenAI’s Shap-E.
2. Save models as `.obj` files compatible with Blender or Unity.
3. Render and visualize the models using both Shap-E rendering and matplotlib.
4. Provide an easy-to-follow pipeline to run the system locally or in Google Colab.

Thought Process
---------------
I firstly researched about 3D models and about their libraries and I chose one of the libraries.  
I faced the issue while saving the file as `.obj`, so I took help from ChatGPT for that part.

The goal was to:
- Enable `.obj` file export of AI-generated 3D meshes
- Support visualization and easy reuse
- Make it beginner-friendly

Steps to Run (Locally)
-----------------------
1. Clone the Repository:
   git clone https://github.com/yourusername/text-to-3d-model-generator.git
   cd text-to-3d-model-generator

2. Create a Virtual Environment:
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv 3dmodel
   source venv/bin/activate

3. Install Dependencies:
   pip install -r requirements.txt

4. Run the script:
   python script.py
if can't able to run the script
5. Open in google colab:
   python script.ipynb
   Run all Cells to get output

Libraries Used
--------------
- torch            (PyTorch for inference)
- shap-e           (OpenAI's 3D generation model)
- matplotlib       (for simple 3D visualization)
- trimesh          (for saving .obj files)
- google.colab     (optional – for file downloading in Colab)

Prompt Example
--------------
You can modify this line in script.py/script.ipynb:
   prompt = "a cute dog"

Outputs
-------
- cute_dog.obj       → Exported 3D model
- cute_dog.gif       → Rotating rendered GIF preview
- Matplotlib plot    → Basic static 3D view

requirements.txt
----------------
torch
shap-e
matplotlib
trimesh
google-colab

License
-------
MIT License

Acknowledgements
----------------
- OpenAI – Shap-E model
- ChatGPT – Help with saving .obj files and bug fixes

