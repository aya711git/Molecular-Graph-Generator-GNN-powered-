![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-PyG-EE4C2C?logo=pytorch)
![Next.js](https://img.shields.io/badge/Next.js-14%2B-000000?logo=nextdotjs)
![License](https://img.shields.io/badge/License-MIT-green)

# 🧬 Molecular Graph Generator (GNN-Powered)
A production-grade, full-stack application that converts chemical input (SMILES strings, drug names, or molecular formulas) into interactive molecular graphs and computes latent-space embeddings using Graph Neural Networks (MPNN / GCN).

تطبيق متكامل يعتمد على شبكات الرسم البياني العصبية (Graph Neural Networks) لتحويل المدخلات الكيميائية (مثل صيغ SMILES أو الأسماء الشائعة للمركبات) إلى تمثيل بياني تفاعلي واستخلاص المتجهات الكامنة (Latent Embeddings).

🌱 يهدف هذا المشروع إلى تقديم نموذج برمجي عملي يربط بين الكيمياء الحاسوبية والتعلم العميق، وتسهيل استكشاف الفضاء الجزيئي لبحوث اكتشاف الأدوية وتطوير المحتوى التقني في هذا المجال.

## ⚠️ Disclaimer & Model Status
The GNN encoder currently ships with randomly initialized weights (see backend/app/services/gnn_model.py). It produces valid, deterministic structural embeddings from molecular topology, but has not been trained on any chemical or pharmacological task.

Embeddings should not be interpreted as predictions of drug activity, safety, or efficacy until a trained checkpoint is loaded (see Training & checkpoints in docs/ARCHITECTURE.md).


## 🚀 Features
🧪 Flexible Chemical Input Parsing: Accepts raw SMILES, common/IUPAC drug names, or molecular formulas resolved via PubChem.

⚛️ Deterministic Graph Featurization: Converts molecules into atom nodes, bond edges, and fixed-size feature vectors via RDKit.

🤖 Modular GNN Embeddings: Dynamic encoder architecture supporting interchangeable GCN and MPNN layers via PyTorch Geometric.

📊 Interactive Graph Visualization: Full-stack integration with Next.js frontend to render dynamic graph topologies and latent vector spaces.

⚡ RESTful API Services: Asynchronous endpoints for parsing, featurization, and inference powered by FastAPI.


## 🛠️ Tech Stack
Backend: Python, FastAPI, PyTorch Geometric (PyG), RDKit, PubChemPy, Pydantic

Frontend: Next.js, TypeScript, Tailwind CSS, Lucide React

DevOps & Testing: Pytest, Uvicorn, OpenAPI / Swagger

## 🧩 Project Structure

```
molgnn/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── main.py
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── lib/
│   │   └── styles/
│   ├── package.json
│   ├── tsconfig.json
│   └── tailwind.config.ts
├── docs/
│   └── ARCHITECTURE.md
└── README.md
```

## ⚙️ Setup & Installation
1. Backend Setup
Navigate to the backend directory:

Bash
```cd backend```
Create and activate a virtual environment:

Bash
 ```python -m venv .venv```
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
Install dependencies:

Bash
```pip install -r requirements.txt```
Run the API server:

Bash
```uvicorn app.main:app --reload --port 8000```
Access OpenAPI docs at http://localhost:8000/docs

2. Frontend Setup
Open a new terminal and navigate to the frontend directory:

Bash
```cd frontend```
Install dependencies:

Bash
```npm install```
Start the development server:

Bash
```npm run dev```
Open http://localhost:3000 in your browser.


## 🧠 Future Improvements
Add 3D conformer generation and spatial molecular embedding display 🌐

Implement multi-objective property prediction metrics (QED, SA Score, Tox21) 📈

Support custom pretrained checkpoint loading for target-specific drug discovery 💊


## 📩 Contact

**Aya Khaled Khrais**

🌐 [GitHub](https://github.com/aya711git)

📧[Email](aya.khuris@gmail.com)

🖇️[LinkedIn](https://www.linkedin.com/in/aya-khaled-khuris/)

💼 Frontend Developer | Passionate about modern UI/UX |AI Research

✨ شكرًا لزيارتك، وأتمنى أن تجد في هذا المشروع فائدة وإلهامًا 💛

✨ Thank you for visiting! I hope you find inspiration and value in this project 💛
