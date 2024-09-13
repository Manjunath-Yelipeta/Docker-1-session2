

🚀 PyTorch MNIST Training in Docker 🚀
Welcome to the PyTorch MNIST Docker Project! This repository demonstrates how to efficiently train a neural network on the MNIST dataset using PyTorch inside a Docker container. The project showcases multi-processing for faster training, saving checkpoints, and resuming from saved states. Perfect for developers looking to containerize their ML workflow.

📑 Table of Contents
Project Structure
Dockerfile Explanation
Requirements
Multi-Processing in Training
Docker Usage
Training Arguments
Data & Model Mounting
Mindmap
🛠 Project Structure
.
├── Dockerfile
├── requirements.txt
├── train.py
├── training.py
└── README.md
🐳 Dockerfile Explanation
Our Dockerfile is based on the lightweight Python 3.9-slim image to minimize footprint while ensuring all necessary dependencies are installed.

Key Features:
Base Image: python:3.9-slim for a small and fast build.
Working Directory Setup: Organizes files and dependencies.
System Dependencies: Installs required OS-level packages.
Python Dependencies: Uses requirements.txt to install CPU-only versions of PyTorch and libraries, keeping the image small and CPU-efficient.
📋 Requirements
The requirements.txt specifies essential libraries:

torch (CPU version)
torchvision
numpy
pillow
These are fully compatible with the base Python 3.9 image.

⚡ Multi-Processing in Training
To speed up training, we leverage multi-processing. The train.py script uses torch.multiprocessing to create multiple worker processes, allowing parallel model training, especially on multi-core CPUs.

Benefits:

Efficiently utilizes multiple CPU cores.
Parallelizes data loading and preprocessing.
Can reduce overall training time.
Tip: Control the number of processes with the --num-processes argument.

🐋 Docker Usage
1. Build the Docker Image:
bash
docker build -t pytorch-mnist .
2. Run the Container:
bash
docker run --rm -v $(pwd)/data:/workspace/data -v $(pwd)/models:/workspace/models pytorch-mnist python train.py [ARGS]
Replace [ARGS] with your desired training arguments (see Training Arguments).

⚙️ Training Arguments
Customize the training process with the following command-line arguments in train.py:

Argument	Description	Default
--batch-size	Input batch size for training	64
--test-batch-size	Input batch size for testing	1000
--epochs	Number of epochs to train	1
--lr	Learning rate	0.01
--momentum	Momentum for SGD optimizer	0.5
--seed	Random seed for reproducibility	1
--log-interval	Batches to wait before logging status	10
--num-processes	Number of processes for parallel training	2
--resume	Resume training from checkpoint	False
Example:

bash
docker run --rm -v $(pwd)/data:/workspace/data -v $(pwd)/models:/workspace/models pytorch-mnist python train.py --epochs 5 --batch-size 32 --num-processes 4
📂 Data & Model Mounting
To keep the Docker image lightweight, we use volume mounting to persist data and models outside the container. This ensures the container remains clean, while still being able to save checkpoints and access datasets.

Mount the data directory:

bash
-v $(pwd)/data:/workspace/data
Mount the models directory:

bash
-v $(pwd)/models:/workspace/models

🧠 Mindmap

graph TD
    A[PyTorch MNIST Docker Project] --> B[Dockerfile]
    A --> C[train.py]
    A --> D[training.py]
    A --> E[requirements.txt]
    
    B --> F[Base Image: python:3.9-slim]
    B --> G[Install Dependencies]
    B --> H[Copy Project Files]
    
    C --> I[Parse Arguments]
    C --> J[Setup Multi-Processing]
    C --> K[Train Model]
    C --> L[Test Model]
    C --> M[Save Checkpoint]
    
    D --> N[train function]
    D --> O[test function]
    D --> P[train_epoch function]
    D --> Q[test_epoch function]
    
    E --> R[torch CPU version]
    E --> S[torchvision]
    E --> T[numpy]
    E --> U[pillow]
    
    K --> V[Multi-Process Training]
    V --> W[Process 1]
    V --> X[Process 2]
    V --> Y[Process N]
    
    M --> Z[Volume Mounting]
    Z --> AA[Data Directory]
    Z --> AB[Models Directory]
💡 Conclusion
By containerizing your PyTorch training workflow, you gain the flexibility to train anywhere, from your local machine to cloud environments. With multi-processing, volume mounting, and Docker, this project is a template for efficient, scalable ML training. Happy coding! ✨