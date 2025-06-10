# Movie Recommendation System with Graph Attention Networks

This project implements a movie recommendation system using Graph Attention Networks (GAT) on the MovieLens 20M dataset. The system includes both a command-line interface for training and inference, as well as a web application for interactive movie recommendations.

## Project Structure

```
├── app.py                 # Flask API server for the web application
├── main.py                # Training script for the GAT model
├── inference.py           # Command-line inference script
├── model.py               # GAT model implementation
├── data_processing.py     # Data loading and preprocessing utilities
├── training.py            # Training utilities and evaluation functions
├── config.py              # Configuration and hyperparameters
├── best_model.pt          # Saved trained model (generated after training)
├── dataset/               # MovieLens dataset files
└── website/               # React frontend application
```

## Requirements

### Python Dependencies
Make sure you have the following Python packages installed:

```bash
pip install torch torch-geometric pandas numpy flask flask-cors pathlib
```

### Node.js Dependencies
For the web application, you'll need Node.js and npm installed. The frontend uses React with Material-UI components.

## Getting Started

### 1. Training the Model

First, train the GAT model on the MovieLens dataset:

```bash
python main.py
```

**Optional training arguments:**
```bash
python main.py --epochs 20 --lr 0.001 --heads 8 --hidden 32 --batch_size_edges 64000
```

**Training parameters:**
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 2e-3)
- `--heads`: Number of attention heads (default: 8)
- `--hidden`: Hidden dimension size (default: 32)
- `--batch_size_edges`: Batch size for edge sampling (default: 64,000)
- `--sample_users`: Train on N random users, 0 = all users (default: 0)
- `--patience`: Early stopping patience (default: 5)
- `--max_tags`: Top-k tags to keep as features (default: 800)
- `--device`: Device to use (default: cuda if available, else cpu)

The training script will:
- Download and process the MovieLens 20M dataset
- Build a heterogeneous graph with users, movies, genres, and tags
- Train the GAT model with early stopping
- Save the best model as `best_model.pt`

### 2. Command-Line Inference

After training, you can get movie recommendations using the inference script:

```bash
python inference.py --liked_movies "Toy Story (1995)" "The Matrix (1999)" "Titanic (1997)" --top_k 10
```

**Example usage:**
```bash
# Basic recommendation
python inference.py --liked_movies "Forrest Gump (1994)" --top_k 5

# Multiple movies for better recommendations
python inference.py --liked_movies "The Shawshank Redemption (1994)" "Pulp Fiction (1994)" "The Dark Knight (2008)" --top_k 15

# Using a different model file
python inference.py --liked_movies "Avatar (2009)" --top_k 10 --model_path my_model.pt
```

**Inference parameters:**
- `--liked_movies`: List of movie titles you liked (space-separated, use quotes for titles with spaces)
- `--top_k`: Number of recommendations to return (default: 10)
- `--model_path`: Path to the saved model (default: best_model.pt)

### 3. Running the Web Application

The project includes a React frontend and Flask backend for an interactive web interface.

#### Starting the Flask API Server

```bash
python app.py
```

The Flask server will start on `http://localhost:5001` and provides the following endpoints:
- `POST /recommend`: Get movie recommendations based on user ratings
- `GET /health`: Health check endpoint

#### Starting the React Frontend

```bash
cd website
npm install
npm start
```

The React application will start on `http://localhost:3000` and automatically connect to the Flask backend.

### 4. Using the Complete System

1. **Train the model** (if not already done):
   ```bash
   python main.py
   ```

2. **Start the Flask API server** in one terminal:
   ```bash
   python app.py
   ```

3. **Start the React frontend** in another terminal:
   ```bash
   cd website
   npm start
   ```

4. **Open your browser** and navigate to `http://localhost:3000` to use the interactive recommendation system.

## API Usage

The Flask API accepts POST requests to `/recommend` with the following format:

```json
{
  "ratings": [
    {"title": "Toy Story (1995)", "rating": 5},
    {"title": "The Matrix (1999)", "rating": 4},
    {"title": "Titanic (1997)", "rating": 3}
  ]
}
```

Response format:
```json
{
  "success": true,
  "recommendations": [
    {
      "movieId": 1234,
      "title": "Movie Title (Year)",
      "genres": "Action|Adventure|Sci-Fi",
      "predicted_rating": 4.2,
      "tmdbId": null
    }
  ]
}
```

## Model Architecture

The system uses a Graph Attention Network (GAT) that learns embeddings for:
- **Users**: Based on their rating patterns
- **Movies**: Based on genres, tags, and user interactions
- **Genres**: Movie genre information
- **Tags**: User-generated tags for movies

The model predicts ratings by computing dot products between user and movie embeddings, with learnable bias terms.

## Dataset

The system uses the MovieLens 20M dataset, which contains:
- 20 million ratings from 138,000 users on 27,000 movies
- Movie metadata including genres
- User-generated tags

The dataset is automatically downloaded when you first run the training script.

## Troubleshooting

**Model file not found error:**
Make sure you've trained the model first by running `python main.py`.

**CUDA/GPU issues:**
The system automatically detects available hardware. For CPU-only training, the model will use CPU automatically.

**Port conflicts:**
- Flask server runs on port 5001
- React frontend runs on port 3000
- Make sure these ports are available

**Package installation issues:**
For PyTorch Geometric, you might need to install with specific CUDA versions:
```bash
pip install torch-geometric -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/{TORCH_VERSION}+{CUDA_VERSION}.html
```
