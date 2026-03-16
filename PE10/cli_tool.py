import click
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

@click.group()
def cli():
    """MLOps CLI tool to manage ML model lifecycle."""
    pass

@cli.command()
def train():
    """Train and save a simple Iris model."""
    click.echo("Training model...")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/trained_model.pkl")

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    click.echo(f" Model trained with accuracy: {acc:.2f}")

@cli.command()
def evaluate():
    """Evaluate the trained model."""
    if not os.path.exists("models/trained_model.pkl"):
        click.echo(" No model found! Run 'python cli_tool.py train' first.")
        return

    click.echo(" Evaluating model...")
    X, y = load_iris(return_X_y=True)
    model = joblib.load("models/trained_model.pkl")
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    click.echo(f" Overall dataset accuracy: {acc:.2f}")

if __name__ == "__main__":
    cli()