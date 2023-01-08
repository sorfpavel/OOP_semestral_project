import click
from picture import Picture

@click.command()
@click.option("--name", prompt = "Enter the name of picture you want to recognize")
def myfunc(name):
    picture = Picture(name)
    picture.recognition()

if __name__ == "__main__":
    myfunc()

