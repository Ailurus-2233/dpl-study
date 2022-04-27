from torch import nn
from rich.console import Console
from rich.table import Table

layer_type = {
    nn.Linear: "Linear",
    nn.ReLU: "ReLU",
    nn.Flatten: "Flatten",
    nn.Dropout: "Dropout",
}

layer_io = {
    nn.Linear: lambda layer: (layer.in_features, layer.out_features),
}



def show_model(model: nn.Module, console: Console = None):
    if console is None:
        console = Console()
    table = Table(title="Model")
    table.add_column("Index", justify="left")
    table.add_column("Layer", justify="left")
    table.add_column("Input", justify="left")
    table.add_column("Output", justify="left")
    index = 0
    for layer in model.net:
        index += 1
        table.add_row(str(index), layer, str(layer), str(layer))
    console.print(table)