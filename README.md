# SimulationHelper

## Features

Description coming soon. maybe. maybe not.



## Usage example of basic logging functionality

```python
import sys
import torch

from torch.utils.tensorboard import SummaryWriter
from utils.Trainer import Trainer
from simulation.simulation import Simulation

if __name__ == '__main__':
    model = ...

    sim_name = f"{cfg['datasetname']}/{model.name}"
    with Simulation(sim_name=sim_name, output_root='runs') as sim:
        writer = SummaryWriter(join(sim.outdir, 'tensorboard'))

        # -------------- MAIN TRAINING LOOP  ----------------------
        for epoch in range(cfg['num_epochs']):
            trainer.do(...)
            checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
            sim.save_pytorch(checkpoint, epoch=epoch)

        print(f'\nRun {sim.outdir} finished\n')
```
