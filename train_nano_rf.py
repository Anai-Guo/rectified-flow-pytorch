import fire

import torchvision.transforms as T
from torch.utils.data import Dataset
from datasets import load_dataset

class OxfordFlowersDataset(Dataset):
    def __init__(
        self,
        image_size
    ):
        self.ds = load_dataset('nelorth/oxford-flowers')['train']

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.PILToTensor()
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pil = self.ds[idx]['image']
        tensor = self.transform(pil)
        return tensor / 255.

# models and trainer

from rectified_flow_pytorch import NanoFlow, Unet, Trainer, XMWrapper

def train(
    candidates = 4,
    batch_size = 4,
    grad_accum_every = 4,
    results_folder = './results',
    checkpoints_folder = './checkpoints',
    num_train_steps = 70_000,
    learning_rate = 3e-4,
    image_size = 64,
    dim = 64
):

    flowers_dataset = OxfordFlowersDataset(
        image_size = image_size
    )

    model = Unet(dim = dim)

    nano_flow = NanoFlow(
        model,
        predict_clean = True,
        times_cond_kwarg = 'times',
        normalize_data_fn = lambda t: t * 2. - 1.,
        unnormalize_data_fn = lambda t: (t + 1.) / 2.
    )

    if candidates > 1:
        nano_flow = XMWrapper(nano_flow, candidates = candidates)

    trainer = Trainer(
        nano_flow,
        dataset = flowers_dataset,
        batch_size = batch_size,
        grad_accum_every = grad_accum_every,
        num_train_steps = num_train_steps,
        learning_rate = learning_rate,
        results_folder = results_folder,
        checkpoints_folder = checkpoints_folder,
        clear_results_folder = True
    )

    trainer()

if __name__ == '__main__':
    fire.Fire(train)
