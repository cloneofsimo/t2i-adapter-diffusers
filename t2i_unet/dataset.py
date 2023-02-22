# Author : Simo Ryu
from torchvision.datasets.coco import CocoDetection

path2data = "../cocodset/val2017"
path2json = "../cocodset/annotations/instances_val2017.json"


from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms


def fourier_feature(x1, y1, w, h, n=10):
    x1, x2, y1, y2 = x1, x1 + w, y1, y1 + h
    ret = torch.zeros(n * 8)
    pii = 3.141592

    for i in range(n):
        ret[8 * i : 8 * i + 4] = torch.sin(pii * torch.tensor([x1, x2, y1, y2]))
        ret[8 * i + 4 : 8 * i + 8] = torch.cos(pii * torch.tensor([x1, x2, y1, y2]))
        pii *= 2

    return ret


class GroundedTokenDataset(Dataset):
    """
    Example usage :
    ds = GroundedTokenDataset(
        root=path2data, annFile=path2json, clip_table=torch.randn(100, 768), transform=None
    )
    """

    def __init__(self, root, annFile, clip_table, transform=None, n_ff_size=4):
        self.root = root
        self.annFile = annFile
        self.transform = transform
        self.dataset = CocoDetection(root, annFile, transform=transform)
        self.clip_table = clip_table
        self.n_ff_size = n_ff_size
        self.max_length = 6

        self.img_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]

        w, h = image.size
        image = self.img_tf(image)

        clip_embs = []
        fouriers = []

        for obj in target:

            x1, y1, w, h = obj["bbox"]
            # normalize bbox
            x1, y1, w, h = x1 / w, y1 / h, w / w, h / h

            clip_embs.append(self.clip_table[obj["category_id"]])
            fouriers.append(fourier_feature(x1, y1, w, h, n=self.n_ff_size))

        if len(clip_embs) < self.max_length:
            clip_embs += [torch.zeros(768)] * (self.max_length - len(clip_embs))
            fouriers += [torch.zeros(self.n_ff_size * 8)] * (
                self.max_length - len(fouriers)
            )
        if len(clip_embs) > self.max_length:
            clip_embs = clip_embs[: self.max_length]
            fouriers = fouriers[: self.max_length]

        return {
            "image": image,
            "clip_embs": torch.stack(clip_embs),
            "fouriers": torch.stack(fouriers),
        }
