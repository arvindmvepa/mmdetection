from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class TestPHDataset(CocoDataset):

    CLASSES = ('Wire', 'Other Component', 'Wood Structure', 'Insulator', 'Metal Crossarm', 'Wood Crossarm',
               'Concrete Structure', 'Metal Structure', 'Vegetation', 'Damper')