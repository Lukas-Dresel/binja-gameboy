from .instruction import GameBoyInstruction
from .lifter import GameBoyLifter
from .gameboy import LR35902
from .gameboyview import register_views

LR35902.register()
register_views()
