from typing import List

def flag_write_type(flags: List[str]) -> str:
    flgs = set(flg.lower() for flg in flags)
    return ''.join(sorted(flgs, key=lambda x: 'znch'.index(x)))

class GameBoyInstruction:
    def __init__(self, addr: int, data: bytes, mnemonic: str, length: int, operands: List[str], flag_write=List[str]) -> None:
        self.mnemonic = mnemonic
        self.length = length
        self.data = data
        self.addr = addr
        self.operand_symbols = operands
        self.flag_write = flag_write

    def __repr__(self):
        data = self.data
        length = self.length
        mnemonic = self.mnemonic
        operands = self.operand_symbols
        flag_write = self.flag_write
        return f'GameBoyInstruction(addr={hex(self.addr)}, {data=}, {length=}, {mnemonic=}, {operands=}, {flag_write})'